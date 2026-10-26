# """Nguon logic dung chung (single source) cho moi notebook: build model, train, eval.
# Được nbmake.py embed vào notebook khi build — notebook tự chứa bản copy sinh tự động,
# KHÔNG sửa trực tiếp trong notebook; sửa file này rồi rebuild.
# """
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ---------------------------------------------------------------- utils

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


# ---------------------------------------------------------------- data


@dataclass
class TrainConfig:
    model_key: str = "resnet18"          # smallcnn | resnet18 | mobilenetv3s | efficientnetb0 | convnextt | asf | cmp
    num_classes: int = 45
    seed: int = 0
    epochs: int = 15
    batch_size: int = 128
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    early_stop_patience: int = 5
    amp: bool = True
    num_workers: int = 4
    input_size: int = 256
    # ASF
    asf_fixed_fusion: bool = False        # A1 (True) vs A2 (False)
    asf_feature_dim: int = 128
    # CMP
    cmp_n_proto: int = 1                  # C1/C2: 1 hoặc 2, 4
    cmp_confusion: bool = False           # C3
    cmp_conf_margin: float = 0.3
    cmp_conf_weight: float = 0.5
    cmp_diversity: bool = False           # C4
    cmp_div_weight: float = 0.01
    cmp_embed_dim: int = 512
    # chung
    label_smoothing: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ResiscDataset(Dataset):
    """Đọc ảnh theo manifest (image_id relpath từ root NWPU-RESISC45)."""

    def __init__(self, manifest: pd.DataFrame, root: str, train: bool, input_size: int = 256):
        self.items = manifest.reset_index(drop=True)
        self.root = root
        self.train = train
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.items)

    def _path(self, i: int) -> str:
        r = self.items.iloc[i]
        return os.path.join(self.root, r.image_id)

    def __getitem__(self, i: int):
        img = Image.open(self._path(i)).convert("RGB")
        img = train_augment(img, self.input_size) if self.train else eval_augment(img, self.input_size)
        return img, int(self.items.iloc[i].class_index)


def train_augment(img: Image.Image, size: int) -> Image.Image:
    """Augmentation chung (thuần PIL, tránh phụ thuộc torchvision.transforms)."""
    import torchvision.transforms.functional as TF

    # RandomResizedCrop(scale 0.7–1.0) + hflip + vflip
    w, h = img.size
    scale = random.uniform(0.7, 1.0)
    ratio = random.uniform(3 / 4, 4 / 3)
    area = w * h * scale
    cw = min(w, int(round(math.sqrt(area * ratio))))
    ch = min(h, int(round(math.sqrt(area / ratio))))
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch)).resize((size, size), Image.BILINEAR)
    if random.random() < 0.5:
        img = TF.hflip(img)
    if random.random() < 0.5:
        img = TF.vflip(img)
    return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


def eval_augment(img: Image.Image, size: int) -> "torch.Tensor":
    import torchvision.transforms.functional as TF

    img = img.resize((size, size), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


def make_loaders(manifest: pd.DataFrame, root: str, cfg: TrainConfig):
    tr = manifest[manifest.split == "train"]
    va = manifest[manifest.split == "val"]
    dl_train = DataLoader(ResiscDataset(tr, root, True, cfg.input_size), batch_size=cfg.batch_size,
                          shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    dl_val = DataLoader(ResiscDataset(va, root, False, cfg.input_size), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return dl_train, dl_val


# ---------------------------------------------------------------- baselines


def build_model(cfg: TrainConfig, pretrained: bool = True) -> nn.Module:
    """B0–B4 và 2 proposed. pretrained=False chỉ dùng cho smoke shape-check."""
    import torchvision

    if cfg.model_key == "smallcnn":
        return SmallCNN(cfg.num_classes)
    if cfg.model_key == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)
        return m
    if cfg.model_key == "mobilenetv3s":
        m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, cfg.num_classes)
        return m
    if cfg.model_key == "efficientnetb0":
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, cfg.num_classes)
        return m
    if cfg.model_key == "convnextt":
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, cfg.num_classes)
        return m
    if cfg.model_key == "asf":
        return ASFResNet18(cfg.num_classes, pretrained=pretrained, fixed_fusion=cfg.asf_fixed_fusion,
                           feature_dim=cfg.asf_feature_dim)
    if cfg.model_key == "cmp":
        return CMPResNet18(cfg.num_classes, pretrained=pretrained, n_proto=cfg.cmp_n_proto,
                           embed_dim=cfg.cmp_embed_dim, confusion=cfg.cmp_confusion,
                           conf_margin=cfg.cmp_conf_margin, diversity=cfg.cmp_diversity)
    raise ValueError(f"unknown model_key: {cfg.model_key}")


class SmallCNN(nn.Module):
    """B0 — CNN nhỏ train từ đầu, cấu trúc hợp lý (3 block, BN, GAP)."""

    def __init__(self, num_classes: int = 45):
        super().__init__()
        ch = [3, 32, 64, 128]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                       nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(F.adaptive_avg_pool2d(self.features(x), 1))


# ---------------------------------------------------------------- ASF-ResNet18


def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Haar DWT 2D level-1 trên tensor (B,C,H,W), H/W chẵn (crop trước nếu lẻ).
    Trả về (LL, LH, HL, HH) — mỗi cái (B,C,H/2,W/2)."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = x[..., : x.shape[-2] // 2 * 2, : x.shape[-1] // 2 * 2]
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) / 4.0
    lh = (a + b - c - d) / 4.0
    hl = (a - b + c - d) / 4.0
    hh = (a - b - c + d) / 4.0
    return ll, lh, hl, hh


class ASFResNet18(nn.Module):
    """Adaptive Scale–Frequency ResNet18.

    - Giữ backbone ResNet18 pretrained NGUYÊN VẸN (không thay conv nào).
    - Lấy feature map layer2 (128ch, 32×32) và layer4 (512ch, 16×16) → Haar DWT → 8 thành phần.
    - Mỗi thành phần: 1×1 conv → F dims → BN+ReLU → GAP → vector f_i.
    - Gate (theo ảnh): MLP(8F→8) softmax; hoặc fixed_fusion=True: softmax over learnable logits (A1).
    - Fused = Σ w_i f_i → Linear zero-init (F→512) → residual cộng vào GAP(layer4) → head.
    - Zero-init residual ⇒ lúc đầu model ≡ ResNet18 (bảo toàn biểu diễn pretrained).
    """

    def __init__(self, num_classes: int = 45, pretrained: bool = True, fixed_fusion: bool = False,
                 feature_dim: int = 128):
        super().__init__()
        import torchvision

        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.fixed_fusion = fixed_fusion
        self.feature_dim = feature_dim
        self.n_components = 8  # 2 stage × 4 subband
        F_ = feature_dim
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, F_, 1, bias=False), nn.BatchNorm2d(F_),
                                                 nn.ReLU(inplace=True)) for c in (128, 128, 128, 128, 512, 512, 512, 512)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        if fixed_fusion:
            self.fixed_logits = nn.Parameter(torch.zeros(self.n_components))
        else:
            self.gate = nn.Linear(F_ * self.n_components, self.n_components)
        self.out_proj = nn.Linear(F_, 512)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h1 = self.layer1(h)
        f2 = self.layer2(h1)     # 128ch
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)     # 512ch
        feats = []
        for i, fm in enumerate((f2, f2, f2, f2, f4, f4, f4, f4)):
            # thành phần i%4==0: LL ≡ full feature map trước khi tách (thông tin low-freq tổng thể);
            # i%4∈{1,2,3}: LH/HL/HH từ Haar DWT level-1
            sub = fm if i % 4 == 0 else haar_dwt2(fm)[i % 4]
            feats.append(self.pool(self.proj[i](sub)).flatten(1))  # (B, F)
        stack = torch.stack(feats, dim=1)                          # (B, 8, F)
        if self.fixed_fusion:
            w = torch.softmax(self.fixed_logits, dim=0)
            w = w.view(1, -1, 1).expand_as(stack)
        else:
            w = torch.softmax(self.gate(stack.flatten(1)), dim=-1).unsqueeze(-1)  # (B, 8, 1)
        fused = (stack * w).sum(dim=1)                             # (B, F)
        backbone_vec = self.pool(f4).flatten(1)                    # (B, 512)
        return self.head(backbone_vec + self.out_proj(fused))

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về (B, 8) để phân tích phân bố gate (chỉ dùng khi fixed_fusion=False)."""
        assert not self.fixed_fusion
        was_training = self.training
        self.eval()
        with torch.no_grad():
            h = self.stem(x)
            h1 = self.layer1(h)
            f2 = self.layer2(h1)
            f4 = self.layer4(self.layer3(f2))
            feats = []
            for i, fm in enumerate((f2, f2, f2, f2, f4, f4, f4, f4)):
                sub = fm if i % 4 == 0 else haar_dwt2(fm)[i % 4]
                feats.append(self.pool(self.proj[i](sub)).flatten(1))
            w = torch.softmax(self.gate(torch.stack(feats, 1).flatten(1)), -1)
        if was_training:
            self.train()
        return w


# ---------------------------------------------------------------- CMP-ResNet18


class CMPResNet18(nn.Module):
    """Confusion-guided Multi-Prototype ResNet18.

    - Backbone ResNet18 pretrained → GAP → embed Linear → L2 normalize.
    - K prototype/lớp (L2 normalize) → sim (B, C, K) → logits = logsumexp(κ·sim, K) − log K
      (chuẩn hóa theo K, κ = logit_scale học được, init ln(10)).
    - Confusion margin (có hướng, chỉ trong loss): target=a → logit_b −= m·w[a,b] với (a,b)
      lấy từ confusion graph train-only; w ∈ [0,1].
    - Diversity reg (tùy chọn): mean max-cos giữa prototype cùng lớp.
    """

    def __init__(self, num_classes: int = 45, pretrained: bool = True, n_proto: int = 1,
                 embed_dim: int = 512, confusion: bool = False, conf_margin: float = 0.3,
                 diversity: bool = False):
        super().__init__()
        import torchvision

        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(bb.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(512, embed_dim)
        self.C, self.K = num_classes, n_proto
        self.prototypes = nn.Parameter(torch.randn(num_classes * n_proto, embed_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.confusion = confusion
        self.conf_margin = conf_margin
        self.diversity = diversity
        self.register_buffer("conf_w", torch.zeros(num_classes, num_classes))  # set_confusion() ghi vào đây

    def set_confusion(self, w: np.ndarray) -> None:
        assert w.shape == (self.C, self.C) and float(w.min()) >= 0.0 and float(w.max()) <= 1.0
        self.conf_w.copy_(torch.from_numpy(w.astype(np.float32)))

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.backbone(x)).flatten(1)
        return F.normalize(self.embed(z), dim=-1)

    def logits_from_emb(self, e: torch.Tensor, apply_margin: bool = False, labels: torch.Tensor | None = None) -> torch.Tensor:
        p = F.normalize(self.prototypes, dim=-1)
        sim = (e @ p.T).view(-1, self.C, self.K)
        kappa = self.logit_scale.clamp(max=math.log(100.0))
        logits = torch.logsumexp(sim * kappa, dim=2) - math.log(self.K)
        if apply_margin and self.confusion and labels is not None:
            labels = labels.to(self.conf_w.device)      # chống lệch device giữa labels và buffer
            w = self.conf_w[labels]                     # (B, C): w[a, b] trên hàng a
            w = w * (1.0 - F.one_hot(labels, self.C).float())
            logits = logits - self.conf_margin * w
        return logits

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.logits_from_emb(self.embeddings(x), apply_margin=True, labels=labels)

    def diversity_loss(self) -> torch.Tensor:
        p = F.normalize(self.prototypes, dim=-1).view(self.C, self.K, -1)
        if self.K < 2:
            return torch.zeros((), device=p.device)
        cos = torch.bmm(p, p.transpose(1, 2))       # (C, K, K)
        eye = torch.eye(self.K, device=p.device).unsqueeze(0)
        return ((cos - eye).abs().sum(dim=(1, 2)) / (self.K * (self.K - 1))).mean()

    def prototype_usage(self, loader: DataLoader, device: str, max_batches: int | None = None) -> dict:
        """Thống kê usage/khoảng cách prototype (phát hiện collapse)."""
        self.eval()
        assign = torch.zeros(self.C, self.K, dtype=torch.long)
        n_seen = 0
        with torch.no_grad():
            for bi, (x, y) in enumerate(loader):
                if max_batches and bi >= max_batches:
                    break
                e = self.embeddings(x.to(device))
                p = F.normalize(self.prototypes, dim=-1).view(self.C, self.K, -1)
                pc = p[y.to(p.device)]               # (B, K, D) prototype của lớp đúng — index cùng device
                sim = torch.einsum("bd,bkd->bk", e, pc)
                for cls, proto in zip(y.tolist(), sim.argmax(1).cpu().tolist()):
                    assign[cls, proto] += 1
                n_seen += len(y)
        inter = []
        with torch.no_grad():
            p = F.normalize(self.prototypes, dim=-1).view(self.C, self.K, -1).cpu()  # phân tích nhỏ → CPU
            if self.K > 1:
                cos = torch.bmm(p, p.transpose(1, 2))
                eye = torch.eye(self.K).bool().unsqueeze(0)
                inter = cos.masked_fill(eye, 0).amax(dim=(1, 2)).tolist()
        return {"n_seen": n_seen, "assign_per_class_proto": assign.tolist(),
                "unused_proto_classes": int((assign.sum(1) == 0).sum()),
                "max_inter_proto_cos_mean": float(np.mean(inter)) if inter else None}


# ---------------------------------------------------------------- data staging


def stage_dataset(manifest: pd.DataFrame, src_root: str, dst_root: str, num_threads: int = 16,
                  log_every: int = 2000) -> str:
    """Copy ảnh từ mount FUSE (/kaggle/input — ~17-35 ảnh/s đọc lẻ) sang disk local MỘT lần,
    song song theo thread (I/O-bound). Trả về dst_root. Verify: đếm + sample MD5 khớp nguồn.
    Kernel huấn luyện đọc từ dst_root sau đó — epoch không còn chờ mạng."""
    import hashlib
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(dst_root, exist_ok=True)
    ids = manifest.image_id.tolist()

    def copy_one(rel: str) -> tuple[str, str]:
        dst = os.path.join(dst_root, rel)
        d = os.path.dirname(dst)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src_root, rel), dst)
        with open(dst, "rb") as f:
            return rel, hashlib.md5(f.read()).hexdigest()

    t0 = time.time()
    hashes = {}
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for k, (rel, h) in enumerate(ex.map(copy_one, ids), start=1):
            hashes[rel] = h
            if k % log_every == 0 or k == len(ids):
                el = time.time() - t0
                print(f"[stage] {k}/{len(ids)} ({100*k/len(ids):.0f}%) | {el:.0f}s | "
                      f"{k/el:.0f}/s | ETA {el/k*(len(ids)-k):.0f}s", flush=True)
    # verify: đếm + sample 100 MD5 đối chiếu nguồn
    staged = sum(len(files) for _, _, files in os.walk(dst_root))
    assert staged == len(ids), f"staged {staged} != {len(ids)}"
    sample = ids[:: max(len(ids) // 100, 1)][:100]
    for rel in sample:
        with open(os.path.join(src_root, rel), "rb") as f:
            src_h = hashlib.md5(f.read()).hexdigest()
        assert hashes[rel] == src_h, f"MD5 lệch sau stage: {rel}"
    print(f"[stage] VERIFY PASS: {staged} file, {len(sample)} MD5 sample khớp nguồn | "
          f"tổng {time.time()-t0:.0f}s")
    return dst_root


# ---------------------------------------------------------------- confusion graph (train-only)


def build_confusion_graph(train_manifest: pd.DataFrame, root: str, graph_frac: float = 0.10,
                          seed: int = 0, probe_epochs: int = 6, device: str = "cuda",
                          min_count: int = 3, num_classes: int = 45,
                          num_workers: int = 4) -> tuple[np.ndarray, dict]:
    """Probe model train trên (1−graph_frac) của TRAIN, dự đoán trên phần graph còn lại (cùng train).
    KHÔNG dùng val/test. Trả về w (C,C) có hướng: w[a,b] = n_ab / max_row, cặp n_ab < min_count → 0.
    """
    rng = np.random.default_rng(seed)
    fit_df, graph_df = [], []
    for cls, gdf in train_manifest.groupby("class_name"):
        idx = rng.permutation(len(gdf))
        n_graph = max(1, int(round(len(gdf) * graph_frac)))
        take = gdf.iloc[idx[:n_graph]]
        graph_df.append(take)
        fit_df.append(gdf.iloc[idx[n_graph:]])
    fit_df, graph_df = pd.concat(fit_df), pd.concat(graph_df)

    pc = TrainConfig(model_key="resnet18", num_classes=num_classes, epochs=probe_epochs, seed=seed,
                     num_workers=num_workers)
    probe = build_model(pc)
    dl_fit = DataLoader(ResiscDataset(fit_df, root, True, pc.input_size), batch_size=pc.batch_size, shuffle=True,
                        num_workers=num_workers)
    dl_graph = DataLoader(ResiscDataset(graph_df, root, False, pc.input_size), batch_size=pc.batch_size, shuffle=False,
                          num_workers=num_workers)
    probe = train_model(probe, pc, dl_fit, None, device, verbose=False)[0]  # không early-stop (val None)

    n = np.zeros((num_classes, num_classes), dtype=np.float64)
    probe.eval()
    with torch.no_grad():
        for x, y in dl_graph:
            pred = probe(x.to(device)).argmax(1).cpu().numpy()
            for a, b in zip(y.numpy(), pred):
                n[a, b] += 1
    off = n.copy()
    np.fill_diagonal(off, 0.0)
    row_max = off.max(axis=1, keepdims=True)
    w = np.where((off >= min_count) & (row_max > 0), off / np.maximum(row_max, 1e-9), 0.0)
    info = {"probe_epochs": probe_epochs, "n_fit": int(len(fit_df)), "n_graph": int(len(graph_df)),
            "n_pairs_active": int((w > 0).sum()), "max_n": int(off.max()),
            "confusion_rate_graph": float(off.sum() / max(len(graph_df), 1))}
    return w.astype(np.float32), info


# ---------------------------------------------------------------- train / eval


def _param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    """Backbone lr thấp hơn head lr — pretrained conv là backbone (mọi model đều có .backbone/.stem...)."""
    head_ids = set()
    for name, _p in model.named_parameters():
        if any(k in name for k in ("head", "fc", "classifier", "embed", "prototypes", "logit_scale",
                                   "proj", "gate", "out_proj", "fixed_logits")):
            head_ids.add(name)
    head, backbone = [], []
    for name, p in model.named_parameters():
        (head if name in head_ids else backbone).append(p)
    return [{"params": backbone, "lr": cfg.lr_backbone}, {"params": head, "lr": cfg.lr_head}]


def train_model(model: nn.Module, cfg: TrainConfig, dl_train: DataLoader, dl_val: DataLoader | None,
                device: str, verbose: bool = True, state_out: str | None = None) -> tuple[nn.Module, dict]:
    """Train với AMP + cosine + warmup + early stop (val macro-F1). Trả về (best_model, history)."""
    model.to(device)
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(dl_train), 1)
    total_steps = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history, best = [], {"f1": -1.0, "state": None, "epoch": -1}
    t_train0 = time.time()
    bad_epochs = 0
    for ep in range(cfg.epochs):
        model.train()
        ep_loss, ep_n, t0 = 0.0, 0, time.time()
        n_batches = len(dl_train)
        for bi, (x, y) in enumerate(dl_train, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.startswith("cuda")):
                logits = model(x, y) if isinstance(model, CMPResNet18) else model(x)
                loss = crit(logits, y)
                if isinstance(model, CMPResNet18) and model.diversity:
                    loss = loss + cfg.cmp_div_weight * model.diversity_loss()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_loss += loss.item() * len(y)
            ep_n += len(y)
            if bi % 50 == 0 or bi == n_batches:  # progress mỗi 50 batch — xem live trên Kaggle
                el = time.time() - t0
                print(f"[train] ep {ep} batch {bi}/{n_batches} | {el:.0f}s | "
                      f"loss={(ep_loss / max(ep_n, 1)):.3f} | ETA {el / bi * (n_batches - bi):.0f}s", flush=True)
        entry = {"epoch": ep, "train_loss": ep_loss / max(ep_n, 1), "epoch_sec": round(time.time() - t0, 2),
                 "lr": sched.get_last_lr()[0]}
        if dl_val is not None:
            m = evaluate(model, dl_val, device)
            entry.update({"val_acc": m["acc"], "val_macro_f1": m["macro_f1"]})
            if m["macro_f1"] > best["f1"]:
                best.update({"f1": m["macro_f1"], "epoch": ep,
                             "state": copy.deepcopy(model.state_dict())})
                bad_epochs = 0
                if state_out:
                    torch.save({"state_dict": best["state"], "config": cfg.to_dict(), "epoch": ep}, state_out)
            else:
                bad_epochs += 1
        history.append(entry)
        if verbose:
            print(json.dumps(entry))
        if dl_val is not None and bad_epochs >= cfg.early_stop_patience:
            if verbose:
                print(f"early stop @ epoch {ep} (best {best['epoch']})")
            break
    train_time = time.time() - t_train0
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    hist_summary = {"train_time_sec": round(train_time, 1), "executed_epochs": len(history),
                    "best_epoch": best["epoch"], "best_val_macro_f1": None if best["f1"] < 0 else best["f1"],
                    "history": history}
    return model, hist_summary


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    """Full metrics: acc + macro P/R/F1 + per-class + confusion matrix."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    model.to(device).eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            logits = model(x.to(device))
        ps.append(logits.argmax(1).cpu())
        ys.append(y)
    y_true, y_pred = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pp, rr, ff, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {"acc": float((y_true == y_pred).mean()), "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f1),
            "per_class": {"precision": pp.tolist(), "recall": rr.tolist(), "f1": ff.tolist(), "support": sup.tolist()},
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(), "n": int(len(y_true))}


@torch.inference_mode()
def measure_latency(model: nn.Module, device: str, input_size: int = 256, n_warmup: int = 10,
                    n_iter: int = 50) -> dict:
    """Latency model-only, batch 1, median/p95 (ms). model.eval() + cuda sync + warmup."""
    model.to(device).eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        if device.startswith("cuda"):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            model(x)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        else:
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "n_iter": n_iter, "batch": 1}
