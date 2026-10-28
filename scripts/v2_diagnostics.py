#!/usr/bin/env python3
"""V2 diagnostics — kiểm chứng các nghi vấn từ review (mục 4-6 spec v2).

Chạy local CPU với checkpoint v1 thật + ví dụ tổng hợp. Mọi kết luận in ra với đủ chữ số.
Output: notes/v2_diagnostics.json + stdout.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import CMPResNet18, ASFResNet18, TrainConfig, build_model, ResiscDataset  # noqa: E402

device = "cpu"
OUT: dict = {}

# ================================================================
print("=" * 70)
print("D1. MARGIN DIRECTION (4.2) — margin hiện tại tác động CE thế nào?")
print("=" * 70)
# logits tổng hợp: target=0, negative dễ nhầm class=1 (w=1.0), khác class=2 (w=0)
z = torch.tensor([[2.0, 1.8, 0.5]])  # model đang gần nhầm về class 1
y = torch.tensor([0])
conf_w = torch.zeros(3, 3)
conf_w[0, 1] = 1.0
m = 0.3

ce_before = F.cross_entropy(z, y)
# cơ chế v1: trừ margin khỏi logit negative
z_sup = z.clone()
w = conf_w[y]
w = w * (1.0 - F.one_hot(y, 3).float())
z_sup = z_sup - m * w
ce_after = F.cross_entropy(z_sup, y)

# gradient theo logit lớp đúng z[0] trước/sau
z_g = z.clone().requires_grad_(True)
F.cross_entropy(z_g, y).backward()
grad_correct_before = float(z_g.grad[0, 0])
z_g2 = (z - m * w).clone().requires_grad_(True)
F.cross_entropy(z_g2, y).backward()
grad_correct_after = float(z_g2.grad[0, 0])

# gradient theo logit lớp khó z[1] trước/sau (mất đạo hàm hay còn?)
z_g3 = z.clone().requires_grad_(True)
loss3 = F.cross_entropy(z_g3, y)
loss3.backward()
grad_hard_before = float(z_g3.grad[0, 1])
z_g4 = (z - m * w).clone().requires_grad_(True)
F.cross_entropy(z_g4, y).backward()
grad_hard_after = float(z_g4.grad[0, 1])

print(f"CE trước margin-suppress: {ce_before:.6f}")
print(f"CE sau  margin-suppress: {ce_after:.6f}  (Δ = {ce_after - ce_before:+.6f})")
print(f"grad w.r.t. logit lớp đúng: trước {grad_correct_before:.6f} → sau {grad_correct_after:.6f}")
print(f"grad w.r.t. logit lớp khó : trước {grad_hard_before:.6f} → sau {grad_hard_after:.6f}")
print(f"p(correct) trước {torch.softmax(z, 1)[0, 0]:.6f} → sau {torch.softmax(z_sup, 1)[0, 0]:.6f}")
OUT["D1_margin"] = {"ce_before": ce_before.item(), "ce_after": ce_after.item(),
                    "grad_correct_before": grad_correct_before, "grad_correct_after": grad_correct_after,
                    "grad_hard_before": grad_hard_before, "grad_hard_after": grad_hard_after}
print("→ KẾT LUẬN: margin v1 GIẢM CE (làm bài tập dễ hơn trên mẫu khó) và GIẢM gradient")
print("  trên mẫu khó → KHÔNG phải discriminative margin kiểu CosFace.")

# ================================================================
print()
print("=" * 70)
print("D2. PROTOTYPE COLLAPSE — tái hiện + metric check + gradient (4.1)")
print("=" * 70)
torch.manual_seed(0)
# synthetic: 1 lớp, embedding 8-dim cố định (đơn vị hóa), K=2 prototype
E = F.normalize(torch.randn(64, 8), dim=1)  # 64 "ảnh" cùng lớp
protos = nn.Parameter(torch.cat([F.normalize(E[:1] + 0.5 * F.normalize(torch.randn(1, 8), dim=1),
                                                dim=1).repeat(1, 1),
                                 F.normalize(E[:1] - 0.5 * F.normalize(torch.randn(1, 8), dim=1),
                                                dim=1).repeat(1, 1)]))  # 2 hướng KHÁC nhau
kappa = 10.0
opt = torch.optim.Adam([protos], lr=0.05)

def inter_cos(p: torch.Tensor) -> float:
    a = F.normalize(p[0], dim=0)
    b = F.normalize(p[1], dim=0)
    return float((a * b).sum())

def lse_logits(e: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    sim = e @ F.normalize(p, dim=-1).T  # (B, K)
    return torch.logsumexp(sim * kappa, dim=1) - math.log(p.shape[0])

hist = []
neg = torch.zeros(len(E), 1)  # lớp negative cố định (logit 0) — cho CE có ý nghĩa
for step in range(200):
    opt.zero_grad()
    logits = torch.cat([lse_logits(E, protos).unsqueeze(1), neg], dim=1)  # (B, 2)
    ce = F.cross_entropy(logits, torch.zeros(len(E), dtype=torch.long))
    ce.backward()
    hist.append({"step": step, "inter_cos": inter_cos(protos),
                 "gnorm": float(protos.grad.norm())})
    opt.step()
print(f"inter-cos khởi đầu: {hist[0]['inter_cos']:.6f} → sau 200 step LSE-only: {hist[-1]['inter_cos']:.6f}")
print(f"gradient norm cuối: {hist[-1]['gnorm']:.6e}")
collapse_reproduced = hist[-1]["inter_cos"] > 0.999

# với diversity penalty (weight 0.01 như v1) và weight mạnh hơn
for w_div in (0.01, 0.2):
    torch.manual_seed(0)
    protos2 = nn.Parameter(torch.cat([F.normalize(E[:1] + 0.5 * F.normalize(torch.randn(1, 8), dim=1), dim=1).repeat(1, 1),
                                      F.normalize(E[:1] - 0.5 * F.normalize(torch.randn(1, 8), dim=1), dim=1).repeat(1, 1)]))
    opt2 = torch.optim.Adam([protos2], lr=0.05)
    for step in range(200):
        opt2.zero_grad()
        logits = torch.cat([lse_logits(E, protos2).unsqueeze(1), neg], dim=1)
        loss = F.cross_entropy(logits, torch.zeros(len(E), dtype=torch.long))
        p = F.normalize(protos2, dim=-1)
        cos = torch.bmm(p.unsqueeze(0), p.unsqueeze(0).transpose(1, 2))[0]
        eye = torch.eye(2)
        loss = loss + w_div * ((cos - eye).abs().sum() / 2)
        loss.backward()
        opt2.step()
    print(f"  + diversity w={w_div}: inter-cos sau 200 step = {inter_cos(protos2):.6f}")
    OUT.setdefault("D2_diversity", {})[str(w_div)] = inter_cos(protos2)

OUT["D2_collapse_reproduced_lse_only"] = collapse_reproduced
OUT["D2_inter_cos_start"] = hist[0]["inter_cos"]
print(f"→ collapse tái hiện được với LSE-only (không cần lỗi code): {collapse_reproduced}")

# ================================================================
print()
print("=" * 70)
print("D3. CHECKPOINT CMP v1 THẬT — FP32 diagnostics (4.1)")
print("=" * 70)
data_root = ROOT / "data" / "raw" / "punfake---resisc45" / "NWPU-RESISC45"
ck = torch.load(OUT_P / "ckpt_CMP_C2_s1.pt", map_location=device, weights_only=False) if (OUT_P := ROOT / "outputs" / "resisc45-final-proposed-asf-cmp-3-seeds") else None
model = build_model(TrainConfig(num_classes=45, model_key="cmp", cmp_n_proto=2), pretrained=False)
model.load_state_dict(ck["state_dict"])
model.eval()
P = model.prototypes.data.detach().float()  # (90, 512)
print(f"prototype tensor: shape={tuple(P.shape)} | dtype={P.dtype} | "
      f"requires_grad_in_ckpt={ck['state_dict']['prototypes'].requires_grad}")
Pn = F.normalize(P, dim=-1)
print(f"norm trung bình trước normalize: {P.norm(dim=1).mean():.6f} (min {P.norm(dim=1).min():.4f} max {P.norm(dim=1).max():.4f})")
Pv = Pn.view(45, 2, -1)
cos_pair = (Pv[:, 0] * Pv[:, 1]).sum(-1)           # cosine 2 prototype CÙNG lớp
euc = (Pv[:, 0] - Pv[:, 1]).norm(dim=1)            # euclidean sau normalize
print(f"same-class pair cos: mean={cos_pair.mean():.8f} min={cos_pair.min():.8f} max={cos_pair.max():.8f}")
print(f"same-class euclidean (sau norm): mean={euc.mean():.8f} max={euc.max():.8f}")
print(f"số lớp có pair-cos == 1.0 (sai số <1e-6): {(cos_pair > 1 - 1e-6).sum().item()}/45")
print(f"số lớp có pair-cos > 0.999: {(cos_pair > 0.999).sum().item()}/45")
# cross-class cos (đối chứng metric)
Pc = Pv.mean(dim=1)  # center mỗi lớp
Pcn = F.normalize(Pc, dim=-1)
cross = (Pcn @ Pcn.T)
cross.fill_diagonal_(float("nan"))
cross_clean = torch.nan_to_num(cross, nan=0.0)
n_off = 45 * 44
print(f"cross-class center cos: mean={cross_clean.sum() / n_off:.6f} max={cross_clean.max():.6f}")

# assignment trên subset train thật
man = pd.read_csv(ROOT / "outputs" / "resisc45-final-evaluation-v2-offline" / "test_predictions.csv")  # chỉ để lấy class list
tr = []
for c in sorted(data_root.iterdir())[:10]:  # 10 lớp × 30 ảnh — đủ cho assignment
    for f in sorted(c.glob("*.jpg"))[:30]:
        tr.append((str(c.name) + "/" + f.name, c.name))
sub = pd.DataFrame(tr, columns=["image_id", "class_name"])
ds = ResiscDataset(sub.assign(class_index=sub.class_name.astype("category").cat.codes), str(data_root), train=False)
dl = torch.utils.data.DataLoader(ds, batch_size=64)
assign = torch.zeros(45, 2, dtype=torch.long)
with torch.no_grad():
    for x, y in dl:
        e = model.embeddings(x)
        sim = (e @ F.normalize(P, dim=-1).T).view(-1, 45, 2)
        a = sim.argmax(-1)  # (B, 45) — prototype được chọn cho MỖI lớp
        for b in range(len(y)):
            assign[y[b], a[b]] += 1
dead = int((assign.sum(1) == 0).sum())
minority = float(np.mean([min(r) / max(sum(r), 1) for r in assign.numpy()]))
print(f"assignment (10 lớp đầu × 30 ảnh): dead classes={dead}/45 | "
      f"tỉ lệ prototype-thiểu-số trung bình: {minority:.4f} (0 = collapse hoàn toàn, 0.5 = cân bằng)")
OUT["D3_ckpt"] = {"pair_cos_mean": float(cos_pair.mean()), "pair_cos_min": float(cos_pair.min()),
                  "pair_cos_max": float(cos_pair.max()), "n_identical": int((cos_pair > 1 - 1e-6).sum()),
                  "dead_classes": dead, "minority_ratio": minority,
                  "euclidean_mean": float(euc.mean())}

# D2b: hard-max (CMP-P) vs LSE với init GIỐNG v1 (randn*0.02) — kiểm chứng parallel drift
print()
print("--- D2b: cơ chế aggregation × init (synthetic, κ=10 cố định) ---")

def run_variant(agg: str, init: str, steps: int = 200) -> float:
    torch.manual_seed(0)
    E = F.normalize(torch.randn(64, 8), dim=1)
    if init == "v1_random":
        p0 = torch.randn(2, 8) * 0.02
    else:
        p0 = torch.cat([F.normalize(E[:1] + 0.5 * F.normalize(torch.randn(1, 8), dim=1), dim=1),
                        F.normalize(E[:1] - 0.5 * F.normalize(torch.randn(1, 8), dim=1), dim=1)])
    p = nn.Parameter(p0)
    opt = torch.optim.Adam([p], lr=0.05)
    neg = torch.zeros(len(E), 1)
    for _ in range(steps):
        opt.zero_grad()
        pn = F.normalize(p, dim=-1)
        sim = E @ pn.T  # (B, 2)
        if agg == "lse":
            z = torch.logsumexp(sim * kappa, dim=1, keepdim=True) - math.log(2)
        else:  # hard-max
            z = (sim.max(dim=1, keepdim=True).values * kappa)
        loss = F.cross_entropy(torch.cat([z, neg], 1), torch.zeros(len(E), dtype=torch.long))
        loss.backward()
        opt.step()
    a = F.normalize(p[0], dim=0)
    b = F.normalize(p[1], dim=0)
    return float((a * b).sum())

for agg in ("lse", "hardmax"):
    for init in ("v1_random", "separated"):
        ic = run_variant(agg, init)
        print(f"  agg={agg:<8} init={init:<10} → inter-cos = {ic:+.6f}")
        OUT.setdefault("D2b_matrix", {})[f"{agg}_{init}"] = ic

# ================================================================
print()
print("=" * 70)
print("D4. CHECKPOINT ASF v1 THẬT — gate, residual, tương đương B1? (6)")
print("=" * 70)
ck_a = torch.load(OUT_P / "ckpt_ASF_A2fix_ent_s1.pt", map_location=device, weights_only=False)
asf = build_model(TrainConfig(num_classes=45, model_key="asf", asf_out_init="small", asf_gate_ent=0.05), pretrained=False)
asf.load_state_dict(ck_a["state_dict"])
asf.eval()
import torchvision  # noqa: E402

b1 = torchvision.models.resnet18(weights=None)
b1.fc = nn.Linear(512, 45)
sd_b1 = torch.load(OUT_B := ROOT / "outputs" / "resisc45-final-baselines-b0-b4-3-seeds" / "ckpt_resnet18_s1.pt", map_location=device, weights_only=False)
b1.load_state_dict(sd_b1["state_dict"])
b1.eval()

ds_asf = ResiscDataset(sub.assign(class_index=sub.class_name.astype("category").cat.codes), str(data_root), train=False)
xs = torch.stack([ds_asf[i][0] for i in range(min(64, len(ds_asf)))])

with torch.no_grad():
    # gate stats
    h = asf.stem(xs)
    h1 = asf.layer1(h)
    f2 = asf.layer2(h1)
    f4 = asf.layer4(asf.layer3(f2))
    feats = asf._component_feats(f2, f4)
    stack = torch.stack(feats, 1)
    gw = torch.softmax(asf.gate(stack.flatten(1)), -1)
    fused = (stack * gw.unsqueeze(-1)).sum(1)
    res_vec = asf.out_proj(fused)
    bb_vec = asf.pool(f4).flatten(1)
    ratio = (res_vec.norm(dim=1) / bb_vec.norm(dim=1))
    # subband magnitudes (trước proj) — signed GAP vs RMS từng thành phần
    from models import haar_dwt2
    stats = []
    for i, fm in enumerate((f2, f2, f2, f2, f4, f4, f4, f4)):
        if i % 4 == 0:
            sub = fm
        else:
            sub = haar_dwt2(fm)[i % 4]
        signed_gap = sub.mean(dim=(2, 3)).abs().mean()
        energy = sub.pow(2).mean(dim=(2, 3)).sqrt().mean()
        stats.append({"component": ["LL2", "LH2", "HL2", "HH2", "LL4", "LH4", "HL4", "HH4"][i],
                      "abs_signed_gap": float(signed_gap), "rms": float(energy)})
    # gate→backbone equivalence: ASF vs B1 logits trên cùng input
    d_asf_b1 = float((asf(xs) - b1(xs)).abs().max())
    # one-hot LL4 thay gate → logits so B1
    w_oh = torch.zeros(len(xs), 8)
    w_oh[:, 4] = 1.0
    fused_oh = (stack * w_oh.unsqueeze(-1)).sum(1)
    logits_oh = asf.head(bb_vec + asf.out_proj(fused_oh))
    d_oh_b1 = float((logits_oh - b1(xs)).abs().max())

print(f"gate mean (seed1, 64 ảnh): {[round(v, 4) for v in gw.mean(0).tolist()]}")
print(f"gate std across images   : {[round(v, 5) for v in gw.std(0).tolist()]}")
print(f"residual norm / backbone norm: mean={ratio.mean():.4f} min={ratio.min():.4f} max={ratio.max():.4f}")
print(f"signed-GAP vs RMS theo thành phần: {[(s['component'], round(s['abs_signed_gap'], 3), round(s['rms'], 3)) for s in stats]}")
print(f"max|logit(ASF) − logit(B1)| trên 64 ảnh: {d_asf_b1:.4f}")
print(f"max|logit(ASF one-hot LL4) − logit(B1)|: {d_oh_b1:.4f}")
OUT["D4_asf"] = {"gate_mean": gw.mean(0).tolist(), "gate_std": gw.std(0).tolist(),
                 "residual_backbone_norm_ratio_mean": float(ratio.mean()),
                 "residual_backbone_norm_ratio_max": float(ratio.max()),
                 "max_logit_diff_asf_vs_b1": d_asf_b1, "max_logit_diff_onehotLL4_vs_b1": d_oh_b1,
                 "subband_stats": stats}

with open(ROOT / "notes" / "v2_diagnostics.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nSaved → notes/v2_diagnostics.json")
