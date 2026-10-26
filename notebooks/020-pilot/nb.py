# %% [md]
# # 020 — Pilot: đo thời gian, kiểm forward/backward 2 proposed
#
# **Mục tiêu (giai đoạn B):** trên Kaggle GPU thật, đo các số liệu để chốt `compute_budget.md`:
# 1. ResNet18 (B1): s/epoch full train, GPU memory, val accuracy sau 2 epoch (sanity).
# 2. ASF: kiểm khởi tạo ≡ ResNet18, forward/backward, 1 epoch trên subset, latency, gate hoạt động.
# 3. CMP: forward/backward, margin hoạt động đúng chiều, 1 epoch subset, prototype stats.
# 4. Latency batch-1 mọi model chính.
# 5. Xuất `pilot_report.json`.
#
# **Ràng buộc:** CHỈ dùng train/val. Không đụng test. Không smoke metrics = final metrics.
# **Nguồn:** dataset `punfake/resisc45` + split manifest từ kernel `resisc45-data-audit` (kernel_sources).

# %%
# ---- CONFIG ----
import os

RUN_ID = "020-pilot-v1"
MANIFEST = None  # resolve ở cell sau
DATA_ROOT = None
SUBSET_FRAC = 0.2     # phần train dùng cho run ngắn ASF/CMP
PILOT_EPOCHS_FULL = 2 # ResNet18 chạy đủ train
PILOT_EPOCHS_SUB = 1
SEED = 0

# %%
# ---- ENV + resolve paths + manifest ----
import glob
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"run_id: {RUN_ID} | device: {device}")
print("python:", sys.version.split()[0], "| torch:", torch.__version__, "| cuda:", torch.version.cuda)
if device == "cuda":
    print("gpu:", torch.cuda.get_device_name(0), "| mem:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")

def find_single(pattern_list: list[str], desc: str) -> str:
    for pat in pattern_list:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise AssertionError(f"FAIL-FAST: khong tim thay {desc}: {pattern_list}")

manifest_path = find_single(
    ["/kaggle/input/resisc45-data-audit/split_manifest.csv",
     "../../notebooks/010-data-audit/split_manifest.smoke.csv",
     "split_manifest.smoke.csv"],
    "split manifest")
DATA_ROOT = find_single(
    ["/kaggle/input/resisc45/NWPU-RESISC45",
     "../../data/raw/punfake---resisc45/NWPU-RESISC45"],
    "dataset root")
manifest = pd.read_csv(manifest_path)
manifest = manifest[manifest.split != "test"].reset_index(drop=True)  # pilot: KHÔNG test
print("manifest:", manifest_path, "| rows (train+val):", len(manifest))
print("DATA_ROOT:", DATA_ROOT)
audit_path = find_single(
    ["/kaggle/input/resisc45-data-audit/audit_report.json",
     "../../notebooks/010-data-audit/audit_report.smoke.json"],
    "audit report")
with open(audit_path) as f:
    audit = json.load(f)
print("audit run_id:", audit["run_id"], "| smoke:", audit["smoke"], "| split sha256:", audit["split_sha256"][:16])

# %% embed:src/models.py

# %%
# ---- Dataset/loaders kiểm tra tốc độ data loader ----
set_seed(SEED)
full_train = manifest[manifest.split == "train"].reset_index(drop=True)
NUM_WORKERS = 0 if sys.platform == "darwin" else 4  # macOS smoke local: 0 worker (spawn hay crash); Kaggle Linux: 4
cfg_probe = TrainConfig(num_classes=45, seed=SEED, num_workers=NUM_WORKERS, batch_size=128)
dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg_probe)  # truyền train+val; make_loaders tự lọc split
t0 = time.time()
nb = 0
for x, y in dl_tr:
    nb += 1
    if nb == 8:
        break
t_loader = (time.time() - t0) / 8
print(f"data loader: {t_loader:.3f}s/batch (128 ảnh) | train imgs: {len(dl_tr.dataset)} | val: {len(dl_va.dataset)}")

# %%
# ---- B1 ResNet18: 2 epoch FULL train/val — đo s/epoch + memory ----
set_seed(SEED)
cfg_b1 = TrainConfig(model_key="resnet18", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_FULL)
b1 = build_model(cfg_b1)
params_b1 = count_params(b1)
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
b1, hist_b1 = train_model(b1, cfg_b1, dl_tr, dl_va, device)
peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
m_b1 = evaluate(b1, dl_va, device)
lat_b1 = measure_latency(b1, device, n_iter=10 if device == "cpu" else 50)
print("B1 params:", params_b1, "| val:", {k: m_b1[k] for k in ("acc", "macro_f1")},
      "| sec/epoch:", [h["epoch_sec"] for h in hist_b1["history"]], "| peak mem GiB:", peak_mem,
      "| latency:", lat_b1)

# %%
# ---- ASF: kiểm khởi tạo ≡ ResNet18 + forward/backward + subset epoch ----
set_seed(SEED)
cfg_asf = TrainConfig(model_key="asf", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_SUB)
asf = build_model(cfg_asf)
params_asf = count_params(asf)
# (a) zero-init ⇒ đầu ra ≡ ResNet18 fresh: so với ResNet18 CHƯA train (b1 đã train,
#     BN running stats đổi → không dùng), copy head để khác biệt chỉ còn residual branch
xb, yb = next(iter(dl_tr))
ref_fresh = build_model(TrainConfig(model_key="resnet18", num_classes=45, seed=SEED))
asf.eval(); ref_fresh.eval()
asf.head.load_state_dict(ref_fresh.fc.state_dict())
with torch.no_grad():
    l_asf, l_ref = asf(xb[:16].to(device)), ref_fresh(xb[:16].to(device))
init_diff = float((l_asf - l_ref).abs().max())
print(f"ASF init vs ResNet18-fresh logits max|Δ| = {init_diff:.4e} (mong đợi ≈ 0 do zero-init residual)")
# (b) gate weights theo ảnh có khác nhau giữa 2 ảnh không (hoạt động)
asf.train()
gw = asf.gate_weights(xb[:4].to(device))
gate_spread = float(gw.var(dim=0).mean())
print("gate shape:", tuple(gw.shape), "| gate var trung bình giữa ảnh:", f"{gate_spread:.2e}")
# (c) 1 epoch subset
sub_train = full_train.groupby("class_name").sample(frac=SUBSET_FRAC, random_state=SEED).reset_index(drop=True)
dl_sub = torch.utils.data.DataLoader(ResiscDataset(sub_train, DATA_ROOT, True, cfg_asf.input_size),
                                     batch_size=128, shuffle=True, num_workers=NUM_WORKERS)
set_seed(SEED)
asf = build_model(cfg_asf)
asf, hist_asf = train_model(asf, cfg_asf, dl_sub, dl_va, device)
lat_asf = measure_latency(asf, device, n_iter=10 if device == "cpu" else 50)
print("ASF params:", params_asf, "| subset epoch sec:", [h["epoch_sec"] for h in hist_asf["history"]],
      "| val:", hist_asf.get("best_val_macro_f1"), "| latency:", lat_asf)

# %%
# ---- CMP: forward/backward + margin chiều đúng + subset epoch ----
set_seed(SEED)
w_fake = np.zeros((45, 45), dtype=np.float32)
w_fake[0, 40] = 0.8  # giả lập cặp dễ nhầm (chỉ để test cơ chế margin)
cfg_cmp = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_SUB, cmp_n_proto=2, cmp_confusion=True)
cmp_model = build_model(cfg_cmp)
cmp_model.set_confusion(w_fake)
params_cmp = count_params(cmp_model)
# margin chiều đúng: ảnh label=0 → logit class 40 phải giảm khi apply margin
cmp_model.eval()
with torch.no_grad():
    e = cmp_model.embeddings(xb[:8].to(device))
    y_test = torch.zeros(8, dtype=torch.long)
    l_margin = cmp_model.logits_from_emb(e, apply_margin=True, labels=y_test)
    l_plain = cmp_model.logits_from_emb(e, apply_margin=False, labels=y_test)
delta_40 = float((l_margin[:, 40] - l_plain[:, 40]).mean())
print(f"CMP margin Δlogit[class40, label0] = {delta_40:.4f} (mong đợi < 0 = −0.3·0.8·?)")
print("CMP params:", params_cmp)
# subset epoch (không confusion — graph chỉ đo mini ở cell sau)
cfg_cmp_train = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_SUB, cmp_n_proto=2)
set_seed(SEED)
cmp_model = build_model(cfg_cmp_train)
cmp_model, hist_cmp = train_model(cmp_model, cfg_cmp_train, dl_sub, dl_va, device)
usage = cmp_model.prototype_usage(dl_sub, device, max_batches=3)
lat_cmp = measure_latency(cmp_model, device, n_iter=10 if device == "cpu" else 50)
print("CMP subset epoch sec:", [h["epoch_sec"] for h in hist_cmp["history"]], "| val F1:", hist_cmp.get("best_val_macro_f1"))
print("prototype usage:", usage, "| latency:", lat_cmp)

# %%
# ---- Đo mini confusion graph (chỉ đo THỜI GIAN, không dùng graph này cho final) ----
t0 = time.time()
probe_train = full_train.groupby("class_name").head(40).reset_index(drop=True)  # mini: 40 ảnh/lớp chỉ để đo thời gian
w_probe, graph_info = build_confusion_graph(probe_train, DATA_ROOT, graph_frac=0.25, seed=SEED,
                                            probe_epochs=1, device=device, num_classes=45, num_workers=NUM_WORKERS)
graph_time = time.time() - t0
print(f"mini graph (45×40 ảnh, 1 epoch): {graph_time:.1f}s | active pairs: {graph_info['n_pairs_active']}")
print("→ ngoại suy chi phí graph thực (full train, 6 epoch probe) ở report")

# %%
# ---- Latency các baseline còn lại (không train) ----
lats = {"resnet18": lat_b1, "asf": lat_asf, "cmp": lat_cmp}
for key in ("smallcnn", "mobilenetv3s", "efficientnetb0", "convnextt"):
    set_seed(SEED)
    m = build_model(TrainConfig(model_key=key, num_classes=45, seed=SEED))
    lats[key] = measure_latency(m, device, n_iter=10 if device == "cpu" else 50)
    print(key, "latency:", lats[key], "params:", count_params(m))

# %%
# ---- Xuất pilot_report.json ----
report = {
    "run_id": RUN_ID, "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    "torch": torch.__version__, "cuda": torch.version.cuda,
    "manifest": os.path.basename(manifest_path), "split_sha256": audit["split_sha256"],
    "audit_smoke": audit["smoke"],
    "resnet18_full": {"params": params_b1, "sec_per_epoch": [h["epoch_sec"] for h in hist_b1["history"]],
                      "train_time_sec": hist_b1["train_time_sec"], "val": {k: m_b1[k] for k in ("acc", "macro_f1")},
                      "peak_mem_gib": peak_mem},
    "asf": {"params": params_asf, "init_max_logit_diff": init_diff, "gate_var": gate_spread,
            "subset_epoch_sec": [h["epoch_sec"] for h in hist_asf["history"]],
            "best_val_macro_f1_subset": hist_asf.get("best_val_macro_f1")},
    "cmp": {"params": params_cmp, "margin_delta_logit": delta_40,
            "subset_epoch_sec": [h["epoch_sec"] for h in hist_cmp["history"]],
            "best_val_macro_f1_subset": hist_cmp.get("best_val_macro_f1"), "prototype_usage": usage},
    "graph": {"mini_time_sec": round(graph_time, 1), "info": graph_info},
    "latency_batch1": lats,
    "loader_sec_per_batch128": round(t_loader, 4),
}
with open("pilot_report.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1, ensure_ascii=False))

# %%
# ---- FAIL-FAST tổng kết pilot ----
ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

check("B1 epoch hợp lý (< 900s)", all(h["epoch_sec"] < 900 for h in hist_b1["history"]),
      f"{[h['epoch_sec'] for h in hist_b1['history']]}")
check("ASF init ≡ ResNet18", init_diff < 1e-3, f"max|Δ|={init_diff:.2e}")
check("gate có hoạt động (var > 0)", gate_spread > 0 or cfg_asf.asf_fixed_fusion, f"{gate_spread:.2e}")
check("CMP margin âm", delta_40 < 0, f"{delta_40:.3f}")
check("latency đo được mọi model", all(v["median_ms"] > 0 for v in lats.values()))
print("PILOT", "PASS — dùng số liệu chốt compute_budget.md" if ok else "CÓ FAIL — sửa trước khi sang giai đoạn C")
