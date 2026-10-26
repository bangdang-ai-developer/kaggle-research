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
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    raise AssertionError(f"FAIL-FAST: khong tim thay {desc}: {pattern_list}")

manifest_path = find_single(
    ["/kaggle/input/**/split_manifest.csv",           # Kaggle mount layout bất kỳ tầng nào
     "../../notebooks/010-data-audit/split_manifest.smoke.csv",
     "split_manifest.smoke.csv"],
    "split manifest")
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    nsub = len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))])
    if nsub >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, f"FAIL-FAST: khong tim thay dataset root; /kaggle/input = {os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'KHONG TON TAI'}"
manifest = pd.read_csv(manifest_path)
manifest = manifest[manifest.split != "test"].reset_index(drop=True)  # pilot: KHÔNG test
assert manifest.split.eq("test").sum() == 0, "FAIL-FAST: pilot không được động tới test"
print("manifest:", manifest_path, "| rows (train+val):", len(manifest))
print("DATA_ROOT:", DATA_ROOT)
audit_path = find_single(
    ["/kaggle/input/**/audit_report.json",
     "../../notebooks/010-data-audit/audit_report.smoke.json"],
    "audit report")
with open(audit_path) as f:
    audit = json.load(f)
print("audit run_id:", audit["run_id"], "| smoke:", audit["smoke"], "| split sha256:", audit["split_sha256"][:16])

# %% embed:src/models.py

# %%
# ---- STAGE dataset sang disk local (1 lần) — sửa bottleneck FUSE ~17-37 ảnh/s làm GPU 0% ----
# v1 pilot đo được: data loader 7.35s/batch (17 ảnh/s) từ /kaggle/input, GPU hầu như 0%.
# Copy train+val (~26.775 ảnh) sang /tmp (disk local, còn ~55GiB) song song 16 thread,
# verify MD5 sample, rồi mọi loader đọc từ local — epoch compute-bound.
full_train = manifest[manifest.split == "train"].reset_index(drop=True)
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
    print("DATA_ROOT (staged):", DATA_ROOT)
else:
    print("local run — bỏ qua stage (DATA_ROOT đã là disk local)")

# %%
# ---- Dataset/loaders kiểm tra tốc độ data loader ----
set_seed(SEED)
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
# ---- B1 ResNet18: 2 epoch FULL train/val — đo s/epoch + memory + chi phí val/checkpoint ----
set_seed(SEED)
cfg_b1 = TrainConfig(model_key="resnet18", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_FULL)
b1 = build_model(cfg_b1)
params_b1 = count_params(b1)
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
b1, hist_b1 = train_model(b1, cfg_b1, dl_tr, dl_va, device)
peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
t0 = time.time()
m_b1 = evaluate(b1, dl_va, device)
val_eval_sec = time.time() - t0
lat_b1 = measure_latency(b1, device, n_iter=10 if device == "cpu" else 50)
ckpt_path = "ckpt_b1_pilot.pt"
torch.save({"state_dict": b1.state_dict(), "config": cfg_b1.to_dict()}, ckpt_path)
ckpt_mb = os.path.getsize(ckpt_path) / 2**20
print("B1 params:", params_b1, "| val:", {k: m_b1[k] for k in ("acc", "macro_f1")},
      "| sec/epoch:", [h["epoch_sec"] for h in hist_b1["history"]], "| peak mem GiB:", peak_mem,
      "| latency:", lat_b1)
print(f"val evaluate: {val_eval_sec:.2f}s / {len(dl_va.dataset)} ảnh | checkpoint: {ckpt_mb:.1f} MB")

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
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
asf, hist_asf = train_model(asf, cfg_asf, dl_sub, dl_va, device)
peak_mem_asf = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
lat_asf = measure_latency(asf, device, n_iter=10 if device == "cpu" else 50)
print("ASF params:", params_asf, "| subset epoch sec:", [h["epoch_sec"] for h in hist_asf["history"]],
      "| val:", hist_asf.get("best_val_macro_f1"), "| latency:", lat_asf, "| peak mem GiB:", peak_mem_asf)

# %%
# ---- ASF gradient mechanics: nhánh residual có được học không? ----
# Lý thuyết: zero-init out_proj ⇒ step 0: grad(out_proj) ≠ 0 (nhánh nhận gradient), grad(gate) = 0
# (chuỗi đạo hàm qua W_out=0). Sau ≥1 optimizer step, W_out ≠ 0 ⇒ grad(gate) > 0.
# Kiểm chứng 3 steps, kiểm tra finite + trọng số thật sự thay đổi.
set_seed(SEED)
asf_g = build_model(TrainConfig(model_key="asf", num_classes=45, seed=SEED)).to(device)
opt_g = torch.optim.AdamW(_param_groups(asf_g, TrainConfig(num_classes=45)), lr=1e-3)
crit_g = torch.nn.CrossEntropyLoss()
w0 = asf_g.out_proj.weight.detach().clone()
grad_hist = []
for step in range(3):
    opt_g.zero_grad(set_to_none=True)
    logits = asf_g(xb.to(device))
    loss = crit_g(logits, yb.to(device))
    loss.backward()
    g_out = float(asf_g.out_proj.weight.grad.abs().sum())
    g_gate = float(asf_g.gate.weight.grad.abs().sum())
    finite = bool(all(torch.isfinite(p.grad).all().item() for p in asf_g.parameters() if p.grad is not None))
    grad_hist.append({"step": step, "out_proj_grad": g_out, "gate_grad": g_gate, "all_finite": finite})
    opt_g.step()
dw = float((asf_g.out_proj.weight - w0).abs().sum())
asf_grad_ok = (grad_hist[0]["out_proj_grad"] > 0 and grad_hist[0]["gate_grad"] == 0.0
               and grad_hist[-1]["gate_grad"] > 0 and all(h["all_finite"] for h in grad_hist) and dw > 0)
for h in grad_hist:
    print(h)
print(f"out_proj weight Δ sau 3 steps: {dw:.3e} (nhánh residual thực sự cập nhật)")
print("ASF GRADIENT MECHANICS:", "PASS" if asf_grad_ok else "FAIL")

# %%
# ---- CMP: forward/backward + margin targeting + gradient prototype/embed + subset epoch ----
set_seed(SEED)
w_fake = np.zeros((45, 45), dtype=np.float32)
w_fake[0, 40] = 0.8   # cặp dễ nhầm giả lập (0→40): chỉ để test cơ chế margin
w_fake[3, 7] = 0.5    # thêm cặp thứ hai để kiểm targeting theo hàng
cfg_cmp = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_SUB, cmp_n_proto=2, cmp_confusion=True)
cmp_model = build_model(cfg_cmp)
cmp_model.set_confusion(w_fake)
params_cmp = count_params(cmp_model)
# (a) margin targeting: Δlogit[i, c] = −m·w[label_i, c] — đúng negative, KHÔNG đụng nhãn đúng
cmp_model.eval()
with torch.no_grad():
    e_t = cmp_model.embeddings(xb[:8].to(device))
    y_mix = torch.tensor([0, 1, 0, 3, 1, 3, 0, 1])  # trộn nhiều nhãn
    l_m = cmp_model.logits_from_emb(e_t, apply_margin=True, labels=y_mix)
    l_p = cmp_model.logits_from_emb(e_t, apply_margin=False, labels=y_mix)
delta = (l_m - l_p)
exp_delta = -(cmp_model.conf_margin * cmp_model.conf_w[y_mix]) * (1.0 - torch.nn.functional.one_hot(y_mix, 45).float())
margin_targeting_ok = bool(torch.allclose(delta, exp_delta, atol=1e-6))
delta_label0_40 = float(delta[y_mix == 0][:, 40].mean())
delta_on_label = float(delta[torch.arange(8), y_mix].abs().max())
print("CMP margin targeting (Δ == −m·w trên negative, Δ[label]=0):", "PASS" if margin_targeting_ok else "FAIL")
print(f"  Δlogit[40 | label 0] = {delta_label0_40:.4f} (mong đợi −0.24) | |Δ| trên nhãn đúng max = {delta_on_label:.1e}")
# (b) gradient prototype + embedding nhận được
cfg_grad = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, cmp_n_proto=2, cmp_confusion=True)
cmp_grad = build_model(cfg_grad)
cmp_grad.set_confusion(w_fake)
cmp_grad.train()
loss_g = torch.nn.CrossEntropyLoss()(cmp_grad(xb[:16].to(device), yb[:16].to(device)), yb[:16].to(device))
loss_g.backward()
g_proto = float(cmp_grad.prototypes.grad.abs().sum())
g_embed = float(cmp_grad.embed.weight.grad.abs().sum())
g_bb = float(sum(p.grad.abs().sum().item() for p in cmp_grad.backbone.parameters() if p.grad is not None))
cmp_proto_embed_grad_ok = g_proto > 0 and g_embed > 0 and g_bb > 0
print(f"CMP grads: prototypes={g_proto:.3e} embed={g_embed:.3e} backbone(sum)={g_bb:.3e} →",
      "PASS" if cmp_proto_embed_grad_ok else "FAIL")
# (c) subset epoch (không confusion — graph chỉ đo mini ở cell sau)
cfg_cmp_train = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=PILOT_EPOCHS_SUB, cmp_n_proto=2)
set_seed(SEED)
cmp_model = build_model(cfg_cmp_train)
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
cmp_model, hist_cmp = train_model(cmp_model, cfg_cmp_train, dl_sub, dl_va, device)
peak_mem_cmp = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
usage = cmp_model.prototype_usage(dl_sub, device, max_batches=3)
lat_cmp = measure_latency(cmp_model, device, n_iter=10 if device == "cpu" else 50)
print("CMP subset epoch sec:", [h["epoch_sec"] for h in hist_cmp["history"]], "| val F1:", hist_cmp.get("best_val_macro_f1"))
print("prototype usage:", usage, "| latency:", lat_cmp, "| peak mem GiB:", peak_mem_cmp)

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
                      "peak_mem_gib": peak_mem, "val_eval_sec": round(val_eval_sec, 2),
                      "checkpoint_mb": round(ckpt_mb, 1)},
    "asf": {"params": params_asf, "init_max_logit_diff": init_diff, "gate_var": gate_spread,
            "subset_epoch_sec": [h["epoch_sec"] for h in hist_asf["history"]],
            "best_val_macro_f1_subset": hist_asf.get("best_val_macro_f1"),
            "peak_mem_gib": peak_mem_asf, "gradient_mechanics": grad_hist, "out_proj_weight_delta_3steps": dw,
            "gradient_mechanics_ok": bool(asf_grad_ok)},
    "cmp": {"params": params_cmp, "margin_targeting_ok": bool(margin_targeting_ok),
            "delta_logit_40_label0": delta_label0_40, "delta_on_true_label_max": delta_on_label,
            "grads": {"prototypes": g_proto, "embed": g_embed, "backbone_sum": g_bb},
            "grad_ok": bool(cmp_proto_embed_grad_ok),
            "subset_epoch_sec": [h["epoch_sec"] for h in hist_cmp["history"]],
            "best_val_macro_f1_subset": hist_cmp.get("best_val_macro_f1"), "prototype_usage": usage,
            "peak_mem_gib": peak_mem_cmp},
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
check("ASF nhánh residual học được (grad đúng lý thuyết + finite + weight đổi)", bool(asf_grad_ok),
      f"step0: out={grad_hist[0]['out_proj_grad']:.2e} gate={grad_hist[0]['gate_grad']:.2e}; step2 gate={grad_hist[-1]['gate_grad']:.2e}; Δw={dw:.2e}")
check("CMP margin targeting đúng negative", bool(margin_targeting_ok), f"Δ[label0→40]={delta_label0_40:.4f}")
check("CMP prototype/embed/backbone có grad", bool(cmp_proto_embed_grad_ok),
      f"proto={g_proto:.2e} embed={g_embed:.2e}")
check("latency đo được mọi model", all(v["median_ms"] > 0 for v in lats.values()))
check("chi phí val/checkpoint đo được", val_eval_sec > 0 and ckpt_mb > 0, f"{val_eval_sec:.2f}s / {ckpt_mb:.1f}MB")
print("PILOT", "PASS — dùng số liệu chốt compute_budget.md" if ok else "CÓ FAIL — sửa trước khi sang giai đoạn C")
