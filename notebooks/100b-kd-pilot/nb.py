# %% [md]
# # 100 — KD Pilot: teacher verify + KD GPU sanity + timing teacher-online
#
# **V3 (mục 7A/7B spec):** xác minh teacher (checkpoint hash, class mapping), KD loss trên GPU,
# pilot 2 epoch ResNet18-KD đo s/epoch có teacher online + peak memory. Đầu ra chốt ngân sách screening.
#
# **Teacher:** ConvNeXt-T seed 2 — best val macro-F1 0.9691 (060a), sha256 `9627ca43…`. Frozen eval.

# %%
# ---- CONFIG ----
import os

RUN_ID = "100-kd-pilot-v1"
TEACHER_SEED = 2
TEACHER_VAL_F1 = 0.9691
TEACHER_SHA_PREFIX = "9627ca43"
KD_ALPHA, KD_T = 0.5, 4.0
PILOT_EPOCHS = 2
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    PILOT_EPOCHS = 1

# %%
# ---- ENV + manifest + teacher checkpoint verify ----
import glob
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
print(f"run_id: {RUN_ID} | smoke: {SMOKE} | device: {device}")
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu")

def find_single(pattern_list: list[str], desc: str) -> str:
    for pat in pattern_list:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    raise AssertionError(f"FAIL-FAST: khong tim thay {desc}: {pattern_list}")

manifest_path = find_single(
    ["/kaggle/input/**/split_manifest.csv",
     "../../notebooks/010-data-audit/split_manifest.smoke.csv",
     "split_manifest.smoke.csv"], "split manifest")
manifest = pd.read_csv(manifest_path)
manifest = manifest[manifest.split != "test"].reset_index(drop=True)
audit_path = find_single(["/kaggle/input/**/audit_report.json", "../../notebooks/010-data-audit/audit_report.smoke.json"], "audit report")
with open(audit_path) as f:
    audit = json.load(f)
if not SMOKE:
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split lệch chuẩn"
print("manifest:", manifest_path, "| rows:", len(manifest))
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    if os.path.isdir(hit) and len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))]) >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, "FAIL-FAST: dataset root"

# %% embed:src/models.py

# %%
# ---- STAGE + TEACHER verify (hash + class mapping + val sanity) ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=NUM_WORKERS * 4)
else:
    print("local run — bỏ qua stage")

teacher_ckpt_rel = f"ckpt_convnextt_s{TEACHER_SEED}.pt"
teacher_path = find_single([f"/kaggle/input/**/{teacher_ckpt_rel}",
                            f"../../outputs/resisc45-final-baselines-b0-b4-3-seeds/{teacher_ckpt_rel}"],
                           f"teacher checkpoint {teacher_ckpt_rel}")
h = hashlib.sha256(open(teacher_path, "rb").read()).hexdigest()
print(f"teacher: {teacher_path}")
print(f"sha256: {h[:24]}… | match prefix: {h.startswith(TEACHER_SHA_PREFIX)}")
if not SMOKE:
    assert h.startswith(TEACHER_SHA_PREFIX), "FAIL-FAST: teacher checkpoint khác bản đã chọn"

teacher = build_model(TrainConfig(num_classes=45, model_key="convnextt"), pretrained=False).to(device)
sd = torch.load(teacher_path, map_location=device, weights_only=False)
teacher.load_state_dict(sd["state_dict"])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"teacher loaded: epoch {sd['epoch']} | eval mode | frozen")

# teacher val sanity (dùng val set — không test)
dl_va = torch.utils.data.DataLoader(ResiscDataset(manifest[manifest.split == "val"].reset_index(drop=True),
                                                  DATA_ROOT, train=False), batch_size=256, num_workers=NUM_WORKERS)
correct = total = 0
with torch.inference_mode():
    for x, y in dl_va:
        pred = teacher(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum()); total += len(y)
teacher_val_acc = correct / total
print(f"teacher val acc (verify): {teacher_val_acc:.4f} (mong đợi ≈ 0.9672 từ 060a)")
if not SMOKE:
    assert abs(teacher_val_acc - 0.9672) < 0.01, "FAIL-FAST: teacher val acc lệch — kiểm tra class mapping"

# %%
# ---- KD loss GPU sanity (K1-K3 trên device thật) ----
import torch.nn.functional as F

ts = torch.randn(64, 45, device=device)
tt = torch.randn(64, 45, device=device)
kl = kd_loss(ts, tt, KD_T)
assert kl.dtype == torch.float32 and float(kl) > 0
print(f"KD loss GPU sanity: {float(kl):.4f} OK (dtype {kl.dtype})")

# %%
# ---- Pilot: ResNet18-KD 2 epoch (teacher online) — đo timing/memory ----
set_seed(0)
cfg = TrainConfig(num_classes=45, seed=0, epochs=PILOT_EPOCHS, num_workers=NUM_WORKERS,
                  kd_alpha=KD_ALPHA, kd_T=KD_T)
student = build_model(cfg).to(device)
dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
print(f"\n=== ResNet18-KD pilot {PILOT_EPOCHS} epoch (teacher online) ===", flush=True)
student, hist = train_model(student, cfg, dl_tr, dl_va, device, teacher=teacher)
peak_kd = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
kd_s_per_epoch = [h["epoch_sec"] for h in hist["history"]]
kd_aux = [h.get("aux", {}).get("kd", 0) for h in hist["history"]]

# đối chứng timing: student-only CE forward+backward 1 epoch cục bộ (không full train)
student_ce = build_model(TrainConfig(num_classes=45, model_key="resnet18"), pretrained=False).to(device)
t0 = time.time()
torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
opt = torch.optim.AdamW(student_ce.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()
for bi, (x, y) in enumerate(dl_tr):
    if bi >= 20: break
    opt.zero_grad(set_to_none=True)
    loss = crit(student_ce(x.to(device)), y.to(device))
    loss.backward(); opt.step()
ce_20b = time.time() - t0
peak_ce = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
peak_s = f"{peak_kd:.2f}" if peak_kd is not None else "n/a"
ce_peak_s = f"{peak_ce:.2f}" if peak_ce is not None else "n/a"
print(f"\nKD s/epoch: {kd_s_per_epoch} | CE 20 batch: {ce_20b:.1f}s | peak KD {peak_s} GiB vs CE {ce_peak_s} GiB")
print(f"KD aux loss theo epoch: {kd_aux}")

report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device,
          "teacher": {"seed": TEACHER_SEED, "val_f1": TEACHER_VAL_F1, "val_acc_verify": round(teacher_val_acc, 4),
                      "sha256_prefix": h[:24], "epoch": sd["epoch"]},
          "kd_config": {"alpha": KD_ALPHA, "T": KD_T},
          "kd_sec_per_epoch": kd_s_per_epoch, "kd_aux_per_epoch": kd_aux,
          "kd_peak_mem_gib": peak_kd, "ce_20batch_sec": round(ce_20b, 1), "ce_peak_mem_gib": peak_ce,
          "manifest_sha256": audit["split_sha256"]}
with open("kd_pilot_report.json", "w") as f:
    json.dump(report, f, indent=1)

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

if not SMOKE:
    check("teacher hash khớp", h.startswith(TEACHER_SHA_PREFIX))
    check("teacher val acc verify", abs(teacher_val_acc - 0.9672) < 0.01, f"{teacher_val_acc:.4f}")
    check("KD epoch < 2× CE epoch (teacher online chấp nhận được)",
          all(s < 2 * 53 for s in kd_s_per_epoch), f"{kd_s_per_epoch}")
    check("kd aux loss > 0", all(a > 0 for a in kd_aux), f"{kd_aux}")
    check("peak mem < 12 GiB", peak_kd < 12, f"{peak_kd:.2f}")
print("KD PILOT", "PASS — chốt ngân sách screening" if ok else "CÓ FAIL", flush=True)
