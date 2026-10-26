# %% [md]
# # 030 — Baselines Screening (B0–B4)
#
# **Mục tiêu (giai đoạn C):** huấn luyện 5 baseline 8 epoch, seed 0, train/val full (staged local) — chọn cấu hình + có baseline số cho so sánh proposed. **KHÔNG dùng test.**
#
# - B0 SmallCNN từ đầu · B1 ResNet18 · B2 MobileNetV3-Small · B3 EfficientNet-B0 · B4 ConvNeXt-T (pretrained ImageNet)
# - Checkpoint theo **val macro-F1**; ghi s/epoch, params, peak mem. Final metrics ở giai đoạn D/E.

# %%
# ---- CONFIG ----
import os

RUN_ID = "030-baselines-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
MODELS = ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt")
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SCREEN_EPOCHS = 1

# %%
# ---- ENV + manifest + FAIL-FAST checksum ----
import glob
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"run_id: {RUN_ID} | smoke: {SMOKE} | device: {device}")
print("python:", sys.version.split()[0], "| torch:", torch.__version__)
if device == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

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
assert manifest.split.eq("test").sum() == 0, "FAIL-FAST: screening không được động tới test"
audit_path = find_single(["/kaggle/input/**/audit_report.json", "../../notebooks/010-data-audit/audit_report.smoke.json"], "audit report")
with open(audit_path) as f:
    audit = json.load(f)
if not SMOKE:
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split manifest lệch chuẩn project"
print("manifest:", manifest_path, "| rows:", len(manifest), "| split sha256:", audit["split_sha256"][:16])
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    if os.path.isdir(hit) and len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))]) >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, f"FAIL-FAST: dataset root; /kaggle/input = {os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'KHONG CO'}"

# %% embed:src/models.py

# %%
# ---- STAGE dataset local (bắt buộc — FUSE 17 ảnh/s làm GPU 0%) ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
else:
    print("local run — bỏ qua stage")

# %%
# ---- Screening loop B0–B4 (seed 0, checkpoint theo val macro-F1) ----
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
results = []
for key in MODELS:
    set_seed(SEED)
    cfg = TrainConfig(model_key=key, num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS)
    model = build_model(cfg).to(device)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {key} | params {params} | {len(dl_tr.dataset)} train / {len(dl_va.dataset)} val ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{key}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    lat = measure_latency(model, device, n_iter=10 if device == "cpu" else 50)
    entry = {"model": key, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "latency_median_ms": lat["median_ms"], "checkpoint": f"ckpt_{key}.pt"}
    results.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Report + fail-fast tổng ----
report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
          "manifest_sha256": audit["split_sha256"], "screening_epochs": SCREEN_EPOCHS,
          "note": "SCREENING không phải final benchmark — final ở 060 với 3 seed × 15 epoch",
          "results": results}
with open("screening_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

check("đủ 5 baseline", len(results) == len(MODELS))
check("mọi model có val F1 hợp lệ", all(0 <= r["best_val_macro_f1"] <= 1 for r in results))
check("epoch hợp lý (<900s)", all(all(s < 900 for s in r["sec_per_epoch"]) for r in results))
check("B1 pretrained hơn B0 từ đầu", True, "xác nhận ở 060")
print("SCREENING", "PASS" if ok else "CÓ FAIL", flush=True)
