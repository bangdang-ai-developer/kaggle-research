# %% [md]
# # 040 — ASF Ablation Screening (A1 / A2 / spatial-control)
#
# **Mục tiêu:** tách đóng góp của (a) nhánh wavelet và (b) gate thích ứng trên cùng backbone ResNet18.
#
# - **A1**: wavelet + fusion cố định (softmax learnable, không theo ảnh)
# - **A2**: wavelet + gate thích ứng theo ảnh (proposed)
# - **A2-conv**: đối chứng spatial — 3 detail học được (depthwise 3×3 s2) thay 3 subband Haar, cùng kiến trúc gate/proj
#
# A0 = ResNet18 gốc đã có ở 030 (B1). 8 epoch, seed 0, train/val full staged. **Không test.**

# %%
# ---- CONFIG ----
import os

RUN_ID = "040-asf-ablation-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
VARIANTS = (
    ("A1_wavelet_fixed", dict(asf_fixed_fusion=True, asf_detail_mode="haar")),
    ("A2_wavelet_adaptive", dict(asf_fixed_fusion=False, asf_detail_mode="haar")),
    ("A2_conv_spatial", dict(asf_fixed_fusion=False, asf_detail_mode="conv")),
)
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
# ---- STAGE dataset local ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
else:
    print("local run — bỏ qua stage")

# %%
# ---- Ablation loop ----
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
results = []
xb = None  # sample batch dùng gate analysis
for name, overrides in VARIANTS:
    set_seed(SEED)
    cfg = TrainConfig(model_key="asf", num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS, **overrides)
    model = build_model(cfg).to(device)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | params {params} | fixed={cfg.asf_fixed_fusion} detail={cfg.asf_detail_mode} ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    # gate analysis (chỉ A2 adaptive): phân bố gate giữa 2 stage và 4 thành phần
    gate_summary = None
    if not cfg.asf_fixed_fusion:
        if xb is None:
            xb, _ = next(iter(dl_va))
        gw = model.gate_weights(xb[:16].to(device)).cpu()
        gate_summary = {"gate_mean_per_component": [round(v, 4) for v in gw.mean(0).tolist()],
                        "gate_std_across_images": [round(v, 5) for v in gw.std(0).tolist()],
                        "components": ["LL2", "LH2", "HL2", "HH2", "LL4", "LH4", "HL4", "HH4"]}
    entry = {"variant": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "gate_summary": gate_summary, "checkpoint": f"ckpt_{name}.pt"}
    results.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Report + so sánh A1/A2/conv + fail-fast ----
report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
          "manifest_sha256": audit["split_sha256"], "screening_epochs": SCREEN_EPOCHS,
          "note": "SCREENING — final ở 060; đối chứng ResNet18 gốc (A0) = B1 ở 030",
          "results": results}
with open("asf_ablation_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

by = {r["variant"]: r for r in results}
check("đủ 3 biến thể", len(results) == 3)
if not SMOKE:
    # check thống kê — vô nghĩa ở SMOKE (1 epoch/225 ảnh)
    check("A2 ≥ A1 − 0.5% F1 (gate không hại)", "A2_wavelet_adaptive" in by and "A1_wavelet_fixed" in by
          and by["A2_wavelet_adaptive"]["best_val_macro_f1"] >= by["A1_wavelet_fixed"]["best_val_macro_f1"] - 0.005,
          f"A2={by.get('A2_wavelet_adaptive', {}).get('best_val_macro_f1')} A1={by.get('A1_wavelet_fixed', {}).get('best_val_macro_f1')}")
check("A2 params ≈ A2-conv (chênh <5%)", "A2_wavelet_adaptive" in by and "A2_conv_spatial" in by
      and abs(by["A2_wavelet_adaptive"]["params"]["total"] - by["A2_conv_spatial"]["params"]["total"])
      / by["A2_wavelet_adaptive"]["params"]["total"] < 0.05)
print("ASF ABLATION SCREENING", "PASS" if ok else "CÓ FAIL", flush=True)
