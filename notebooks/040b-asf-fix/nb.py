# %% [md]
# # 040b — ASF Fix Screening (sau negative result 040: gate collapse 100% về LL4)
#
# **Bằng chứng 040:** gate mean=[0,0,0,0,1,0,0,0], std=0 → zero-init residual khiến gradient gate
# phụ thuộc out_proj (=0 lúc đầu), optimizer kéo về thành phần trùng backbone.
#
# **2 cấu hình fix (trần tuning ≤2/proposed):**
# - **A2fix-small**: `out_init="small"` (std=1e-3) — gradient gate sống từ batch đầu
# - **A2fix-ent**: small-init **+ entropy regularizer** λ=0.05 (phạt 1-hot, giữ phân bố mềm)
#
# Đối chiếu: A2 cũ 0.9456, A1 0.9469, B1 0.9448. Thành công tối thiểu: gate KHÔNG sụp (std>0,
# max mean < 0.9) và F1 > B1. 8 epoch, seed 0, train/val staged. **Không test.**

# %%
# ---- CONFIG ----
import os

RUN_ID = "040b-asf-fix-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
GATE_ENT_LAMBDA = 0.05
VARIANTS = (
    ("A2fix_small", dict(asf_out_init="small", asf_gate_ent=0.0)),
    ("A2fix_ent", dict(asf_out_init="small", asf_gate_ent=GATE_ENT_LAMBDA)),
)
B1_F1 = 0.9448
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SCREEN_EPOCHS = 1

# %%
# ---- ENV + manifest + FAIL-FAST ----
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
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split manifest lệch chuẩn"
print("manifest:", manifest_path, "| rows:", len(manifest))
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    if os.path.isdir(hit) and len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))]) >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, f"FAIL-FAST: dataset root"

# %% embed:src/models.py

# %%
# ---- STAGE dataset local ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
else:
    print("local run — bỏ qua stage")

# %%
# ---- Fix screening loop ----
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
results = []
for name, overrides in VARIANTS:
    set_seed(SEED)
    cfg = TrainConfig(model_key="asf", num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS, **overrides)
    model = build_model(cfg).to(device)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | out_init={cfg.asf_out_init} gate_ent={cfg.asf_gate_ent} | params {params} ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    xb_s, _ = next(iter(dl_va))
    gw = model.gate_weights(xb_s[:16].to(device)).cpu()
    gate_summary = {"gate_mean_per_component": [round(v, 4) for v in gw.mean(0).tolist()],
                    "gate_std_across_images": [round(v, 5) for v in gw.std(0).tolist()],
                    "max_mean_component": round(float(gw.mean(0).max()), 4),
                    "components": ["LL2", "LH2", "HL2", "HH2", "LL4", "LH4", "HL4", "HH4"]}
    entry = {"variant": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "delta_vs_B1": round(m["macro_f1"] - B1_F1, 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "gate_summary": gate_summary, "checkpoint": f"ckpt_{name}.pt"}
    results.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Report + fail-fast ----
report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
          "manifest_sha256": audit["split_sha256"], "screening_epochs": SCREEN_EPOCHS,
          "B1_baseline_f1": B1_F1, "context": "040 negative: gate collapse 100% LL4 (std=0)",
          "results": results}
with open("asf_fix_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

by = {r["variant"]: r for r in results}
check("đủ 2 cấu hình fix", len(results) == 2)
if not SMOKE:
    for r in results:
        gs = r["gate_summary"]
        check(f"{r['variant']} KHÔNG sụp 1-hot (max mean < 0.9)", gs["max_mean_component"] < 0.9,
              f"max_mean={gs['max_mean_component']}")
        check(f"{r['variant']} gate phân hóa theo ảnh (std>0)", max(gs["gate_std_across_images"]) > 0)
    best = max(results, key=lambda r: r["best_val_macro_f1"])
    check(f"best fix ({best['variant']}) > B1", best["best_val_macro_f1"] > B1_F1,
          f"{best['best_val_macro_f1']} vs {B1_F1}")
print("ASF FIX SCREENING", "PASS" if ok else "CÓ FAIL", flush=True)
