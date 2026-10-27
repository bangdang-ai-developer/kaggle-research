# %% [md]
# # 060a — Final Training: Baselines B0–B4 × 3 seeds
#
# **Protocol khóa** (`notes/final_plan.json`): 15 epoch, early-stop patience 5, checkpoint theo
# **val macro-F1**, seeds {0,1,2}, train/val staged local. **KHÔNG dùng test** — test chỉ ở 070.
# Report ghi tăng dần sau mỗi run (chống mất khi crash giữa session dài).

# %%
# ---- CONFIG (khớp notes/final_plan.json) ----
import os

RUN_ID = "060a-final-baselines-v1"
SEEDS = (0, 1, 2)
EPOCHS = 15
MODELS = ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt")
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SEEDS = (0,)
    EPOCHS = 1

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
assert manifest.split.eq("test").sum() == 0, "FAIL-FAST: final training không được động tới test"
audit_path = find_single(["/kaggle/input/**/audit_report.json", "../../notebooks/010-data-audit/audit_report.smoke.json"], "audit report")
with open(audit_path) as f:
    audit = json.load(f)
if not SMOKE:
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split manifest lệch chuẩn final_plan"
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
# ---- STAGE dataset local ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
else:
    print("local run — bỏ qua stage")

# %%
# ---- Final loop: seed × model, report tăng dần ----
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
results = []
REPORT = "final_baselines_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seeds": list(SEEDS),
                   "epochs": EPOCHS, "manifest_sha256": audit["split_sha256"],
                   "plan": "notes/final_plan.json", "results": results}, f, indent=1, ensure_ascii=False)

for seed in SEEDS:
    for key in MODELS:
        ck = f"ckpt_{key}_s{seed}.pt"
        if os.path.exists(ck) and os.path.getsize(ck) > 1e6:
            print(f"[skip] {ck} đã tồn tại — resume không train lại", flush=True)
            continue
        set_seed(seed)
        cfg = TrainConfig(model_key=key, num_classes=45, seed=seed, epochs=EPOCHS,
                          num_workers=NUM_WORKERS)
        model = build_model(cfg).to(device)
        params = count_params(model)
        dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        print(f"\n=== {key} seed {seed} | params {params['total']} ===", flush=True)
        model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=ck)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        m = evaluate(model, dl_va, device)
        entry = {"model": key, "seed": seed, "config": cfg.to_dict(), "params": params,
                 "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
                 "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
                 "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
                 "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
                 "checkpoint": ck}
        results.append(entry)
        save_report()  # ghi ngay sau mỗi run — crash-safe
        print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Tổng hợp mean ± std theo model + fail-fast ----
def ms(vals: list[float]) -> dict:
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}

summary = {}
for key in MODELS:
    rows = [r for r in results if r["model"] == key and r["seed"] in SEEDS]
    if rows:
        summary[key] = {"n_seeds": len(rows), "val_macro_f1": ms([r["best_val_macro_f1"] for r in rows]),
                        "val_acc": ms([r["best_val_acc"] for r in rows]),
                        "train_time_sec": ms([r["train_time_sec"] for r in rows])}
print(json.dumps(summary, indent=1, ensure_ascii=False))
report = json.load(open(REPORT))
report["summary"] = summary
with open(REPORT, "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

if not SMOKE:
    check("đủ 15 run (5 model × 3 seed)", len(results) == 15, f"{len(results)}")
    check("mọi run val F1 hợp lệ", all(0 < r["best_val_macro_f1"] < 1 for r in results))
    check("B1 ± nhiễu seed < 1%", max(s["val_macro_f1"]["std"] for s in summary.values()
          if s["n_seeds"] == 3) < 0.01, "cảnh báo variance cao nếu FAIL")
print("FINAL BASELINES", "PASS" if ok else "CÓ FAIL", flush=True)
