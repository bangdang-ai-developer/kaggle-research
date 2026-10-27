# %% [md]
# # 060b — Final Training: Proposed ASF (A2fix-ent) + CMP (C2) × 3 seeds
#
# **Protocol khóa** (`notes/final_plan.json`): như 060a. ASF = A2fix-ent (small-init + entropy 0.05),
# CMP = C2 thuần (K=2 — cấu hình tốt nhất 050; confusion margin đã negative, giữ trung thực).
# Gate analysis lưu theo seed. **KHÔNG dùng test.**

# %%
# ---- CONFIG (khớp notes/final_plan.json) ----
import os

RUN_ID = "060b-final-proposed-v1"
SEEDS = (0, 1, 2)
EPOCHS = 15
VARIANTS = (
    ("ASF_A2fix_ent", dict(model_key="asf", asf_out_init="small", asf_gate_ent=0.05,
                           asf_detail_mode="haar", asf_fixed_fusion=False)),
    ("CMP_C2", dict(model_key="cmp", cmp_n_proto=2, cmp_confusion=False)),
)
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
# ---- Final loop: seed × variant, report tăng dần ----
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
results = []
REPORT = "final_proposed_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seeds": list(SEEDS),
                   "epochs": EPOCHS, "manifest_sha256": audit["split_sha256"],
                   "plan": "notes/final_plan.json", "results": results}, f, indent=1, ensure_ascii=False)

for seed in SEEDS:
    for name, overrides in VARIANTS:
        ck = f"ckpt_{name}_s{seed}.pt"
        if os.path.exists(ck) and os.path.getsize(ck) > 1e6:
            print(f"[skip] {ck} đã tồn tại — resume không train lại", flush=True)
            continue
        set_seed(seed)
        cfg = TrainConfig(num_classes=45, seed=seed, epochs=EPOCHS, num_workers=NUM_WORKERS, **overrides)
        model = build_model(cfg).to(device)
        params = count_params(model)
        dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        print(f"\n=== {name} seed {seed} | params {params['total']} ===", flush=True)
        model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=ck)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        m = evaluate(model, dl_va, device)
        gate_summary = None
        usage = None
        if cfg.model_key == "asf":
            xb_s, _ = next(iter(dl_va))
            gw = model.gate_weights(xb_s[:16].to(device)).cpu()
            gate_summary = {"gate_mean_per_component": [round(v, 4) for v in gw.mean(0).tolist()],
                            "gate_std_across_images": [round(v, 5) for v in gw.std(0).tolist()],
                            "max_mean_component": round(float(gw.mean(0).max()), 4),
                            "components": ["LL2", "LH2", "HL2", "HH2", "LL4", "LH4", "HL4", "HH4"]}
        else:
            usage = model.prototype_usage(dl_va, device)
        entry = {"variant": name, "seed": seed, "config": cfg.to_dict(), "params": params,
                 "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
                 "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
                 "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
                 "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
                 "gate_summary": gate_summary, "prototype_usage": usage, "checkpoint": ck}
        results.append(entry)
        save_report()
        print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Tổng hợp mean ± std + so B1 (số 060a sẽ merge ở 070) ----
def ms(vals: list[float]) -> dict:
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}

summary = {}
for name, _ in VARIANTS:
    rows = [r for r in results if r["variant"] == name and r["seed"] in SEEDS]
    if rows:
        summary[name] = {"n_seeds": len(rows), "val_macro_f1": ms([r["best_val_macro_f1"] for r in rows]),
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
    check("đủ 6 run (2 model × 3 seed)", len(results) == 6, f"{len(results)}")
    asf_gates = [r["gate_summary"]["max_mean_component"] for r in results if r["variant"] == "ASF_A2fix_ent"]
    check("ASF gate không sụp ở mọi seed (max mean < 0.9)", all(g < 0.9 for g in asf_gates), f"{asf_gates}")
print("FINAL PROPOSED", "PASS" if ok else "CÓ FAIL", flush=True)
