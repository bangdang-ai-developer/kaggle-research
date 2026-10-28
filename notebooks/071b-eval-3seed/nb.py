# %% [md]
# # 071 — Expanded Test Evaluation: 7 model × 3 seeds (21 checkpoints)
#
# **Theo `notes/final_plan_v2.json` (đã khóa):** mở rộng báo cáo test cho ĐỦ checkpoints v1 —
# mean ± std theo seed, paired ASF-vs-B1 và CMP-vs-B1. **Không dùng kết quả để chọn lại model**
# (model chính vẫn là v1 best-val-seed đã công bố). Không robustness/latency mới — giữ 070b.

# %%
# ---- CONFIG ----
import os

RUN_ID = "071-eval-3seed-v1"
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SEEDS = (0, 1, 2)
MODELS = ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt",
          "ASF_A2fix_ent", "CMP_C2")
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SEEDS = (0,)

# %%
# ---- ENV + manifest ----
import glob
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
print(f"run_id: {RUN_ID} | smoke: {SMOKE} | device: {device}")

def find_all(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern, recursive=True))

manifest_path = find_all("/kaggle/input/**/split_manifest.csv") or find_all("../../notebooks/010-data-audit/split_manifest.smoke.csv")
full = pd.read_csv(manifest_path[0])
audit_path = find_all("/kaggle/input/**/audit_report.json") or find_all("../../notebooks/010-data-audit/audit_report.smoke.json")
with open(audit_path[0]) as f:
    audit = json.load(f)
if not SMOKE:
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split lệch chuẩn"
co_path = find_all("/kaggle/input/**/class_order.json") or find_all("../../notebooks/010-data-audit/class_order.smoke.json")
CLASS_ORDER = json.load(open(co_path[0]))
CLASSES = [c for c, i in sorted(CLASS_ORDER.items(), key=lambda kv: kv[1])]
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    if os.path.isdir(hit) and len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))]) >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, "FAIL-FAST: dataset root"

CONFIG_OVERRIDES = {"ASF_A2fix_ent": dict(asf_out_init="small", asf_gate_ent=0.05),
                    "CMP_C2": dict(cmp_n_proto=2)}

# %% embed:src/models.py

# %%
# ---- STAGE full (lần đầu cần test) ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(full, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=NUM_WORKERS * 4)
else:
    print("local run — bỏ qua stage")

test_manifest = full[full.split == "test"].reset_index(drop=True)
test_ds = ResiscDataset(test_manifest, DATA_ROOT, train=False, input_size=256)
print("test images:", len(test_manifest))

# %%
# ---- Eval 21 checkpoints ----
from sklearn.metrics import precision_recall_fscore_support

results = []
REPORT = "test_3seed_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device,
                   "manifest_sha256": audit["split_sha256"], "seeds": list(SEEDS),
                   "plan": "notes/final_plan_v2.json", "results": results}, f, indent=1, ensure_ascii=False)

for key in MODELS:
    ov = CONFIG_OVERRIDES.get(key, {})
    for seed in SEEDS:
        ck_rel = f"ckpt_{key}_s{seed}.pt"
        hits = (find_all(f"/kaggle/input/**/{ck_rel}") or find_all(f"outputs/**/{ck_rel}")
                or find_all(f"../../outputs/**/{ck_rel}"))
        assert hits, f"FAIL-FAST: thiếu checkpoint {ck_rel}"
        set_seed(seed)
        cfg = TrainConfig(num_classes=45, seed=seed, model_key=("asf" if key.startswith("ASF")
                          else "cmp" if key.startswith("CMP") else key), **ov)
        model = build_model(cfg, pretrained=False).to(device)
        sd = torch.load(hits[0], map_location=device, weights_only=False)
        model.load_state_dict(sd["state_dict"])
        model.eval()
        dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
        ys, ps = [], []
        with torch.inference_mode():
            for x, y in dl:
                logits = model(x.to(device))
                ps.append(logits.argmax(1).cpu())
                ys.append(y)
        y_true, y_pred = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        acc = float((y_true == y_pred).mean())
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                            labels=list(range(45)), zero_division=0)
        entry = {"model": key, "seed": seed, "test_acc": round(acc, 4),
                 "test_macro_precision": round(float(p_m), 4), "test_macro_recall": round(float(r_m), 4),
                 "test_macro_f1": round(float(f1_m), 4), "n": int(len(y_true)),
                 "latency": measure_latency(model, device, n_iter=10 if device == "cpu" else 50)}
        results.append(entry)
        save_report()
        print(f"[eval] {key} s{seed}: acc={acc:.4f} F1={f1_m:.4f}", flush=True)

# %%
# ---- Tổng hợp mean ± std + paired ----
def ms(vals: list[float]) -> dict:
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}

summary = {}
for key in MODELS:
    rows = [e for e in results if e["model"] == key]
    if rows:
        summary[key] = {"n_seeds": len(rows), "test_acc": ms([e["test_acc"] for e in rows]),
                        "test_macro_f1": ms([e["test_macro_f1"] for e in rows]),
                        "latency_median_ms": ms([e["latency"]["median_ms"] for e in rows])}
# paired theo seed
paired = {}
if not SMOKE and len(results) == 21:
    b1 = {e["seed"]: e["test_macro_f1"] for e in results if e["model"] == "resnet18"}
    for prop in ("ASF_A2fix_ent", "CMP_C2"):
        rows = {e["seed"]: e for e in results if e["model"] == prop}
        diffs = [rows[s]["test_macro_f1"] - b1[s] for s in SEEDS]
        wins = int(sum(d > 0 for d in diffs))
        paired[prop] = {"diffs_per_seed": [round(d, 4) for d in diffs],
                        "mean_diff": round(float(np.mean(diffs)), 4),
                        "wins_vs_B1": f"{wins}/3"}
print(json.dumps({"summary": summary, "paired_vs_B1": paired}, indent=1, ensure_ascii=False))
rep = json.load(open(REPORT))
rep["summary"], rep["paired_vs_B1"] = summary, paired
with open(REPORT, "w") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

if not SMOKE:
    check("đủ 21 run", len(results) == 21, f"{len(results)}")
    check("test acc mọi checkpoint trong [0.4, 1]", all(0.4 < e["test_acc"] <= 1 for e in results))
    check("B1 test mean ≈ công bố (0.9530 ± 0.005)", abs(summary["resnet18"]["test_acc"]["mean"] - 0.9530) < 0.005,
          f"{summary['resnet18']['test_acc']}")
print("EXPANDED EVAL", "PASS" if ok else "CÓ FAIL", flush=True)
