# %% [md]
# # 112 — KD Models Test Evaluation (3 seeds, theo final_plan_v3.json)
#
# Đánh giá test cho 6 checkpoints KD (RN18-KD, ASF-KD × seeds 0/1/2), so paired với CE 3-seed
# từ 071b. **Test đã quan sát v1/v2 — v3 là nghiên cứu mở rộng, công khai trong báo cáo.**
# Không dùng kết quả để chọn cấu hình (đã khóa từ screening).

# %%
# ---- CONFIG ----
import os

RUN_ID = "112-eval-kd-3seed-v1"
SEEDS = (0, 1, 2)
KD_STUDENTS = ("resnet18_kd", "asf_kd")
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
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

# %% embed:src/models.py

# %%
# ---- STAGE full + test dataset ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(full, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=NUM_WORKERS * 4)
else:
    print("local run — bỏ qua stage")
test_manifest = full[full.split == "test"].reset_index(drop=True)
test_ds = ResiscDataset(test_manifest, DATA_ROOT, train=False, input_size=256)
print("test images:", len(test_manifest))

# CE 3-seed test numbers từ 071b (paired reference — artifact nguồn)
ce_path = find_all("/kaggle/input/**/test_3seed_report.json") or find_all("../../outputs/resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json")
ce_report = json.load(open(ce_path[0]))
ce_test = {}
for e in ce_report["results"]:
    if e["model"] in ("resnet18",) or e["model"] == "ASF_A2fix_ent":
        ce_test.setdefault(e["model"], {})[e["seed"]] = e["test_macro_f1"]
print("CE test F1 (071b):", ce_test)

# %%
# ---- Eval 6 KD checkpoints ----
from sklearn.metrics import precision_recall_fscore_support

CONFIG_OVERRIDES = {"resnet18_kd": dict(model_key="resnet18"),
                    "asf_kd": dict(model_key="asf", asf_out_init="small", asf_gate_ent=0.05)}
results = []
REPORT = "test_kd_3seed_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device,
                   "manifest_sha256": audit["split_sha256"], "seeds": list(SEEDS),
                   "plan": "notes/final_plan_v3.json",
                   "caveat": "test đã quan sát v1/v2 — v3 là nghiên cứu mở rộng",
                   "results": results}, f, indent=1, ensure_ascii=False)

for student in KD_STUDENTS:
    ov = CONFIG_OVERRIDES[student]
    for seed in SEEDS:
        # 110b lưu seed 0 KHÔNG có suffix seed; 111 lưu seeds 1,2 CÓ suffix
        ck_cands = ([f"ckpt_{student}.pt", f"ckpt_{student}_s{seed}.pt"] if seed == 0
                    else [f"ckpt_{student}_s{seed}.pt"])
        hits = []
        for ck_rel in ck_cands:
            hits = (find_all(f"/kaggle/input/**/{ck_rel}") or find_all(f"outputs/**/{ck_rel}")
                    or find_all(f"../../outputs/**/{ck_rel}"))
            if hits:
                break
        assert hits, f"FAIL-FAST: thiếu checkpoint {ck_cands}"
        set_seed(seed)
        cfg = TrainConfig(num_classes=45, seed=seed, epochs=15, kd_alpha=0.5, kd_T=4.0, **ov)
        model = build_model(cfg, pretrained=False).to(device)
        sd = torch.load(hits[0], map_location=device, weights_only=False)
        model.load_state_dict(sd["state_dict"])
        model.eval()
        dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
        ys, ps = [], []
        with torch.inference_mode():
            for x, y in dl:
                ps.append(model(x.to(device)).argmax(1).cpu())
                ys.append(y)
        y_true, y_pred = torch.cat(ys).numpy(), torch.cat(ps).numpy()
        acc = float((y_true == y_pred).mean())
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                            labels=list(range(45)), zero_division=0)
        lat = measure_latency(model, device, n_iter=10 if device == "cpu" else 50)
        entry = {"student": student, "seed": seed, "test_acc": round(acc, 4),
                 "test_macro_precision": round(float(p_m), 4), "test_macro_recall": round(float(r_m), 4),
                 "test_macro_f1": round(float(f1_m), 4), "n": int(len(y_true)),
                 "latency": lat}
        results.append(entry)
        save_report()
        print(f"[eval] {student} s{seed}: acc={acc:.4f} F1={f1_m:.4f} lat={lat['median_ms']:.2f}ms", flush=True)

# %%
# ---- Tổng hợp mean±std + paired vs CE 3-seed (071b) ----
def ms(vals: list[float]) -> dict:
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}

summary, paired = {}, {}
ce_f1_by_model = {"resnet18_kd": ce_test["resnet18"], "asf_kd": ce_test["ASF_A2fix_ent"]}
ce_acc_by_model = {}
for e in ce_report["results"]:
    if e["model"] == "resnet18":
        ce_acc_by_model.setdefault("resnet18_kd", {})[e["seed"]] = e["test_acc"]
    if e["model"] == "ASF_A2fix_ent":
        ce_acc_by_model.setdefault("asf_kd", {})[e["seed"]] = e["test_acc"]

for student in KD_STUDENTS:
    rows = [e for e in results if e["student"] == student and e["seed"] in SEEDS]
    if rows:
        f1s = [e["test_macro_f1"] for e in rows]
        accs = [e["test_acc"] for e in rows]
        summary[student] = {"n_seeds": len(rows), "test_acc": ms(accs), "test_macro_f1": ms(f1s),
                            "latency_median_ms": ms([e["latency"]["median_ms"] for e in rows])}
        if not SMOKE and len(rows) == 3:
            diffs_f1 = [f1s[i] - ce_f1_by_model[student][s] for i, s in enumerate(SEEDS)]
            diffs_acc = [accs[i] - ce_acc_by_model[student][s] for i, s in enumerate(SEEDS)]
            paired[student] = {"f1_diffs_per_seed": [round(d, 4) for d in diffs_f1],
                               "f1_mean_diff": round(float(np.mean(diffs_f1)), 4),
                               "f1_wins": f"{sum(d > 0 for d in diffs_f1)}/3",
                               "acc_diffs_per_seed": [round(d, 4) for d in diffs_acc],
                               "acc_mean_diff": round(float(np.mean(diffs_acc)), 4),
                               "acc_wins": f"{sum(d > 0 for d in diffs_acc)}/3"}
print(json.dumps({"summary": summary, "paired_vs_CE": paired}, indent=1, ensure_ascii=False))
rep = json.load(open(REPORT))
rep["summary"], rep["paired_vs_CE"] = summary, paired
with open(REPORT, "w") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

if not SMOKE:
    check("đủ 6 run", len(results) == 6, f"{len(results)}")
    check("test acc hợp lệ", all(0.4 < e["test_acc"] <= 1 for e in results))
print("KD TEST EVAL", "PASS" if ok else "CÓ FAIL", flush=True)
