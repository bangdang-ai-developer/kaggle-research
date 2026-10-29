# %% [md]
# # 111 — KD Final: seeds 1, 2 cho ResNet18-KD và ASF-KD
#
# **Theo `notes/final_plan_v3.json` (đã khóa):** seed 0 đã có từ 110 (RN18-KD 0.9550, ASF-KD 0.9565).
# Run này bổ sung seeds 1, 2 → đủ 3 seed cho so paired với CE (071b).
# CMP-KD dừng (negative — KD loss không hội tụ tốt với hard-max head). **Không test.**

# %%
# ---- CONFIG (khớp notes/final_plan_v3.json) ----
import os

RUN_ID = "111-kd-final-v1"
SEEDS = (1, 2)
EPOCHS = 15
KD_ALPHA, KD_T = 0.5, 4.0
STUDENTS = (
    ("resnet18_kd", dict(model_key="resnet18")),
    ("asf_kd", dict(model_key="asf", asf_out_init="small", asf_gate_ent=0.05)),
)
TEACHER_SEED = 2
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SEEDS = (1,)
    EPOCHS = 1

# %%
# ---- ENV + manifest + teacher ----
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
# ---- STAGE + teacher ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=NUM_WORKERS * 4)
else:
    print("local run — bỏ qua stage")

teacher_rel = f"ckpt_convnextt_s{TEACHER_SEED}.pt"
teacher_path = find_single([f"/kaggle/input/**/{teacher_rel}",
                            f"../../outputs/resisc45-final-baselines-b0-b4-3-seeds/{teacher_rel}"],
                           f"teacher {teacher_rel}")
h = hashlib.sha256(open(teacher_path, "rb").read()).hexdigest()
if not SMOKE:
    assert h.startswith("9627ca43"), "FAIL-FAST: teacher checkpoint lệch"
teacher = build_model(TrainConfig(num_classes=45, model_key="convnextt"), pretrained=False).to(device)
sd = torch.load(teacher_path, map_location=device, weights_only=False)
teacher.load_state_dict(sd["state_dict"])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"teacher verified: seed {TEACHER_SEED}, sha {h[:16]}…, epoch {sd['epoch']}")

# %%
# ---- Final loop seeds 1,2 ----
results = []
REPORT = "kd_final_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seeds": list(SEEDS),
                   "epochs": EPOCHS, "kd": {"alpha": KD_ALPHA, "T": KD_T},
                   "teacher": {"seed": TEACHER_SEED, "sha256": h},
                   "manifest_sha256": audit["split_sha256"],
                   "note": "seed 0 từ 110b (kd_screen_report.json) — merge ở 112",
                   "results": results}, f, indent=1, ensure_ascii=False)

for seed in SEEDS:
    for name, overrides in STUDENTS:
        ck = f"ckpt_{name}_s{seed}.pt"
        if os.path.exists(ck) and os.path.getsize(ck) > 1e6:
            print(f"[skip] {ck} đã tồn tại — resume", flush=True)
            continue
        set_seed(seed)
        cfg = TrainConfig(num_classes=45, seed=seed, epochs=EPOCHS, num_workers=NUM_WORKERS,
                          kd_alpha=KD_ALPHA, kd_T=KD_T, **overrides)
        student = build_model(cfg).to(device)
        params = count_params(student)
        dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        print(f"\n=== {name} seed {seed} | params {params['total']} ===", flush=True)
        student, hist = train_model(student, cfg, dl_tr, dl_va, device,
                                    state_out=ck, teacher=teacher)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        m = evaluate(student, dl_va, device)
        usage, gate_summary = None, None
        if cfg.model_key == "cmp":
            usage = student.prototype_usage(dl_va, device)
        if cfg.model_key == "asf":
            xb_s, _ = next(iter(dl_va))
            gw = student.gate_weights(xb_s[:16].to(device)).cpu()
            gate_summary = {"gate_mean": [round(v, 4) for v in gw.mean(0).tolist()],
                            "gate_std": [round(v, 5) for v in gw.std(0).tolist()],
                            "max_mean": round(float(gw.mean(0).max()), 4)}
        entry = {"student": name, "seed": seed, "config": cfg.to_dict(), "params": params,
                 "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
                 "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
                 "history": hist["history"],
                 "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
                 "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
                 "prototype_usage": usage, "gate_summary": gate_summary, "checkpoint": ck}
        results.append(entry)
        save_report()
        print(json.dumps({k: v for k, v in entry.items() if k != "history"}, ensure_ascii=False), flush=True)

# %%
# ---- Tổng hợp 3-seed (merge seed 0 từ 110b report nếu có) ----
def ms(vals: list[float]) -> dict:
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}

seed0_path = find_single(["/kaggle/input/**/kd_screen_report.json",
                          "../../outputs/resisc45-screen-kd-resnet18-asf-cmp-v2/kd_screen_report.json"],
                         "kd screen report (seed 0)")
seed0 = json.load(open(seed0_path))
summary = {}
for name, _ in STUDENTS:
    f1s, accs = [], []
    for e in seed0["results"]:
        if e["student"] == name:
            f1s.append(e["best_val_macro_f1"]); accs.append(e["best_val_acc"])
    for e in results:
        if e["student"] == name:
            f1s.append(e["best_val_macro_f1"]); accs.append(e["best_val_acc"])
    if f1s:
        summary[name] = {"n_seeds": len(f1s), "val_macro_f1": ms(f1s), "val_acc": ms(accs)}
print(json.dumps(summary, indent=1, ensure_ascii=False))
rep = json.load(open(REPORT))
rep["summary_3seed"] = summary
with open(REPORT, "w") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

if not SMOKE:
    check("đủ 4 run (2 student × seeds 1,2)", len(results) == 4, f"{len(results)}")
    check("summary đủ 3 seed/student", all(s["n_seeds"] == 3 for s in summary.values()))
print("KD FINAL", "PASS" if ok else "CÓ FAIL", flush=True)
