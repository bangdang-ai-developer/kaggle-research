# %% [md]
# # 110 — KD Screening: ResNet18-KD / ASF-KD / CMP-KD (15 epoch, seed 0)
#
# **V3 (mục 5/6 spec):** 3 student × KD từ teacher ConvNeXt-T seed 2 (val F1 0.9691, T=4, α=0.5).
# **Controls REUSE từ 060a/060b** (cùng source/init/epochs/augmentation/optimizer/ckpt-rule, seed 0):
# - ResNet18-CE s0: 0.9483 · ASF-CE s0: 0.9507 · CMP-CE s0: 0.9503 (val F1, 15 epoch)
#
# CMP student = hard-max + kmeans init (v2 đã xác minh cơ chế), KHÔNG pair/confusion (vòng KD đầu).
# ASF student = A2fix-ent v1 (small-init + entropy 0.05).
# Tiêu chí screening: KD model > CE counterpart cùng seed (s0) → có tín hiệu trước khi final.

# %%
# ---- CONFIG ----
import os

RUN_ID = "110-kd-screen-v1"
SEED = 0
EPOCHS = 15
KD_ALPHA, KD_T = 0.5, 4.0
STUDENTS = (
    ("resnet18_kd", dict(model_key="resnet18")),
    ("asf_kd", dict(model_key="asf", asf_out_init="small", asf_gate_ent=0.05)),
    ("cmp_kd", dict(model_key="cmp", cmp_n_proto=2, cmp_agg="hardmax", cmp_kmeans_init=True)),
)
CE_S0 = {"resnet18": 0.9483, "asf": 0.9507, "cmp": 0.9503}  # từ 060a/060b seed 0
TEACHER_SEED = 2
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    EPOCHS = 1

# %%
# ---- ENV + manifest + teacher verify ----
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
# ---- KD screening loop ----
results = []
REPORT = "kd_screen_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
                   "epochs": EPOCHS, "kd": {"alpha": KD_ALPHA, "T": KD_T},
                   "teacher": {"seed": TEACHER_SEED, "sha256": h},
                   "manifest_sha256": audit["split_sha256"],
                   "controls_ce_s0": CE_S0, "results": results}, f, indent=1, ensure_ascii=False)

train_manifest = manifest[manifest.split == "train"].reset_index(drop=True)
for name, overrides in STUDENTS:
    set_seed(SEED)
    cfg = TrainConfig(num_classes=45, seed=SEED, epochs=EPOCHS, num_workers=NUM_WORKERS,
                      kd_alpha=KD_ALPHA, kd_T=KD_T, **overrides)
    student = build_model(cfg).to(device)
    kmeans_info = None
    if cfg.model_key == "cmp" and cfg.cmp_kmeans_init:
        kmeans_info = init_prototypes_kmeans(student, train_manifest, DATA_ROOT, device, seed=SEED,
                                             num_workers=NUM_WORKERS)
    params = count_params(student)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | params {params['total']} ===", flush=True)
    student, hist = train_model(student, cfg, dl_tr, dl_va, device,
                                state_out=f"ckpt_{name}.pt", teacher=teacher)
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
    ce_ref = CE_S0["resnet18" if name.startswith("resnet18") else "asf" if name.startswith("asf") else "cmp"]
    entry = {"student": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "delta_vs_CE_s0": round(m["macro_f1"] - ce_ref, 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "history": hist["history"],  # logging v2: per-epoch đầy đủ (CE/KD/aux)
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "prototype_usage": usage, "gate_summary": gate_summary,
             "kmeans_info": kmeans_info, "checkpoint": f"ckpt_{name}.pt"}
    results.append(entry)
    save_report()
    print(json.dumps({k: v for k, v in entry.items() if k != "history"}, ensure_ascii=False), flush=True)

# %%
# ---- Report + checks ----
report = json.load(open(REPORT))
ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

by = {r["student"]: r for r in results}
check("đủ 3 student", len(results) == 3)
if not SMOKE:
    rn = by["resnet18_kd"]
    check("RN18-KD > RN18-CE s0 (KD có tín hiệu trên control)",
          rn["best_val_macro_f1"] > CE_S0["resnet18"], f"{rn['best_val_macro_f1']} vs {CE_S0['resnet18']}")
    for k in ("asf_kd", "cmp_kd"):
        u_or_g = by[k].get("prototype_usage") or by[k].get("gate_summary")
        if k == "cmp_kd":
            check("CMP-KD không collapse", u_or_g["max_inter_proto_cos_mean"] < 0.95,
                  f"inter-cos={u_or_g['max_inter_proto_cos_mean']:.4f}")
        if k == "asf_kd":
            check("ASF-KD gate không sụp", u_or_g["max_mean"] < 0.9, f"max_mean={u_or_g['max_mean']}")
print("KD SCREENING", "PASS" if ok else "CÓ FAIL — đọc kết quả trước khi kết luận", flush=True)
