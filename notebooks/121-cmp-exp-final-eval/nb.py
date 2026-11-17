# %% [md]
# # 121 — CMP-exp Final (seeds 1, 2) + 3-seed TEST Evaluation
#
# **Theo `notes/final_plan_v4_cmp_exp.json` (đã khóa trước khi chạy test):**
# - Seed 0 đã train xong ở `120-h1a-screen` (`cmp_exp_ce` val F1 `0.9542`, `cmp_exp_kd` val F1 `0.9561`).
# - Run này train bổ sung seeds `(1, 2)` cho `cmp_exp_ce` và `cmp_exp_kd` (từ `train+val`, KHÔNG chạm `test`).
# - Sau khi đủ 3 seed `{0, 1, 2}`, đánh giá **1 lần duy nhất trên TEST** (4.725 ảnh) cho đủ 6 checkpoints,
#   đo latency batch-1 cùng máy với `ResNet18` baseline để đối chứng công bằng.

# %%
# ---- CONFIG ----
import os

RUN_ID = "121-cmp-exp-final-eval-v1"
TRAIN_SEEDS = (1, 2)
ALL_SEEDS = (0, 1, 2)
EPOCHS = 15
KD_ALPHA, KD_T = 0.5, 4.0
VARIANTS = (
    ("cmp_exp_ce", dict(model_key="cmp", cmp_n_proto=2, cmp_agg="hardmax", cmp_kmeans_init=True,
                        cmp_scale_mode="exp", kd_alpha=0.0)),
    ("cmp_exp_kd", dict(model_key="cmp", cmp_n_proto=2, cmp_agg="hardmax", cmp_kmeans_init=True,
                        cmp_scale_mode="exp", kd_alpha=KD_ALPHA, kd_T=KD_T)),
)
TEACHER_SEED = 2
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    TRAIN_SEEDS = (1,)
    ALL_SEEDS = (0, 1)
    EPOCHS = 1

# %%
# ---- ENV + manifest + teacher verify ----
import glob
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
full_manifest = pd.read_csv(manifest_path)
manifest = full_manifest[full_manifest.split != "test"].reset_index(drop=True)
test_manifest = full_manifest[full_manifest.split == "test"].reset_index(drop=True)

audit_path = find_single(
    ["/kaggle/input/**/audit_report.json",
     "../../notebooks/010-data-audit/audit_report.smoke.json"], "audit report")
with open(audit_path) as f:
    audit = json.load(f)
if not SMOKE:
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split lệch chuẩn"
print("manifest:", manifest_path, "| train+val:", len(manifest), "| test:", len(test_manifest))

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
    DATA_ROOT = stage_dataset(full_manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=NUM_WORKERS * 4)
else:
    print("local run — bỏ qua stage")

teacher_rel = f"ckpt_convnextt_s{TEACHER_SEED}.pt"
teacher_path = find_single([f"/kaggle/input/**/{teacher_rel}",
                            f"../../outputs/resisc45-final-baselines-b0-b4-3-seeds/{teacher_rel}"],
                           f"teacher {teacher_rel}")
with open(teacher_path, "rb") as tf:
    teacher_sha = hashlib.sha256(tf.read()).hexdigest()
if not SMOKE:
    assert teacher_sha.startswith("9627ca43"), "FAIL-FAST: teacher checkpoint lệch"
teacher = build_model(TrainConfig(num_classes=45, model_key="convnextt"), pretrained=False).to(device)
sd = torch.load(teacher_path, map_location=device, weights_only=False)
teacher.load_state_dict(sd["state_dict"])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"teacher verified: seed {TEACHER_SEED}, sha {teacher_sha[:16]}…, epoch {sd['epoch']}")

# %%
# ---- Train seeds (1, 2) ----
train_results = []
TRAIN_REPORT = "cmp_exp_final_report.json"


def save_train_report() -> None:
    with open(TRAIN_REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device,
                   "seeds": list(TRAIN_SEEDS), "epochs": EPOCHS,
                   "teacher": {"seed": TEACHER_SEED, "sha256": teacher_sha},
                   "manifest_sha256": audit["split_sha256"],
                   "results": train_results}, f, indent=1, ensure_ascii=False)


train_manifest = manifest[manifest.split == "train"].reset_index(drop=True)
ckpt_paths: dict[tuple[str, int], str] = {}

# Load seed 0 paths (từ 120-h1a-screen)
for name, _ov in VARIANTS:
    if SMOKE:
        # smoke tạo nhanh dummy ckpt s0 nếu không có ở local
        s0_local = f"../../outputs/resisc45-h1a-screen-cmp-exp-asf-lite/ckpt_{name}.pt"
        if os.path.exists(s0_local):
            ckpt_paths[(name, 0)] = s0_local
        else:
            ckpt_paths[(name, 0)] = f"ckpt_{name}_s0_smoke.pt"
            m_tmp = build_model(TrainConfig(num_classes=45, **_ov), pretrained=False)
            torch.save({"state_dict": m_tmp.state_dict(), "config": {}, "epoch": 0}, ckpt_paths[(name, 0)])
    else:
        ckpt_paths[(name, 0)] = find_single(
            [f"/kaggle/input/**/ckpt_{name}.pt",
             f"../../outputs/resisc45-h1a-screen-cmp-exp-asf-lite/ckpt_{name}.pt"],
            f"seed 0 ckpt_{name}.pt")
        print(f"[seed 0] mapped {name} -> {ckpt_paths[(name, 0)]}")

for seed in TRAIN_SEEDS:
    for name, overrides in VARIANTS:
        ck = f"ckpt_{name}_s{seed}.pt"
        ckpt_paths[(name, seed)] = ck
        set_seed(seed)
        cfg = TrainConfig(num_classes=45, seed=seed, epochs=EPOCHS, num_workers=NUM_WORKERS,
                          amp=device.startswith("cuda"), **overrides)
        student = build_model(cfg, pretrained=not SMOKE).to(device)
        kmeans_info = init_prototypes_kmeans(student, train_manifest, DATA_ROOT, device, seed=seed,
                                             num_workers=NUM_WORKERS)
        params = count_params(student)
        dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\n=== {name} seed={seed} | params {params['total']} ===", flush=True)
        use_teacher = teacher if cfg.kd_alpha > 0 else None
        student, hist = train_model(student, cfg, dl_tr, dl_va, device, state_out=ck, teacher=use_teacher)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        m = evaluate(student, dl_va, device)
        usage = student.prototype_usage(dl_va, device, max_batches=2 if SMOKE else None)
        entry = {"variant": name, "seed": seed, "config": cfg.to_dict(), "params": params,
                 "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
                 "kappa": round(float(student.current_kappa().item()), 4),
                 "history": hist["history"], "train_time_sec": hist["train_time_sec"],
                 "best_epoch": hist["best_epoch"], "executed_epochs": hist["executed_epochs"],
                 "peak_mem_gib": peak, "prototype_usage": usage, "checkpoint": ck}
        train_results.append(entry)
        save_train_report()
        print(json.dumps({k: v for k, v in entry.items() if k not in ("history", "prototype_usage")},
                         ensure_ascii=False), flush=True)

# %%
# ---- 3-seed TEST Evaluation + paired latency ----
dl_test = DataLoader(ResiscDataset(test_manifest, DATA_ROOT, False, 256),
                     batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

test_results = []
for name, overrides in VARIANTS:
    for seed in ALL_SEEDS:
        ck = ckpt_paths[(name, seed)]
        cfg = TrainConfig(num_classes=45, seed=seed, **overrides)
        model = build_model(cfg, pretrained=False).to(device)
        sd_ck = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd_ck["state_dict"], strict=True)
        m = evaluate(model, dl_test, device)
        lat = measure_latency(model, device, input_size=256,
                              n_warmup=2 if SMOKE else 10, n_iter=5 if SMOKE else 50)
        res = {"variant": name, "seed": seed,
               "test_acc": round(m["acc"], 4),
               "test_macro_precision": round(m["macro_precision"], 4),
               "test_macro_recall": round(m["macro_recall"], 4),
               "test_macro_f1": round(m["macro_f1"], 4),
               "n": m["n"], "kappa": round(float(model.current_kappa().item()), 4),
               "latency": lat, "checkpoint": os.path.basename(ck)}
        test_results.append(res)
        print(f"[TEST] {name} s{seed}: acc={res['test_acc']:.4f} f1={res['test_macro_f1']:.4f} "
              f"lat={lat['median_ms']:.2f}ms kappa={res['kappa']}", flush=True)

# Same-machine latency reference: ResNet18 s2
rn18_path = find_single(["/kaggle/input/**/ckpt_resnet18_s2.pt",
                         "../../outputs/resisc45-final-baselines-b0-b4-3-seeds/ckpt_resnet18_s2.pt"],
                        "ckpt_resnet18_s2.pt")
rn18 = build_model(TrainConfig(num_classes=45, model_key="resnet18"), pretrained=False).to(device)
rn18.load_state_dict(torch.load(rn18_path, map_location=device, weights_only=False)["state_dict"], strict=True)
rn18_lat = measure_latency(rn18, device, input_size=256, n_warmup=2 if SMOKE else 10, n_iter=5 if SMOKE else 50)

summary = {}
for name, _ov in VARIANTS:
    rows = [r for r in test_results if r["variant"] == name]
    accs = [r["test_acc"] for r in rows]
    f1s = [r["test_macro_f1"] for r in rows]
    lats = [r["latency"]["median_ms"] for r in rows]
    summary[name] = {
        "n_seeds": len(rows),
        "test_acc": {"mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs, ddof=1)), 4)},
        "test_macro_f1": {"mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s, ddof=1)), 4)},
        "latency_median_ms": {"mean": round(float(np.mean(lats)), 2), "std": round(float(np.std(lats, ddof=1)), 2)},
    }

TEST_REPORT = "cmp_exp_test_3seed_report.json"
with open(TEST_REPORT, "w") as f:
    json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device,
               "manifest_sha256": audit["split_sha256"],
               "same_machine_resnet18_latency": rn18_lat,
               "results": test_results, "summary": summary}, f, indent=1, ensure_ascii=False)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("ResNet18 same-machine latency:", rn18_lat["median_ms"])
