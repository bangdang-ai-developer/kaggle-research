# %% [md]
# # 120 — Hướng A Screening: CMP-exp (fix κ) & ASF-Lite (15 epoch, seed 0)
#
# **Mục tiêu:** kiểm chứng 2 cải tiến kiến trúc trực tiếp giải quyết failure modes đã chẩn đoán:
# 1. **CMP-exp (`cmp_scale_mode="exp"`):** sửa lỗi quên `.exp()` ở `logit_scale` (`κ = exp(θ) ∈ [10, 64]`)
#    + `hardmax` + `k-means init`, xóa bỏ trần softmax 13.32% dưới KD.
# 2. **ASF-Lite (`model_key="asf_lite"`):** Haar DWT trên `layer2` (32×32) & `layer3` (16×16),
#    dùng GAP(LL) + RMS(LH, HL, HH) không triệt tiêu dấu, bỏ 8 nhánh Conv2d+BN lặp để giữ latency ~ResNet18.
#
# **Controls seed 0 (cùng split/protocol):** B1 ResNet18-CE `0.9483` (060a), ResNet18-KD `0.9550` (110b).
# **Strict protocol:** KHÔNG chạm tập `test` trong screening; chỉ kiểm trên `val` + đo latency batch-1.

# %%
# ---- CONFIG ----
import os

RUN_ID = "120-h1a-screen-v1"
SEED = 0
EPOCHS = 15
KD_ALPHA, KD_T = 0.5, 4.0
EXPERIMENTS = (
    ("cmp_exp_ce", dict(model_key="cmp", cmp_n_proto=2, cmp_agg="hardmax", cmp_kmeans_init=True,
                        cmp_scale_mode="exp", kd_alpha=0.0)),
    ("cmp_exp_kd", dict(model_key="cmp", cmp_n_proto=2, cmp_agg="hardmax", cmp_kmeans_init=True,
                        cmp_scale_mode="exp", kd_alpha=KD_ALPHA, kd_T=KD_T)),
    ("asf_lite_ce", dict(model_key="asf_lite", asf_out_init="small", asf_gate_ent=0.05, kd_alpha=0.0)),
    ("asf_lite_kd", dict(model_key="asf_lite", asf_out_init="small", asf_gate_ent=0.05,
                         kd_alpha=KD_ALPHA, kd_T=KD_T)),
)
REF_S0 = {"ce": 0.9483, "kd": 0.9550}
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
audit_path = find_single(
    ["/kaggle/input/**/audit_report.json",
     "../../notebooks/010-data-audit/audit_report.smoke.json"], "audit report")
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
# ---- Screening loop ----
results = []
REPORT = "h1a_screen_report.json"


def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
                   "epochs": EPOCHS, "kd": {"alpha": KD_ALPHA, "T": KD_T},
                   "teacher": {"seed": TEACHER_SEED, "sha256": teacher_sha},
                   "manifest_sha256": audit["split_sha256"],
                   "ref_resnet18_s0": REF_S0, "results": results}, f, indent=1, ensure_ascii=False)


train_manifest = manifest[manifest.split == "train"].reset_index(drop=True)
for name, overrides in EXPERIMENTS:
    ck = f"ckpt_{name}.pt"
    set_seed(SEED)
    cfg = TrainConfig(num_classes=45, seed=SEED, epochs=EPOCHS, num_workers=NUM_WORKERS,
                      amp=device.startswith("cuda"), **overrides)
    student = build_model(cfg, pretrained=not SMOKE).to(device)
    kmeans_info = None
    if cfg.model_key == "cmp" and cfg.cmp_kmeans_init:
        kmeans_info = init_prototypes_kmeans(student, train_manifest, DATA_ROOT, device, seed=SEED,
                                             num_workers=NUM_WORKERS)
    params = count_params(student)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"\n=== {name} | params {params['total']} ===", flush=True)
    use_teacher = teacher if cfg.kd_alpha > 0 else None
    student, hist = train_model(student, cfg, dl_tr, dl_va, device,
                                state_out=ck, teacher=use_teacher)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(student, dl_va, device)
    lat = measure_latency(student, device, input_size=cfg.input_size,
                          n_warmup=2 if SMOKE else 10, n_iter=5 if SMOKE else 50)
    usage, gate_summary, kappa_val = None, None, None
    if cfg.model_key == "cmp":
        usage = student.prototype_usage(dl_va, device, max_batches=2 if SMOKE else None)
        kappa_val = round(float(student.current_kappa().item()), 4)
    if cfg.model_key == "asf_lite":
        xb_s, _ = next(iter(dl_va))
        gw = student.gate_weights(xb_s[:16].to(device)).cpu()
        gate_summary = {"gate_mean": [round(v, 4) for v in gw.mean(0).tolist()],
                        "gate_std": [round(v, 5) for v in gw.std(0).tolist()],
                        "max_mean": round(float(gw.mean(0).max()), 4)}
    ref = REF_S0["kd" if cfg.kd_alpha > 0 else "ce"]
    entry = {"name": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "delta_vs_resnet18_s0": round(m["macro_f1"] - ref, 4),
             "latency_ms": lat, "kappa": kappa_val,
             "history": hist["history"],
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "prototype_usage": usage, "gate_summary": gate_summary,
             "checkpoint": ck}
    results.append(entry)
    save_report()
    print(json.dumps({k: v for k, v in entry.items() if k not in ("history", "prototype_usage")},
                     ensure_ascii=False), flush=True)

# %%
# ---- Summary & checks ----
with open(REPORT) as f:
    rep = json.load(f)
assert len(rep["results"]) == len(EXPERIMENTS)
for r in rep["results"]:
    print(f"{r['name']:<14} | valF1 {r['best_val_macro_f1']:.4f} (ΔRN18 {r['delta_vs_resnet18_s0']:+.4f}) "
          f"| lat {r['latency_ms']['median_ms']:.2f}ms | kappa={r['kappa']} | params={r['params']['total']/1e6:.2f}M")
print("H1A SCREEN DONE")
