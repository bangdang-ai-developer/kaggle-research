# %% [md]
# # 050b — CMP Fix Screening (sau negative result 050: prototype collapse inter-cos=1.0)
#
# **Bằng chứng 050:** K=2 prototype trùng tuyệt đối (không có lực phân tách); C3 margin 0.3 làm hại.
#
# **2 cấu hình fix (trần tuning ≤2/proposed):**
# - **C2fix**: k-means init + diversity reg 0.01 (C4)
# - **C3fix**: C2fix + confusion margin **0.1** (giảm từ 0.3)
#
# Đối chiếu: C2 cũ 0.9441, B1 0.9448. Thành công tối thiểu: inter-cos < 0.95 (không trùng) và F1 > B1.
# Probe graph rebuild trong kernel (train-only, ~5 phút). 8 epoch, seed 0. **Không test.**

# %%
# ---- CONFIG ----
import os

RUN_ID = "050b-cmp-fix-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
PROBE_EPOCHS = 6
K_PROTO = 2
CONF_MARGIN_FIX = 0.1
B1_F1 = 0.9448
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SCREEN_EPOCHS = 1
    PROBE_EPOCHS = 1

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
# ---- C3fix: confusion graph từ probe (train-only) ----
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
train_manifest = manifest[manifest.split == "train"].reset_index(drop=True)
t0 = time.time()
conf_w, graph_info = build_confusion_graph(train_manifest, DATA_ROOT, graph_frac=0.10, seed=SEED,
                                           probe_epochs=PROBE_EPOCHS, device=device, num_workers=NUM_WORKERS)
graph_time = time.time() - t0
print(f"confusion graph: {graph_time:.0f}s | {graph_info}")
with open("confusion_graph.json", "w") as f:
    json.dump({"shape": list(conf_w.shape), "info": graph_info,
               "top_pairs": [{"a": int(i), "b": int(j), "w": float(conf_w[i, j])}
                             for i, j in zip(*np.nonzero(conf_w))]}, f, indent=1)

# %%
# ---- Fix screening loop ----
VARIANTS = (
    ("C2fix_kmeans_div", dict(cmp_n_proto=K_PROTO, cmp_kmeans_init=True, cmp_diversity=True,
                              cmp_confusion=False)),
    ("C3fix_kmeans_div_m01", dict(cmp_n_proto=K_PROTO, cmp_kmeans_init=True, cmp_diversity=True,
                                  cmp_confusion=True, cmp_conf_margin=CONF_MARGIN_FIX)),
)
results = []
for name, overrides in VARIANTS:
    set_seed(SEED)
    cfg = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS, **overrides)
    model = build_model(cfg).to(device)
    # k-means init trên embedding pretrained của TRAIN (eval transform) — chống collapse
    kmeans_info = init_prototypes_kmeans(model, train_manifest, DATA_ROOT, device, seed=SEED,
                                         num_workers=NUM_WORKERS)
    if cfg.cmp_confusion:
        model.set_confusion(conf_w)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | K={cfg.cmp_n_proto} margin={cfg.cmp_conf_margin} confusion={cfg.cmp_confusion} ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    usage = model.prototype_usage(dl_va, device)
    entry = {"variant": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "delta_vs_B1": round(m["macro_f1"] - B1_F1, 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "prototype_usage": usage, "kmeans_info": kmeans_info,
             "checkpoint": f"ckpt_{name}.pt",
             "graph_time_sec": round(graph_time, 1) if cfg.cmp_confusion else None}
    results.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Report + fail-fast ----
report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
          "manifest_sha256": audit["split_sha256"], "screening_epochs": SCREEN_EPOCHS,
          "B1_baseline_f1": B1_F1, "conf_margin_fix": CONF_MARGIN_FIX,
          "context": "050 negative: prototype collapse inter-cos=1.0; margin 0.3 hại",
          "results": results}
with open("cmp_fix_report.json", "w") as f:
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
        u = r["prototype_usage"]
        ic = u["max_inter_proto_cos_mean"]
        check(f"{r['variant']} không collapse (inter-cos < 0.95)", ic is not None and ic < 0.95, f"inter-cos={ic}")
    best = max(results, key=lambda r: r["best_val_macro_f1"])
    check(f"best fix ({best['variant']}) > B1", best["best_val_macro_f1"] > B1_F1,
          f"{best['best_val_macro_f1']} vs {B1_F1}")
print("CMP FIX SCREENING", "PASS" if ok else "CÓ FAIL", flush=True)
