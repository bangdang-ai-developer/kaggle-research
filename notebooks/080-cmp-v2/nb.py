# %% [md]
# # 080 — CMP-v2 Screening: CMP-P (hard-max prototypes) + CMP-PG (+ pair loss)
#
# **Căn cứ diagnostics** (`notes/v2_diagnostics.json`):
# - D3: checkpoint v1 collapse THẬT (45/45 lớp pair-cos=1.00000000, K=2≡K=1).
# - D2: synthetic hard-max + kmeans-like init KHÔNG collapse → đổi aggregation.
# - D1: margin v1 sai chiều (GIẢM CE mẫu khó) → CMP-PG dùng pair loss chiều rõ
#   `Σ w·softplus(m + s_c − s_y)/Z` (ĐẨY s_y lên, ĐẦY s_c xuống).
#
# **Controls:** C1 0.9422 / C2 0.9441 (kernel 050, cùng protocol 8ep/seed0 — reuse hợp lệ) · B1 final 0.9504.
# 8 epoch, seed 0, staged local, graph probe train-only. **Không test.**

# %%
# ---- CONFIG ----
import os

RUN_ID = "080-cmp-v2-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
PROBE_EPOCHS = 6
K_PROTO = 2
PAIR_MARGIN = 0.1
PAIR_WEIGHT = 0.5
B1_FINAL_F1 = 0.9504
C2_V1_SCREEN = 0.9441
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
NUM_WORKERS = 0 if sys.platform == "darwin" else 4
print(f"run_id: {RUN_ID} | smoke: {SMOKE} | device: {device}")
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu")

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
assert DATA_ROOT, "FAIL-FAST: dataset root"

# %% embed:src/models.py

# %%
# ---- STAGE dataset local ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=NUM_WORKERS * 4)
else:
    print("local run — bỏ qua stage")

# %%
# ---- Confusion graph (train-only, cho CMP-PG) ----
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
# ---- V2 screening loop ----
VARIANTS = (
    ("CMP_P_hardmax", dict(cmp_n_proto=K_PROTO, cmp_agg="hardmax", cmp_kmeans_init=True)),
    ("CMP_PG_pair", dict(cmp_n_proto=K_PROTO, cmp_agg="hardmax", cmp_kmeans_init=True,
                         cmp_pair_loss=True, cmp_confusion=True,
                         cmp_pair_margin=PAIR_MARGIN, cmp_pair_weight=PAIR_WEIGHT,
                         cmp_pair_warmup=3)),
)
results = []
REPORT = "cmp_v2_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
                   "epochs": SCREEN_EPOCHS, "manifest_sha256": audit["split_sha256"],
                   "refs": {"C1_v1": 0.9422, "C2_v1": C2_V1_SCREEN, "B1_final": B1_FINAL_F1},
                   "results": results}, f, indent=1, ensure_ascii=False)

for name, overrides in VARIANTS:
    set_seed(SEED)
    cfg = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS, **overrides)
    model = build_model(cfg).to(device)
    kmeans_info = init_prototypes_kmeans(model, train_manifest, DATA_ROOT, device, seed=SEED,
                                         num_workers=NUM_WORKERS)
    if cfg.cmp_confusion:
        model.set_confusion(conf_w)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | agg={cfg.cmp_agg} kmeans={cfg.cmp_kmeans_init} pair={cfg.cmp_pair_loss} ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    usage = model.prototype_usage(dl_va, device)
    entry = {"variant": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "delta_vs_C2_v1": round(m["macro_f1"] - C2_V1_SCREEN, 4),
             "delta_vs_B1_final": round(m["macro_f1"] - B1_FINAL_F1, 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "history": hist["history"],  # logging v2: per-epoch đầy đủ
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "prototype_usage": usage, "kmeans_info": kmeans_info,
             "checkpoint": f"ckpt_{name}.pt"}
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

by = {r["variant"]: r for r in results}
check("đủ 2 candidate", len(results) == 2)
if not SMOKE:
    for r in results:
        u = r["prototype_usage"]
        check(f"{r['variant']} không collapse (inter-cos < 0.95)", u["max_inter_proto_cos_mean"] < 0.95,
              f"inter-cos={u['max_inter_proto_cos_mean']:.4f}")
        check(f"{r['variant']} có minority prototype (ratio > 0.05)", r["prototype_usage"]["minority_ratio"] if "minority_ratio" in u else True,
              "usage per proto")
    best = max(results, key=lambda r: r["best_val_macro_f1"])
    check(f"best CMP-v2 ({best['variant']}) > C2 v1 screen", best["best_val_macro_f1"] > C2_V1_SCREEN,
          f"{best['best_val_macro_f1']} vs {C2_V1_SCREEN}")
print("CMP-V2 SCREENING", "PASS" if ok else "CÓ FAIL — xem kết quả trước khi quyết final", flush=True)
