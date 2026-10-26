# %% [md]
# # 050 — CMP Ablation Screening (C1 / C2 / C3)
#
# **Mục tiêu:** tách đóng góp của (a) cosine classifier + multi-prototype, (b) confusion graph train-only.
#
# - **C1**: cosine classifier 1 prototype/lớp (đối chứng trực tiếp của head)
# - **C2**: K=2 prototype, không confusion
# - **C3**: C2 + confusion graph từ probe (train-only, 90/10 fit/graph, 6 epoch probe; chi phí probe tính vào phương pháp)
#
# C0 = ResNet18 classifier gốc = B1 ở 030. 8 epoch, seed 0. **Không test.** Chi phí probe mini đo ở pilot: ~5 phút full.

# %%
# ---- CONFIG ----
import os

RUN_ID = "050-cmp-ablation-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
PROBE_EPOCHS = 6
K_PROTO = 2
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SCREEN_EPOCHS = 1
    PROBE_EPOCHS = 1

# %%
# ---- ENV + manifest + FAIL-FAST checksum ----
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
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split manifest lệch chuẩn project"
print("manifest:", manifest_path, "| rows:", len(manifest), "| split sha256:", audit["split_sha256"][:16])
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    if os.path.isdir(hit) and len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))]) >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, f"FAIL-FAST: dataset root; /kaggle/input = {os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'KHONG CO'}"

# %% embed:src/models.py

# %%
# ---- STAGE dataset local ----
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(manifest, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
else:
    print("local run — bỏ qua stage")

# %%
# ---- C3: confusion graph từ probe (TRAIN-ONLY, không val/test) ----
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
# ---- Ablation loop C1/C2/C3 ----
VARIANTS = (
    ("C1_cosine_k1", dict(cmp_n_proto=1, cmp_confusion=False)),
    ("C2_proto_k2", dict(cmp_n_proto=K_PROTO, cmp_confusion=False)),
    ("C3_proto_confusion", dict(cmp_n_proto=K_PROTO, cmp_confusion=True)),
)
results = []
for name, overrides in VARIANTS:
    set_seed(SEED)
    cfg = TrainConfig(model_key="cmp", num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS, **overrides)
    model = build_model(cfg).to(device)
    if cfg.cmp_confusion:
        model.set_confusion(conf_w)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | params {params} | K={cfg.cmp_n_proto} confusion={cfg.cmp_confusion} ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    usage = model.prototype_usage(dl_va, device, max_batches=None if not SMOKE else 4)  # full val: đủ phủ 45 lớp
    entry = {"variant": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "prototype_usage": usage, "checkpoint": f"ckpt_{name}.pt",
             "graph_time_sec": round(graph_time, 1) if cfg.cmp_confusion else None,
             "graph_info": graph_info if cfg.cmp_confusion else None}
    results.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

# %%
# ---- Report + fail-fast ----
report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
          "manifest_sha256": audit["split_sha256"], "screening_epochs": SCREEN_EPOCHS,
          "note": "SCREENING — final ở 060; C0 (classifier gốc) = B1 ở 030; graph cost tính vào C3",
          "results": results}
with open("cmp_ablation_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

by = {r["variant"]: r for r in results}
check("đủ 3 biến thể", len(results) == 3)
if not SMOKE:
    # collapse check trên full val — ở SMOKE (1 batch) luôn có unused do lớp vắng
    check("C2 không collapse (0 unused prototype)",
          by["C2_proto_k2"]["prototype_usage"]["unused_proto_classes"] == 0)
check("C3 graph đã tạo từ train-only", by["C3_proto_confusion"]["graph_info"]["n_graph"] > 0)
print("CMP ABLATION SCREENING", "PASS" if ok else "CÓ FAIL", flush=True)
