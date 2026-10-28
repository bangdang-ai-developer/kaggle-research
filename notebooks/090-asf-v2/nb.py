# %% [md]
# # 090 — ASF-v2 Screening: ASF-E (energy gate) + ASF-EA (entropy anneal) + control
#
# **Căn cứ diagnostics** (`notes/v2_diagnostics.json`):
# - D4b: signed-GAP của detail subband triệt tiêu dấu (LH4 0.054 vs LL4 1.025 — 19×; RMS chỉ 4.3×)
#   → gate từ GAP(ReLU(proj)) bỏ lỡ thông tin độ lớn → **ASF-E**: gate từ [RMS, mean|·|] 16-dim → BN → Linear.
# - ASF-EA: entropy anneal λ=0.05 → 0 sau 8 epoch (giữ gate sống đầu, thả phân hóa sau).
# - D4a: residual/backbone = 0.37 ở v1 — nhánh KHÔNG chết;Gate gần đều là điểm cần cải thiện.
#
# **Controls:** A1fix-small (fixed fusion, cùng init — uniform control cùng recipe) · ASF-v1 final 0.9543 · B1 0.9504.
# 8 epoch, seed 0, staged local. **Không test.**

# %%
# ---- CONFIG ----
import os

RUN_ID = "090-asf-v2-screen-v1"
SEED = 0
SCREEN_EPOCHS = 8
VARIANTS = (
    ("A1fix_small_control", dict(asf_fixed_fusion=True, asf_out_init="small")),
    ("ASF_E_energy", dict(asf_gate_descriptor="energy", asf_out_init="small", asf_gate_ent=0.05)),
    ("ASF_EA_anneal", dict(asf_gate_descriptor="energy", asf_out_init="small",
                           asf_gate_ent=0.05, asf_gate_ent_anneal=8)),
)
B1_FINAL_F1 = 0.9504
ASF_V1_FINAL_F1 = 0.9543
A2FIX_ENT_SCREEN = 0.9472
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"
if SMOKE:
    SCREEN_EPOCHS = 1

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
# ---- V2 screening loop + gate/residual diagnostics ----
results = []
REPORT = "asf_v2_report.json"

def save_report() -> None:
    with open(REPORT, "w") as f:
        json.dump({"run_id": RUN_ID, "smoke": SMOKE, "device": device, "seed": SEED,
                   "epochs": SCREEN_EPOCHS, "manifest_sha256": audit["split_sha256"],
                   "refs": {"B1_final": B1_FINAL_F1, "ASF_v1_final": ASF_V1_FINAL_F1,
                            "A2fix_ent_screen": A2FIX_ENT_SCREEN},
                   "results": results}, f, indent=1, ensure_ascii=False)

xb = None
for name, overrides in VARIANTS:
    set_seed(SEED)
    cfg = TrainConfig(model_key="asf", num_classes=45, seed=SEED, epochs=SCREEN_EPOCHS,
                      num_workers=NUM_WORKERS, **overrides)
    model = build_model(cfg).to(device)
    params = count_params(model)
    dl_tr, dl_va = make_loaders(manifest, DATA_ROOT, cfg)
    if xb is None:
        xb, _ = next(iter(dl_va))
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    print(f"\n=== {name} | descriptor={cfg.asf_gate_descriptor} fixed={cfg.asf_fixed_fusion} "
          f"ent={cfg.asf_gate_ent} anneal={cfg.asf_gate_ent_anneal} ===", flush=True)
    model, hist = train_model(model, cfg, dl_tr, dl_va, device, state_out=f"ckpt_{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    m = evaluate(model, dl_va, device)
    gate_summary, residual_ratio = None, None
    if not cfg.asf_fixed_fusion:
        with torch.no_grad():
            model.eval()
            f2, f4 = model._backbone_feats(xb[:16].to(device))
            raw = model._raw_components(f2, f4)
            feats = [model.pool(model.proj[i](c)).flatten(1) for i, c in enumerate(raw)]
            stack = torch.stack(feats, 1)
            gw = torch.softmax(model._gate_input(raw, stack), -1).cpu()
            fused = (stack * gw.unsqueeze(-1).to(stack.device)).sum(1)
            bb_vec = model.pool(f4).flatten(1)
            res_vec = model.out_proj(fused)
            residual_ratio = float((res_vec.norm(dim=1) / bb_vec.norm(dim=1)).mean())
        gate_summary = {"gate_mean_per_component": [round(v, 4) for v in gw.mean(0).tolist()],
                        "gate_std_across_images": [round(v, 5) for v in gw.std(0).tolist()],
                        "max_mean_component": round(float(gw.mean(0).max()), 4),
                        "residual_backbone_norm_ratio": round(residual_ratio, 4),
                        "components": ["LL2", "LH2", "HL2", "HH2", "LL4", "LH4", "HL4", "HH4"]}
    entry = {"variant": name, "config": cfg.to_dict(), "params": params,
             "best_val_acc": round(m["acc"], 4), "best_val_macro_f1": round(m["macro_f1"], 4),
             "delta_vs_B1_final": round(m["macro_f1"] - B1_FINAL_F1, 4),
             "sec_per_epoch": [h["epoch_sec"] for h in hist["history"]],
             "history": hist["history"],  # logging v2: per-epoch đầy đủ
             "train_time_sec": hist["train_time_sec"], "best_epoch": hist["best_epoch"],
             "executed_epochs": hist["executed_epochs"], "peak_mem_gib": peak,
             "gate_summary": gate_summary, "checkpoint": f"ckpt_{name}.pt"}
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
check("đủ 3 cấu hình (2 candidate + control)", len(results) == 3)
if not SMOKE:
    for r in results:
        if r["gate_summary"]:
            gs = r["gate_summary"]
            check(f"{r['variant']} gate không sụp (max mean < 0.9)", gs["max_mean_component"] < 0.9,
                  f"max_mean={gs['max_mean_component']}")
    ctrl = by["A1fix_small_control"]["best_val_macro_f1"]
    best = max(by[k]["best_val_macro_f1"] for k in ("ASF_E_energy", "ASF_EA_anneal"))
    check("best ASF-v2 > control fixed cùng recipe", best > ctrl, f"{best:.4f} vs {ctrl:.4f}")
    check("best ASF-v2 > A2fix-ent v1 screen", best > A2FIX_ENT_SCREEN, f"{best:.4f} vs {A2FIX_ENT_SCREEN}")
print("ASF-V2 SCREENING", "PASS" if ok else "CÓ FAIL — xem kết quả trước khi quyết final", flush=True)
