# %% [md]
# # 070 — Final Evaluation: Test + Latency + Robustness
#
# **Chạy MỘT LẦN duy nhất sau khi `final_plan.json` khóa và 060a/060b hoàn thành** (protocol mục 9).
#
# 1. **Test metrics** 7 model (checkpoint best-val, ghi rõ seed): ACC, macro P/R/F1, per-class + support, confusion matrix, predictions per-image.
# 2. **Latency** batch-1 (median/p95, warmup + sync) + throughput batch-256.
# 3. **Robustness test**: blur σ∈{1,2}, JPEG q∈{30,60}, brightness ×{0.6,1.4} — cùng biến đổi cho mọi model, báo clean vs delta (synthetic corruption ≠ deployment ngoài đời).
#
# Đầu vào: checkpoints từ 060a/060b (kernel_sources), manifest chuẩn `cc70020a`.

# %%
# ---- CONFIG ----
import os

RUN_ID = "070-final-evaluation-v1"
SPLIT_SHA_EXPECTED = "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a"
CORRUPTIONS = (("blur", 1.0), ("blur", 2.0), ("jpeg", 30), ("jpeg", 60), ("bright", 0.6), ("bright", 1.4))
SMOKE = os.environ.get("PILOT_SMOKE", "0" if os.path.isdir("/kaggle/input") else "1") == "1"

# %%
# ---- ENV + manifest + reports + chọn best seed ----
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

def find_all(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern, recursive=True))

manifest_path = find_all("/kaggle/input/**/split_manifest.csv") or find_all("../../notebooks/010-data-audit/split_manifest.smoke.csv")
manifest = pd.read_csv(manifest_path[0])
audit_path = find_all("/kaggle/input/**/audit_report.json") or find_all("../../notebooks/010-data-audit/audit_report.smoke.json")
with open(audit_path[0]) as f:
    audit = json.load(f)
co_path = find_all("/kaggle/input/**/class_order.json") or find_all("../../notebooks/010-data-audit/class_order.smoke.json")
CLASS_ORDER = json.load(open(co_path[0]))
CLASSES = [c for c, i in sorted(CLASS_ORDER.items(), key=lambda kv: kv[1])]  # 45 lớp theo index chuẩn
if not SMOKE:
    assert audit["split_sha256"] == SPLIT_SHA_EXPECTED, "FAIL-FAST: split lệch final_plan"
DATA_ROOT = None
for hit in sorted(glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)) + [
        "../../data/raw/punfake---resisc45/NWPU-RESISC45"]:
    if os.path.isdir(hit) and len([d for d in os.listdir(hit) if os.path.isdir(os.path.join(hit, d))]) >= 40:
        DATA_ROOT = hit
        break
assert DATA_ROOT, "FAIL-FAST: dataset root"

# reports 060a/060b (từ kernel_sources) hoặc local fallback
rep_b = find_all("/kaggle/input/**/final_baselines_report.json") or find_all("../../outputs/resisc45-final-baselines-b0-b4-3-seeds/final_baselines_report.json")
rep_p = find_all("/kaggle/input/**/final_proposed_report.json") or find_all("../../outputs/resisc45-final-proposed-asf-cmp-3-seeds/final_proposed_report.json")
assert rep_b and rep_p, "FAIL-FAST: thiếu final reports (060a/060b)"
rep_b = json.load(open(rep_b[0]))
rep_p = json.load(open(rep_p[0]))

# bảng model: display_key → (model_key, overrides, ckpt glob, best seed theo val F1)
all_runs = {f"{e['model']}_s{e['seed']}": (e["model"], e) for e in rep_b["results"]}
all_runs.update({f"{e['variant']}_s{e['seed']}": (e["variant"], e) for e in rep_p["results"]})
MODEL_TABLE = {}
for run_key, (mkey, entry) in all_runs.items():
    if mkey == "smallcnn":
        mk, ov = "smallcnn", {}
    elif mkey == "resnet18":
        mk, ov = "resnet18", {}
    elif mkey == "mobilenetv3s":
        mk, ov = "mobilenetv3s", {}
    elif mkey == "efficientnetb0":
        mk, ov = "efficientnetb0", {}
    elif mkey == "convnextt":
        mk, ov = "convnextt", {}
    elif mkey == "ASF_A2fix_ent":
        mk, ov = "asf", dict(asf_out_init="small", asf_gate_ent=0.05)
    elif mkey == "CMP_C2":
        mk, ov = "cmp", dict(cmp_n_proto=2)
    else:
        continue
    MODEL_TABLE[run_key] = {"model_key": mk, "overrides": ov, "val_f1": entry["best_val_macro_f1"],
                            "seed": entry["seed"], "rel_ckpt": f"ckpt_{mkey}_s{entry['seed']}.pt"
                            if mkey in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt")
                            else f"ckpt_{mkey}_s{entry['seed']}.pt"}
# mỗi model giữ đúng 1 seed: best val F1
best = {}
for run_key, info in MODEL_TABLE.items():
    bare = run_key.rsplit("_s", 1)[0]
    if bare not in best or info["val_f1"] > best[bare]["val_f1"]:
        best[bare] = {**info, "run_key": run_key}
print(f"{len(best)} model (best-val seed):")
for bare, info in best.items():
    print(f"  {bare}: {info['run_key']} val_f1={info['val_f1']}")

# %% embed:src/models.py

# %%
# ---- STAGE FULL dataset (lần đầu cần cả TEST) ----
full = pd.read_csv(manifest_path[0])  # train+val+test
if DATA_ROOT.startswith("/kaggle/"):
    DATA_ROOT = stage_dataset(full, DATA_ROOT, "/tmp/resisc45_staged/NWPU-RESISC45", num_threads=16)
else:
    print("local run — bỏ qua stage")

# %%
# ---- Load checkpoints ----
def find_ckpt(rel: str) -> str:
    hits = (find_all(f"/kaggle/input/**/{rel}") or find_all(f"outputs/**/{rel}")
            or find_all(f"../../outputs/**/{rel}"))
    assert hits, f"FAIL-FAST: thiếu checkpoint {rel}"
    return hits[0]

loaded = {}
for bare, info in best.items():
    cfg = TrainConfig(num_classes=45, seed=info["seed"], model_key=info["model_key"], **info["overrides"])
    model = build_model(cfg).to(device)
    ck_path = find_ckpt(info["rel_ckpt"])
    sd = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    loaded[bare] = model
    print(f"loaded {bare} ← {ck_path} (epoch {sd.get('epoch')})")

# %%
# ---- Test DataLoader + corrupt helper ----
import io

from PIL import Image, ImageEnhance, ImageFilter

test_manifest = full[full.split == "test"].reset_index(drop=True)
print("test images:", len(test_manifest))
assert len(test_manifest) == 4725 or SMOKE

def corrupt_pil(img: Image.Image, kind, param) -> Image.Image:
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(param))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "bright":
        return ImageEnhance.Brightness(img).enhance(param)
    raise ValueError(kind)

class CorruptDataset(torch.utils.data.Dataset):
    """Test dataset với corruption on-the-fly (eval transform sau corruption)."""
    def __init__(self, manifest_df: pd.DataFrame, root: str, kind=None, param=None):
        self.items = manifest_df
        self.root = root
        self.kind, self.param = kind, param

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        r = self.items.iloc[i]
        img = Image.open(os.path.join(self.root, r.image_id)).convert("RGB")
        if self.kind is not None:
            img = corrupt_pil(img, self.kind, self.param)
        return eval_augment(img, 256), int(r.class_index)

@torch.inference_mode()
def predict(model: nn.Module, ds: torch.utils.data.Dataset, device: str, batch: int = 256):
    """Trả về (y_true, y_pred, probs) toàn bộ dataset."""
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0 if sys.platform == "darwin" else 4)
    ys, ps, pr = [], [], []
    for x, y in dl:
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            logits = model(x.to(device))
        pr.append(F.softmax(logits.float(), dim=1).cpu())
        ps.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(pr).numpy()

# %%
# ---- 1) TEST metrics 7 model (một lần duy nhất) ----
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

test_ds = CorruptDataset(test_manifest, DATA_ROOT)
test_results, all_predictions = {}, {}
t0 = time.time()
for bare, model in loaded.items():
    y, p, probs = predict(model, test_ds, device)
    ncls = len(CLASSES)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y, p, average="macro", labels=list(range(ncls)), zero_division=0)
    acc = float((y == p).mean())
    pp, rr, ff, sup = precision_recall_fscore_support(y, p, average=None, labels=list(range(ncls)), zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(ncls)))
    test_results[bare] = {"acc": round(acc, 4), "macro_precision": round(float(p_m), 4),
                          "macro_recall": round(float(r_m), 4), "macro_f1": round(float(f1_m), 4),
                          "per_class": {"precision": [round(float(v), 4) for v in pp],
                                        "recall": [round(float(v), 4) for v in rr],
                                        "f1": [round(float(v), 4) for v in ff],
                                        "support": sup.tolist()},
                          "confusion_matrix": cm.tolist(), "seed": best[bare]["seed"]}
    all_predictions[bare] = (y, p, probs)
    print(f"[test] {bare}: acc={acc:.4f} macroF1={f1_m:.4f} ({time.time()-t0:.0f}s elapsed)", flush=True)

# predictions CSV + probs NPZ
pred_df = test_manifest[["image_id", "class_name"]].copy()
pred_df["true_index"] = all_predictions["resnet18"][0]
for bare, (y, p, probs) in all_predictions.items():
    pred_df[f"pred_{bare}"] = p
    pred_df[f"correct_{bare}"] = (p == y)
    np.savez_compressed(f"test_probs_{bare}.npz", probs=probs.astype(np.float16))
pred_df.to_csv("test_predictions.csv", index=False)

# confusion matrices PNG (CLASSES = 45 lớp từ class_order — không shadow bằng manifest subset)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for bare in loaded:
    fig, ax = plt.subplots(figsize=(11, 9))
    cm = np.array(test_results[bare]["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(45)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=5)
    ax.set_yticks(range(45)); ax.set_yticklabels(CLASSES, fontsize=5)
    ax.set_title(f"{bare} — test confusion (row-normalized), acc={test_results[bare]['acc']:.4f}", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"confusion_{bare}.png", dpi=110)
    plt.close(fig)
print(f"TEST done {time.time()-t0:.0f}s")

# %%
# ---- 2) Latency + throughput (checkpoint final) ----
lat = {}
for bare, model in loaded.items():
    lat[bare] = measure_latency(model, device, n_iter=10 if device == "cpu" else 50)
    dl_b = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
    n = 0; t1 = time.time()
    with torch.inference_mode():
        for bi, (x, _) in enumerate(dl_b):
            if bi >= 5: break
            model(x.to(device)); n += len(x)
    torch.cuda.synchronize() if device == "cuda" else None
    lat[bare]["throughput_imgs_sec"] = round(n / (time.time() - t1), 1)
    print(f"[lat] {bare}: {lat[bare]}", flush=True)

# %%
# ---- 3) Robustness: 6 corruptions × 7 model ----
rob = {}
t0 = time.time()
for kind, param in CORRUPTIONS:
    ds_c = CorruptDataset(test_manifest, DATA_ROOT, kind, param)
    rob[f"{kind}_{param}"] = {}
    for bare, model in loaded.items():
        y, p, _ = predict(model, ds_c, device)
        acc = float((y == p).mean())
        f1 = precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]
        rob[f"{kind}_{param}"][bare] = {"acc": round(acc, 4), "macro_f1": round(float(f1), 4),
                                        "acc_delta": round(acc - test_results[bare]["acc"], 4)}
    print(f"[rob] {kind}_{param} xong ({time.time()-t0:.0f}s)", flush=True)

# bảng delta tổng hợp
rob_table = pd.DataFrame({k: {m: v["acc"] for m, v in d.items()} for k, d in rob.items()})
rob_table["clean"] = [test_results[m]["acc"] for m in rob_table.index]
print(rob_table.round(4).to_string())
rob_table.to_csv("robustness_table.csv")

# %%
# ---- Report tổng + fail-fast ----
report = {"run_id": RUN_ID, "smoke": SMOKE, "device": device,
          "manifest_sha256": audit["split_sha256"],
          "model_selection": {b: {"run_key": i["run_key"], "seed": i["seed"], "val_f1": i["val_f1"]}
                              for b, i in best.items()},
          "test_results": test_results, "latency": lat, "robustness": rob,
          "note": "Test chạy MỘT lần sau khi final_plan.json khóa; corruption là synthetic — không chứng minh deployment ngoài đời"}
with open("test_report.json", "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)

ok = True
def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    print(("PASS" if cond else "FAIL"), name, detail)
    ok = ok and cond

if not SMOKE:
    check("đủ 7 model", len(test_results) == 7, f"{len(test_results)}")
    check("test đủ 4725 ảnh", all(r["confusion_matrix"] and sum(sum(row) for row in r["confusion_matrix"]) == 4725
          for r in test_results.values()))
    check("đủ 6 corruption × 7 model", len(rob) == 6 and all(len(d) == 7 for d in rob.values()))
    check("latency đo được mọi model", all(v["median_ms"] > 0 for v in lat.values()))
print("FINAL EVALUATION", "PASS" if ok else "CÓ FAIL", flush=True)
