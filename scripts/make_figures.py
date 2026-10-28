#!/usr/bin/env python3
"""Sinh hình báo cáo từ artifacts đã pull (không số giả — đọc trực tiếp JSON/CSV).

Chạy: uv run python3 scripts/make_figures.py
Output: notes/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "notes" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OUT_B = ROOT / "outputs" / "resisc45-final-baselines-b0-b4-3-seeds"
OUT_P = ROOT / "outputs" / "resisc45-final-proposed-asf-cmp-3-seeds"
OUT_E = ROOT / "outputs" / "resisc45-final-evaluation-v2-offline"

rb = json.load(open(OUT_B / "final_baselines_report.json"))
rp = json.load(open(OUT_P / "final_proposed_report.json"))
re_ = json.load(open(OUT_E / "test_report.json"))

DISPLAY = {"smallcnn": "B0 SmallCNN", "resnet18": "B1 ResNet18", "mobilenetv3s": "B2 MobileNetV3-S",
           "efficientnetb0": "B3 EffNet-B0", "convnextt": "B4 ConvNeXt-T",
           "ASF_A2fix_ent": "ASF (proposed)", "CMP_C2": "CMP (proposed)"}
ORDER = ["smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt", "ASF_A2fix_ent", "CMP_C2"]
COL = {"B1": "#4c78a8", "proposed": "#e45756", "other": "#9aa0a6"}


def color_of(key: str) -> str:
    if key == "resnet18":
        return COL["B1"]
    if key in ("ASF_A2fix_ent", "CMP_C2"):
        return COL["proposed"]
    return COL["other"]


# 1) Val F1 mean ± std — bar
fig, ax = plt.subplots(figsize=(9, 4.5))
keys = [k for k in ORDER if k in DISPLAY]
means, stds, colors = [], [], []
for k in keys:
    rows = ([e for e in rb["results"] if e["model"] == k] if k in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt")
            else [e for e in rp["results"] if e["variant"] == k])
    f1s = [e["best_val_macro_f1"] for e in rows]
    means.append(float(np.mean(f1s)))
    stds.append(float(np.std(f1s, ddof=1)))
    colors.append(color_of(k))
bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=4, color=colors)
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([DISPLAY[k] for k in keys], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Val macro-F1 (3 seeds)")
ax.set_title("Final validation macro-F1, mean ± std over seeds {0,1,2} (15 epochs)")
ax.set_ylim(0.5, 1.0)
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width() / 2, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_val_f1_final.png", dpi=150)
plt.close(fig)

# 2) TEST F1 vs latency — scatter
fig, ax = plt.subplots(figsize=(7.5, 5))
for k in keys:
    t = re_["test_results"][k]
    l = re_["latency"][k]
    ax.scatter(l["median_ms"], t["macro_f1"], s=110, color=color_of(k), zorder=3)
    ax.annotate(DISPLAY[k], (l["median_ms"], t["macro_f1"]), textcoords="offset points",
                xytext=(7, 4), fontsize=8)
ax.set_xlabel("Latency batch-1 (ms, median)")
ax.set_ylabel("TEST macro-F1")
ax.set_title("TEST macro-F1 vs latency (4,725 test images)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_test_f1_latency.png", dpi=150)
plt.close(fig)

# 3) Robustness delta — grouped bars (5 model chính, bỏ B0 cho đọc)
rob = pd.read_csv(OUT_E / "robustness_table.csv", index_col=0)
sel = [k for k in ORDER if k in rob.index and k != "smallcnn"]
corrs = [c for c in rob.columns if c != "clean"]
x = np.arange(len(corrs))
w = 0.8 / len(sel)
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, k in enumerate(sel):
    delta = (rob.loc[k, corrs].astype(float) - rob.loc[k, "clean"]).values
    ax.bar(x + i * w, delta, w, label=DISPLAY[k], color=color_of(k))
ax.set_xticks(x + w * (len(sel) - 1) / 2)
ax.set_xticklabels(["blur σ1", "blur σ2", "JPEG q30", "JPEG q60", "bright ×0.6", "×1.4"])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Δ test accuracy vs clean")
ax.set_title("Robustness: accuracy drop under synthetic corruptions (test)")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(FIG / "fig_robustness_delta.png", dpi=150)
plt.close(fig)

# 4) ASF gate mean per component — 3 seeds
fig, ax = plt.subplots(figsize=(8, 4.2))
comps = ["LL2", "LH2", "HL2", "HH2", "LL4", "LH4", "HL4", "HH4"]
xs = np.arange(8)
for e in sorted([e for e in rp["results"] if e["variant"] == "ASF_A2fix_ent"], key=lambda e: e["seed"]):
    g = e["gate_summary"]
    ax.plot(xs, g["gate_mean_per_component"], marker="o", label=f"seed {e['seed']}")
ax.axhline(1 / 8, color="gray", ls="--", lw=0.8, label="đều (1/8)")
ax.set_xticks(xs, comps)
ax.set_ylabel("Gate weight trung bình")
ax.set_title("ASF gate distribution per component — 3 seeds (sau entropy reg)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_gate_mean.png", dpi=150)
plt.close(fig)

# 5) Class distribution + sample images (từ data/raw)
data_root = ROOT / "data" / "raw" / "punfake---resisc45" / "NWPU-RESISC45"
if data_root.is_dir():
    classes = sorted(d for d in data_root.iterdir() if d.is_dir())
    counts = [len(list(c.glob("*.jpg"))) for c in classes]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(len(classes)), counts, color="#4c78a8")
    ax.set_xticks(range(len(classes)), [c.name for c in classes], rotation=90, fontsize=6)
    ax.set_ylabel("Số ảnh")
    ax.set_title("NWPU-RESISC45: 45 lớp × 700 ảnh (đếm thực tế)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_class_dist.png", dpi=150)
    plt.close(fig)

    from PIL import Image
    fig, axes = plt.subplots(5, 9, figsize=(11, 6.5))
    for ax, c in zip(axes.flat, classes):
        img = Image.open(next(c.glob("*.jpg")))
        ax.imshow(img)
        ax.set_title(c.name, fontsize=6)
        ax.axis("off")
    fig.suptitle("Ảnh mẫu mỗi lớp (nguồn: NWPU-RESISC45)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_samples.png", dpi=130)
    plt.close(fig)

print("figures saved to", FIG)
for p in sorted(FIG.glob("*.png")):
    print(" ", p.name)
