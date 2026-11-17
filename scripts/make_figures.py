#!/usr/bin/env python3
"""Sinh hình báo cáo từ deliverables/results_master.json & artifacts đã pull.

Chạy: uv run python3 scripts/make_figures.py
Output: notes/figures/*.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ROOT, load_json, load_output_json
from PIL import Image

FIG = ROOT / "notes" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

master = load_json("deliverables/results_master.json")
MODELS = master["models"]
rp = load_output_json("resisc45-final-proposed-asf-cmp-3-seeds/final_proposed_report.json")
re_ = load_output_json("resisc45-final-evaluation-v2-offline/test_report.json")

ORDER = [
    "smallcnn", "mobilenetv3s", "resnet18", "CMP_C2", "ASF_A2fix_ent",
    "cmp_exp_ce", "cmp_exp_kd", "asf_kd", "resnet18_kd", "efficientnetb0", "convnextt",
]
SHORT = {
    "smallcnn": "B0 SmallCNN",
    "mobilenetv3s": "B2 MobileNetV3-S",
    "resnet18": "B1 ResNet18-CE",
    "CMP_C2": "CMP-CE (v1)",
    "ASF_A2fix_ent": "ASF-CE (v1)",
    "cmp_exp_ce": "CMP-exp-CE (v4)",
    "cmp_exp_kd": "CMP-exp-KD (v4)",
    "asf_kd": "ASF-KD (v3)",
    "resnet18_kd": "ResNet18-KD (v3)",
    "efficientnetb0": "B3 EffNet-B0",
    "convnextt": "B4 ConvNeXt-T",
}
COL = {"B1": "#4c78a8", "proposed": "#e45756", "cmp_exp": "#54a24b", "kd": "#f58518", "other": "#9aa0a6"}


def color_of(key: str) -> str:
    if key == "resnet18":
        return COL["B1"]
    if key in ("cmp_exp_ce", "cmp_exp_kd"):
        return COL["cmp_exp"]
    if key in ("ASF_A2fix_ent", "CMP_C2"):
        return COL["proposed"]
    if key in ("resnet18_kd", "asf_kd"):
        return COL["kd"]
    return COL["other"]


# 1) Val F1 mean ± std — bar (đủ 11 mô hình từ results_master)
fig, ax = plt.subplots(figsize=(11, 4.8))
keys = [k for k in ORDER if k in MODELS]
means = [MODELS[k]["val_f1_3seed"]["mean"] for k in keys]
stds = [MODELS[k]["val_f1_3seed"]["std"] for k in keys]
colors = [color_of(k) for k in keys]
bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=4, color=colors)
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([SHORT[k] for k in keys], rotation=22, ha="right", fontsize=9)
ax.set_ylabel("Val macro-F1 (3 seeds)")
ax.set_title("Validation macro-F1, mean ± std over seeds {0,1,2} (15 epochs)")
ax.set_ylim(0.5, 1.0)
for b, m, s in zip(bars, means, stds, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "fig_val_f1_final.png", dpi=150)
plt.close(fig)

# 2) TEST F1 vs latency — 2 panels: zoom vùng pretrained (0.943–0.972) + toàn cảnh gồm B0
fig, (ax_all, ax) = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.0, 2.2]})
for k in keys:
    e = MODELS[k]
    lat = e["latency_ms_3seed_mean"]
    f1_val = e["test_3seed_f1"]["mean"]
    ax_all.scatter(lat, f1_val, s=65, color=color_of(k), zorder=3)
ax_all.annotate("B0 SmallCNN\n(0.5904, 1.54ms)", (MODELS["smallcnn"]["latency_ms_3seed_mean"], MODELS["smallcnn"]["test_3seed_f1"]["mean"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
ax_all.set_xlabel("Latency batch-1 (ms)")
ax_all.set_ylabel("TEST macro-F1 (3-seed)")
ax_all.set_title("Toàn cảnh (gồm B0)")
ax_all.set_xlim(1.0, 8.2)
ax_all.set_ylim(0.55, 1.0)
ax_all.grid(alpha=0.3)

OFFSETS = {
    "mobilenetv3s": (8, -10),
    "resnet18": (-18, -15),
    "CMP_C2": (8, -12),
    "ASF_A2fix_ent": (8, -14),
    "cmp_exp_ce": (-78, 8),
    "cmp_exp_kd": (-82, 16),
    "asf_kd": (8, 8),
    "resnet18_kd": (8, 6),
    "efficientnetb0": (-72, -14),
    "convnextt": (-80, 4),
}
for k in keys:
    if k == "smallcnn":
        continue
    e = MODELS[k]
    lat = e["latency_ms_3seed_mean"]
    f1_val = e["test_3seed_f1"]["mean"]
    ax.scatter(lat, f1_val, s=110, color=color_of(k), zorder=3)
    dx, dy = OFFSETS.get(k, (7, 4))
    ax.annotate(SHORT[k], (lat, f1_val), textcoords="offset points", xytext=(dx, dy), fontsize=8.2,
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6})
ax.set_xlabel("Latency batch-1 (ms, 3-seed mean trên T4)")
ax.set_ylabel("TEST macro-F1 (3-seed mean)")
ax.set_title("Zoom vùng pretrained (4,725 ảnh test — chú ý khác phiên đo T4)")
ax.set_xlim(2.0, 7.9)
ax.set_ylim(0.943, 0.971)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_test_f1_latency.png", dpi=150)
plt.close(fig)

# 3) Robustness delta — grouped bars (5 model chính từ 070b)
rob = pd.read_csv(ROOT / "outputs" / "resisc45-final-evaluation-v2-offline" / "robustness_table.csv", index_col=0)
sel = [k for k in ("resnet18", "ASF_A2fix_ent", "CMP_C2", "efficientnetb0", "convnextt") if k in rob.index]
corrs = [c for c in rob.columns if c != "clean"]
x = np.arange(len(corrs))
w = 0.8 / len(sel)
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, k in enumerate(sel):
    delta = (rob.loc[k, corrs].astype(float) - rob.loc[k, "clean"]).values
    ax.bar(x + i * w, delta, w, label=SHORT[k], color=color_of(k))
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
comps = ["ORIG2", "LH2", "HL2", "HH2", "ORIG4", "LH4", "HL4", "HH4"]
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

    fig, axes = plt.subplots(5, 9, figsize=(11, 6.5))
    for ax, c in zip(axes.flat, classes, strict=True):
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
