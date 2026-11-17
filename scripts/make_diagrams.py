#!/usr/bin/env python3
"""Sơ đồ kiến trúc ASF / CMP-exp / KD — vẽ vector matplotlib (không dùng ảnh AI).

Output: notes/figures/fig_asf_arch.png, fig_cmp_arch.png, fig_kd_arch.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "notes" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_BACK = "#dbe9f6"   # backbone
C_MODULE = "#fde2d4"  # module nhóm
C_GATE = "#d9f0d3"   # gate/adaptive
C_OUT = "#f4f4f4"
EDGE = "#444444"


def box(ax, x, y, w, h, text, color, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                fc=color, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, label=None, fs=7, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=EDGE, lw=1.0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label, ha="center", fontsize=fs, color="#333333")


# ================= ASF =================
fig, ax = plt.subplots(figsize=(11.2, 6.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
box(ax, 0.02, 0.44, 0.11, 0.13, "Ảnh input\n256×256×3", C_OUT, fs=8.2)
box(ax, 0.17, 0.44, 0.14, 0.13, "ResNet18 stem\n+layer1 (pretrained)", C_BACK, fs=8.2)
box(ax, 0.35, 0.65, 0.14, 0.12, "layer2\n128ch, 32×32", C_BACK, fs=8.2)
box(ax, 0.35, 0.25, 0.14, 0.12, "layer3 → layer4\n512ch, 8×8", C_BACK, fs=8.2)
box(ax, 0.53, 0.65, 0.14, 0.12, "ORIG + Haar DWT\nLH·HL·HH (16×16)", C_MODULE, fs=8.0)
box(ax, 0.53, 0.36, 0.14, 0.12, "ORIG + Haar DWT\nLH·HL·HH (4×4)", C_MODULE, fs=8.0)
box(ax, 0.53, 0.505, 0.14, 0.10, "8× [1×1 conv → BN\n→ ReLU → GAP]", C_MODULE, fs=8.0)
box(ax, 0.53, 0.08, 0.14, 0.10, "Gate Linear(8F→8)\nsoftmax → w₈/ảnh", C_GATE, fs=8.0)
box(ax, 0.72, 0.505, 0.125, 0.11, "Fused = Σ wᵢ·fᵢ\n→ Linear (std 1e-3)", C_MODULE, fs=8.0)
box(ax, 0.72, 0.19, 0.125, 0.12, "GAP(layer4)\n512-d backbone", C_BACK, fs=8.2)
box(ax, 0.875, 0.37, 0.105, 0.14, "backbone + res\n→ Linear\n→ 45 lớp", C_OUT, fs=8.0)

arrow(ax, 0.135, 0.505, 0.165, 0.505)
arrow(ax, 0.315, 0.505, 0.345, 0.71)
arrow(ax, 0.315, 0.505, 0.345, 0.31)
arrow(ax, 0.495, 0.71, 0.525, 0.71)
arrow(ax, 0.495, 0.31, 0.525, 0.42)
arrow(ax, 0.495, 0.25, 0.715, 0.25)
arrow(ax, 0.60, 0.645, 0.60, 0.61)
arrow(ax, 0.60, 0.485, 0.60, 0.50)
arrow(ax, 0.675, 0.555, 0.715, 0.555)
arrow(ax, 0.675, 0.15, 0.72, 0.50)
arrow(ax, 0.85, 0.55, 0.875, 0.46)
arrow(ax, 0.85, 0.26, 0.875, 0.41)

ax.text(0.5, 0.94, "ASF-ResNet18 (A2fix-ent): wavelet đa tỉ lệ + gate theo ảnh + entropy reg λ=0.05",
        ha="center", fontsize=10, weight="bold")
ax.text(0.5, 0.015, "Backbone pretrained giữ nguyên; nhánh phụ khởi tạo small-init (std=1e-3); "
                    "ORIG ≡ feature map gốc trước tách; entropy reg chống sụp 1-hot",
        ha="center", fontsize=7.6, color="#555555")
fig.tight_layout()
fig.savefig(FIG / "fig_asf_arch.png", dpi=200)
plt.close(fig)

# ================= CMP-exp (V4) =================
fig, ax = plt.subplots(figsize=(11.0, 5.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
box(ax, 0.02, 0.43, 0.11, 0.14, "Ảnh input\n256×256×3", C_OUT, fs=8.3)
box(ax, 0.17, 0.43, 0.14, 0.14, "ResNet18\n(pretrained) → GAP\n(512-d)", C_BACK, fs=8.3)
box(ax, 0.35, 0.43, 0.15, 0.14, "Embedding Linear\n512→512\n→ L2 normalize (e)", C_MODULE, fs=8.3)
box(ax, 0.55, 0.64, 0.20, 0.16, "Prototypes K=2 × 45 lớp\nL2 normalize (p_{c,k})\nk-means init trên TRAIN", C_MODULE, fs=8.2)
box(ax, 0.55, 0.23, 0.20, 0.15, "Cosine sim: s = e · p_{c,k}\nHard-max theo K=2\nmax_k(s_{c,k})", C_GATE, fs=8.2)
box(ax, 0.80, 0.43, 0.17, 0.15, "Scale mũ κ·max_k(s)\nκ = exp(θ) ∈ [10, 64]\n→ CE (V4) / + KD (V4-KD)", C_OUT, fs=8.2)

arrow(ax, 0.135, 0.50, 0.165, 0.50)
arrow(ax, 0.315, 0.50, 0.345, 0.50)
arrow(ax, 0.505, 0.48, 0.545, 0.31)
arrow(ax, 0.65, 0.635, 0.65, 0.39)
arrow(ax, 0.755, 0.31, 0.795, 0.49)

ax.text(0.5, 0.94, "CMP-exp-ResNet18 (V4): k-means init + cosine similarity + hard-max + κ = exp(θ)",
        ha="center", fontsize=10, weight="bold")
ax.text(0.5, 0.05, "Cấu hình ablation cũ (không dùng trong V4 chính): v1 LSE-logits (collapse), "
                   "C3/CMP-PG dùng confusion graph (train-only 90/10), margin và pair loss.",
        ha="center", fontsize=7.8, color="#555555")
fig.tight_layout()
fig.savefig(FIG / "fig_cmp_arch.png", dpi=200)
plt.close(fig)

# ================= KD =================
fig, ax = plt.subplots(figsize=(10.5, 5.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
box(ax, 0.02, 0.40, 0.13, 0.14, "Ảnh train\n256×256\n(cùng view)", C_OUT)
box(ax, 0.21, 0.58, 0.17, 0.13, "Teacher: ConvNeXt-T\nFROZEN eval, no grad\n(val F1 0.9691)\nchỉ dùng lúc train", "#eee6f7")
box(ax, 0.21, 0.22, 0.17, 0.13, "Student:\nResNet18 / ASF / CMP-exp\n(ImageNet init)", C_BACK)
box(ax, 0.45, 0.58, 0.16, 0.13, "Soft targets\nsoftmax(z_t / T=4)", C_MODULE)
box(ax, 0.45, 0.22, 0.16, 0.13, "Logits student\n+ CE nhãn thật", C_MODULE)
box(ax, 0.68, 0.40, 0.18, 0.13, "L = (1−α)·CE\n+ α·T²·KL(t‖s)\nα=0.5, T=4", C_GATE)
arrow(ax, 0.15, 0.47, 0.21, 0.645)
arrow(ax, 0.15, 0.47, 0.21, 0.285)
arrow(ax, 0.38, 0.645, 0.45, 0.645)
arrow(ax, 0.38, 0.285, 0.45, 0.285)
arrow(ax, 0.61, 0.645, 0.68, 0.475)
arrow(ax, 0.61, 0.285, 0.68, 0.45)
box(ax, 0.68, 0.10, 0.18, 0.11, "Chỉ student học\n→ inference KHÔNG cần teacher", C_OUT)
arrow(ax, 0.77, 0.40, 0.77, 0.21)
ax.text(0.5, 0.95, "Knowledge Distillation (Hinton et al.): teacher frozen, same-view, KD loss T²·KL",
        ha="center", fontsize=10, weight="bold")
ax.text(0.5, 0.01, "Teacher: ConvNeXt-T s2 best-val (0.9691) — chi phí teacher chỉ ở training; "
                    "student inference độc lập, latency không đổi",
        ha="center", fontsize=7.5, color="#555555")
fig.tight_layout()
fig.savefig(FIG / "fig_kd_arch.png", dpi=200)
plt.close(fig)

print("saved:", [p.name for p in sorted(FIG.glob("fig_*arch*.png"))])
