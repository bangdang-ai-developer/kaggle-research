#!/usr/bin/env python3
"""Sơ đồ kiến trúc ASF / CMP / KD — vẽ vector matplotlib (không dùng ảnh AI).

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
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, label=None, fs=7, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=EDGE, lw=1.0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label, ha="center", fontsize=fs, color="#333333")


# ================= ASF =================
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.44, 0.13, 0.12, "Ảnh input\n256×256×3", C_OUT)
box(ax, 0.20, 0.44, 0.15, 0.12, "ResNet18 stem\n+layer1 (pretrained)", C_BACK)
box(ax, 0.40, 0.62, 0.15, 0.11, "layer2\n128ch, 32×32", C_BACK)
box(ax, 0.40, 0.27, 0.15, 0.11, "layer3 → layer4\n512ch, 16×16", C_BACK)
# Haar
box(ax, 0.60, 0.62, 0.13, 0.11, "Haar DWT\nLL·LH·HL·HH", C_MODULE)
box(ax, 0.60, 0.27, 0.13, 0.11, "Haar DWT\nLL·LH·HL·HH", C_MODULE)
arrow(ax, 0.55, 0.675, 0.60, 0.675)
arrow(ax, 0.55, 0.325, 0.60, 0.325)
# proj + GAP 8 thành phần
box(ax, 0.60, 0.455, 0.13, 0.09, "8× [1×1 conv → BN\n→ ReLU → GAP]", C_MODULE)
arrow(ax, 0.665, 0.62, 0.665, 0.545)
arrow(ax, 0.665, 0.38, 0.665, 0.455)
# gate
box(ax, 0.60, 0.06, 0.13, 0.10, "Gate MLP\nsoftmax → w₈ theo ảnh", C_GATE)
# fusion
box(ax, 0.78, 0.455, 0.15, 0.09, "Fused = Σ wᵢ·fᵢ\n→ Linear (small-init)", C_MODULE)
arrow(ax, 0.73, 0.50, 0.78, 0.50)
arrow(ax, 0.665, 0.16, 0.85, 0.455)
# backbone path
arrow(ax, 0.35, 0.50, 0.40, 0.325)
arrow(ax, 0.35, 0.50, 0.40, 0.675)
box(ax, 0.78, 0.27, 0.15, 0.10, "GAP(layer4)\n512-d backbone", C_BACK)
arrow(ax, 0.55, 0.325, 0.78, 0.32)
# head
box(ax, 0.62, 0.75, 0.24, 0.11, "Head: backbone + residual\n→ Linear → 45 lớp", C_OUT)
arrow(ax, 0.855, 0.37, 0.80, 0.75)
arrow(ax, 0.855, 0.545, 0.80, 0.75)
ax.text(0.5, 0.955, "ASF-ResNet18 (A2fix-ent): wavelet đa tỉ lệ + gate theo ảnh + entropy reg 0.05",
        ha="center", fontsize=10, weight="bold")
ax.text(0.5, 0.012, "Nhánh backbone pretrained giữ nguyên; nhánh wavelet khởi tạo small-init; "
                    "LL ≡ feature map gốc trước tách; entropy reg chống gate sụp 1-hot",
        ha="center", fontsize=7.5, color="#555555")
fig.tight_layout()
fig.savefig(FIG / "fig_asf_arch.png", dpi=200)
plt.close(fig)

# ================= CMP =================
fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.42, 0.13, 0.12, "Ảnh input\n256×256×3", C_OUT)
box(ax, 0.20, 0.42, 0.15, 0.12, "ResNet18\n(pretrained) → GAP", C_BACK)
box(ax, 0.40, 0.42, 0.14, 0.12, "Embedding Linear\n→ L2 normalize", C_MODULE)
box(ax, 0.62, 0.60, 0.16, 0.13, "Prototypes\nK=2/lớp × 45 lớp\nL2 normalize (k-means init)", C_MODULE)
box(ax, 0.62, 0.24, 0.16, 0.12, "sim = e·pᵏ\nlogit = κ·maxₖ (hard-max)\nκ học được, clamp", C_GATE)
arrow(ax, 0.35, 0.48, 0.40, 0.48)
arrow(ax, 0.54, 0.48, 0.62, 0.655)
arrow(ax, 0.54, 0.48, 0.62, 0.30)
arrow(ax, 0.70, 0.60, 0.70, 0.36)
box(ax, 0.40, 0.06, 0.16, 0.11, "Confusion graph W\ntrain-only probe (90/10)\n8 cặp đối xứng", C_MODULE)
arrow(ax, 0.48, 0.17, 0.665, 0.24)
box(ax, 0.84, 0.42, 0.14, 0.12, "CE + pair loss\nsoftplus(m+s_c−s_y)\n(v2; KD-v3 bỏ pair)", C_OUT)
arrow(ax, 0.78, 0.30, 0.84, 0.45)
ax.text(0.5, 0.95, "CMP-ResNet18 (C2/CMP-P): multi-prototype cosine classifier",
        ha="center", fontsize=10, weight="bold")
ax.text(0.5, 0.01, "v1 LSE → prototype collapse (inter-cos 1.0); v2 hard-max phá collapse nhưng F1 không tăng; "
                    "KD không hội tụ tốt trên hard-max → CMP dừng ở C2",
        ha="center", fontsize=7.5, color="#555555")
fig.tight_layout()
fig.savefig(FIG / "fig_cmp_arch.png", dpi=200)
plt.close(fig)

# ================= KD =================
fig, ax = plt.subplots(figsize=(10.5, 5.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.40, 0.13, 0.14, "Ảnh train\n256×256\n(cùng view)", C_OUT)
box(ax, 0.21, 0.58, 0.17, 0.13, "Teacher: ConvNeXt-T\nFROZEN eval, no grad\n(val F1 0.9691)\nchỉ dùng lúc train", "#eee6f7")
box(ax, 0.21, 0.22, 0.17, 0.13, "Student:\nResNet18 / ASF / CMP\n(ImageNet init)", C_BACK)
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

print("saved:", [p.name for p in sorted(FIG.glob("fig_*arch*.png"))] + ["fig_kd_arch.png"])
