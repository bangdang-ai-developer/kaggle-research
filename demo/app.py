#!/usr/bin/env python3
"""Demo phân loại cảnh ảnh viễn thám NWPU-RESISC45 — chạy local:

    cd kaggle-research
    uv run streamlit run demo/app.py

Yêu cầu: checkpoints đã pull về outputs/ (từ kernel 060a/060b) + class_order.json từ 010.
Models được cache sau lần load đầu. Chỉ dùng ảnh mới/upload — không dùng ảnh test làm gallery.
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import TrainConfig, build_model, eval_augment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "B1 ResNet18 (baseline CE)": ("resnet18", {}, "resisc45-final-baselines-b0-b4-3-seeds", "ckpt_resnet18_s2.pt"),
    "B1 ResNet18-KD (distilled v3)": ("resnet18", {"kd_alpha": 0.5, "kd_T": 4.0},
                                      "resisc45-screen-kd-resnet18-asf-cmp-v2", "ckpt_resnet18_kd.pt"),
    "ASF A2fix-ent (proposed v1 CE)": ("asf", {"asf_out_init": "small", "asf_gate_ent": 0.05},
                                       "resisc45-final-proposed-asf-cmp-3-seeds", "ckpt_ASF_A2fix_ent_s1.pt"),
    "ASF-KD (distilled v3, best val)": ("asf", {"asf_out_init": "small", "asf_gate_ent": 0.05},
                                        "resisc45-kd-final-seeds-1-2", "ckpt_asf_kd_s1.pt"),
    "CMP C2 (proposed v1)": ("cmp", {"cmp_n_proto": 2},
                             "resisc45-final-proposed-asf-cmp-3-seeds", "ckpt_CMP_C2_s1.pt"),
    "CMP-exp-CE (proposed v4, best val s1)": (
        "cmp",
        {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp"},
        "resisc45-cmp-exp-3-seeds-and-test-eval",
        "ckpt_cmp_exp_ce_s1.pt",
    ),
    "CMP-exp-KD (proposed v4, best val s0)": (
        "cmp",
        {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp", "kd_alpha": 0.5, "kd_T": 4.0},
        "resisc45-h1a-screen-cmp-exp-asf-lite",
        "ckpt_cmp_exp_kd.pt",
    ),
}
CORRUPTIONS = {"Không": (None, None), "Blur σ1": ("blur", 1.0), "Blur σ2": ("blur", 2.0),
               "JPEG q30": ("jpeg", 30), "JPEG q60": ("jpeg", 60),
               "Sáng ×0.6": ("bright", 0.6), "Tối ×1.4": ("bright", 1.4)}


def _jpeg_corrupt(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTION_STRATEGIES: dict[str, Callable[[Image.Image, Any], Image.Image]] = {
    "blur": lambda img, sigma: img.filter(ImageFilter.GaussianBlur(sigma)),
    "jpeg": _jpeg_corrupt,
    "bright": lambda img, factor: ImageEnhance.Brightness(img).enhance(factor),
}


@st.cache_resource
def load_class_order() -> dict:
    p = ROOT / "outputs" / "resisc45-data-audit-split-manifest" / "class_order.json"
    if not p.exists():
        st.error(f"Thiếu {p} — pull output kernel 010 trước.")
        st.stop()
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(display_key: str):
    mk, ov, out_dir, ck_name = MODELS[display_key]
    ck_path = ROOT / "outputs" / out_dir / ck_name
    if not ck_path.exists():
        st.error(f"Thiếu checkpoint {ck_path} — pull output kernel 060a/060b trước.")
        st.stop()
    cfg = TrainConfig(num_classes=45, model_key=mk, **ov)
    model = build_model(cfg, pretrained=False)  # checkpoint chứa toàn bộ weights — offline
    sd = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd["state_dict"])
    model.to(DEVICE).eval()
    return model


def corrupt(img: Image.Image, kind: str | None, param: Any) -> Image.Image:
    strategy = CORRUPTION_STRATEGIES.get(kind) if kind is not None else None
    return strategy(img, param) if strategy is not None else img


st.set_page_config(page_title="NWPU-RESISC45 Demo", layout="wide")
st.title("Phân loại cảnh ảnh viễn thám — NWPU-RESISC45")
st.caption("Models: ResNet18 baseline và 2 mô hình nhóm (ASF, CMP) — checkpoint final v1, seed best-val. "
           "Chạy local, không gửi dữ liệu đi đâu.")

class_order = load_class_order()
CLASSES = [c for c, i in sorted(class_order.items(), key=lambda kv: kv[1])]

sel_model = st.selectbox("Model", list(MODELS.keys()))
sel_corr = st.selectbox("Corruption (synthetic — minh họa độ bền)", list(CORRUPTIONS.keys()))
uploaded = st.file_uploader("Upload ảnh (JPG/PNG)", type=("jpg", "jpeg", "png"))

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    kind, param = CORRUPTIONS[sel_corr]
    img_c = corrupt(img, kind, param)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Gốc", width=280)
    with col2:
        st.image(img_c, caption=f"Sau corruption: {sel_corr}", width=280)

    if st.button("Phân loại", type="primary"):
        model = load_model(sel_model)
        x = eval_augment(img_c, 256).unsqueeze(0).to(DEVICE)
        # model-only latency (warmup + sync)
        for _ in range(3):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(logits.float(), dim=1)[0].cpu()
        top3 = torch.topk(probs, 3)
        e2e_ms = (time.perf_counter() - t0) * 1000

        st.subheader(f"Top-3 dự đoán — {sel_model}")
        for prob, idx in zip(top3.values, top3.indices, strict=True):
            st.progress(float(prob), text=f"{CLASSES[int(idx)]} — {float(prob)*100:.2f}%")
        st.info(f"Model-only latency: **{model_ms:.1f} ms** | end-to-end (gồm upload/preprocess/postprocess): "
                f"**{e2e_ms:.1f} ms** | device: {DEVICE}")
        st.caption("Lưu ý: softmax score là xác suất dự đoán, chưa hiệu chuẩn (không phải confidence đã calibration). "
                   "Corruption là synthetic — minh họa, không chứng minh độ bền deployment.")
else:
    st.info("Upload một ảnh để bắt đầu.")

st.divider()
st.caption("Checkpoint: outputs/ (kernel Kaggle 060a/060b) · Split manifest sha256 cc70020a… · "
           "Demo mục 12/14 spec — không triển khai public.")
