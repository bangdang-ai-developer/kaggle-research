#!/usr/bin/env python3
"""Demo smoke test headless — kiểm tra toàn bộ 7 lựa chọn model của demo/app.py.

Kiểm tra: load checkpoint thật, warm-up + model-only latency, preprocess→forward→postprocess (app e2e),
top-3 dự đoán trên ảnh train thật (không dùng tập test), và đường dẫn corruption.
Output: stdout + deliverables/demo_smoke.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import TrainConfig, build_model, eval_augment

DEVICE = "cpu"  # smoke CPU
CHECKS: dict = {}
DATA = ROOT / "data" / "raw" / "punfake---resisc45" / "NWPU-RESISC45"

CO = json.loads((ROOT / "outputs" / "resisc45-data-audit-split-manifest" / "class_order.json").read_text())
CLASSES = [c for c, _i in sorted(CO.items(), key=lambda kv: kv[1])]
print(f"class_order: {len(CLASSES)} lớp")

MODELS = {
    "B1 ResNet18-CE s2": (
        "resnet18", {},
        ROOT / "outputs/resisc45-final-baselines-b0-b4-3-seeds/ckpt_resnet18_s2.pt",
    ),
    "B1 ResNet18-KD s0": (
        "resnet18", {"kd_alpha": 0.5, "kd_T": 4.0},
        ROOT / "outputs/resisc45-screen-kd-resnet18-asf-cmp-v2/ckpt_resnet18_kd.pt",
    ),
    "ASF-CE s1": (
        "asf", {"asf_out_init": "small", "asf_gate_ent": 0.05},
        ROOT / "outputs/resisc45-final-proposed-asf-cmp-3-seeds/ckpt_ASF_A2fix_ent_s1.pt",
    ),
    "ASF-KD s1": (
        "asf", {"asf_out_init": "small", "asf_gate_ent": 0.05},
        ROOT / "outputs/resisc45-kd-final-seeds-1-2/ckpt_asf_kd_s1.pt",
    ),
    "CMP-CE v1 s1": (
        "cmp", {"cmp_n_proto": 2},
        ROOT / "outputs/resisc45-final-proposed-asf-cmp-3-seeds/ckpt_CMP_C2_s1.pt",
    ),
    "CMP-exp-CE s1": (
        "cmp", {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp"},
        ROOT / "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/ckpt_cmp_exp_ce_s1.pt",
    ),
    "CMP-exp-KD s0": (
        "cmp", {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp", "kd_alpha": 0.5, "kd_T": 4.0},
        ROOT / "outputs/resisc45-h1a-screen-cmp-exp-asf-lite/ckpt_cmp_exp_kd.pt",
    ),
}

sample_img = DATA / "airplane" / "airplane_001.jpg"  # ảnh TRAIN — không dùng test làm gallery
img = Image.open(sample_img).convert("RGB")

for name, (mk, ov, ck_path) in MODELS.items():
    assert ck_path.exists(), f"missing {ck_path}"
    cfg = TrainConfig(num_classes=45, model_key=mk, **ov)
    model = build_model(cfg, pretrained=False)
    sd = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["state_dict"], strict=True)
    model.eval()

    # Preprocess
    t_pre0 = time.perf_counter()
    x = eval_augment(img, 256).unsqueeze(0)
    pre_ms = (time.perf_counter() - t_pre0) * 1000

    # Warm-up
    with torch.inference_mode():
        for _ in range(3):
            model(x)

    # Model-only forward pass
    t_fwd0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
    model_ms = (time.perf_counter() - t_fwd0) * 1000

    # Postprocess
    t_post0 = time.perf_counter()
    probs = torch.softmax(logits.float(), 1)[0]
    top3 = torch.topk(probs, 3)
    post_ms = (time.perf_counter() - t_post0) * 1000
    app_e2e_ms = pre_ms + model_ms + post_ms

    top3_out = [(CLASSES[int(i)], round(float(p), 4)) for p, i in zip(top3.values, top3.indices, strict=True)]
    CHECKS[name] = {
        "top3": top3_out,
        "model_only_ms": round(model_ms, 1),
        "app_e2e_ms": round(app_e2e_ms, 1),
        "probs_sum": round(float(probs.sum()), 4),
    }
    print(f"{name:<20}: top-3 {top3_out} | model={model_ms:.1f}ms | app_e2e={app_e2e_ms:.1f}ms")

# corruption path (như demo)
img_c = img.filter(ImageFilter.GaussianBlur(1.0))
x2 = eval_augment(img_c, 256).unsqueeze(0)
print("corruption path OK:", tuple(x2.shape))

all_ok = (
    len(CHECKS) == len(MODELS)
    and all(v["probs_sum"] > 0.99 for v in CHECKS.values())
    and all(len(v["top3"]) == 3 for v in CHECKS.values())
    and all(v["top3"][0][0] == "airplane" for v in CHECKS.values())
)
print("DEMO SMOKE", "PASS" if all_ok else "FAIL")
Path(ROOT / "deliverables").mkdir(exist_ok=True)
(ROOT / "deliverables" / "demo_smoke.json").write_text(
    json.dumps(
        {
            "checks": CHECKS,
            "n_models_checked": len(CHECKS),
            "device": DEVICE,
            "timing_scope": (
                "model_only_ms: forward pass after load + 3 warm-up runs; "
                "app_e2e_ms: preprocess+forward+postprocess (excludes file transfer)"
            ),
            "pass": all_ok,
            "sample": str(sample_img.relative_to(ROOT)),
        },
        indent=1,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
