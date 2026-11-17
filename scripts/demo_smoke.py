#!/usr/bin/env python3
"""Demo smoke test headless — kiểm tra logic demo/app.py không cần UI:
load model (cache), predict 1 ảnh train thật, top-3, latency fields, corruption.
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
    "B1-CE s2": ("resnet18", {}, ROOT / "outputs/resisc45-final-baselines-b0-b4-3-seeds/ckpt_resnet18_s2.pt"),
    "ASF-KD s1": ("asf", {"asf_out_init": "small", "asf_gate_ent": 0.05},
                  ROOT / "outputs/resisc45-kd-final-seeds-1-2/ckpt_asf_kd_s1.pt"),
    "CMP-exp-CE s1": ("cmp", {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp"},
                      ROOT / "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/ckpt_cmp_exp_ce_s1.pt"),
    "CMP-exp-KD s0": ("cmp", {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp",
                              "kd_alpha": 0.5, "kd_T": 4.0},
                      ROOT / "outputs/resisc45-h1a-screen-cmp-exp-asf-lite/ckpt_cmp_exp_kd.pt"),
}

sample_img = DATA / "airplane" / "airplane_001.jpg"  # ảnh TRAIN — không dùng test làm gallery
img = Image.open(sample_img).convert("RGB")
x = eval_augment(img, 256).unsqueeze(0)

for name, (mk, ov, ck_path) in MODELS.items():
    assert ck_path.exists(), f"missing {ck_path}"
    cfg = TrainConfig(num_classes=45, model_key=mk, **ov)
    model = build_model(cfg, pretrained=False)
    sd = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits.float(), 1)[0]
    lat_ms = (time.perf_counter() - t0) * 1000
    top3 = torch.topk(probs, 3)
    top3_out = [(CLASSES[int(i)], round(float(p), 4)) for p, i in zip(top3.values, top3.indices, strict=True)]
    CHECKS[name] = {"top3": top3_out, "latency_ms": round(lat_ms, 1),
                    "probs_sum": round(float(probs.sum()), 4)}
    print(f"{name}: top-3 {top3_out} | {lat_ms:.1f}ms | probs sum = {CHECKS[name]['probs_sum']}")

# corruption path (như demo)
img_c = img.filter(ImageFilter.GaussianBlur(1.0))
x2 = eval_augment(img_c, 256).unsqueeze(0)
print("corruption path OK:", tuple(x2.shape))

all_ok = (all(v["probs_sum"] > 0.99 for v in CHECKS.values())
          and all(len(v["top3"]) == 3 for v in CHECKS.values()))
print("DEMO SMOKE", "PASS" if all_ok else "FAIL")
Path(ROOT / "deliverables").mkdir(exist_ok=True)
(ROOT / "deliverables" / "demo_smoke.json").write_text(
    json.dumps({"checks": CHECKS, "pass": all_ok, "sample": str(sample_img)}, indent=1, ensure_ascii=False),
    encoding="utf-8",
)
