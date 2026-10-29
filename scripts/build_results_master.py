#!/usr/bin/env python3
"""Build bảng số liệu chuẩn (master) từ artifacts — nguồn duy nhất cho báo cáo/slides/hình.

Output: deliverables/results_master.json + results_master.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)

def load(p: str) -> dict:
    return json.load(open(ROOT / "outputs" / p))

r060a = load("resisc45-final-baselines-b0-b4-3-seeds/final_baselines_report.json")
r060b = load("resisc45-final-proposed-asf-cmp-3-seeds/final_proposed_report.json")
r070b = load("resisc45-final-evaluation-v2-offline/test_report.json")
r071b = load("outputs_resisc45_tmp.json") if False else load("resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json")
r112c = load("resisc45-eval-kd-3-seeds-v2-fixed-sources/test_kd_3seed_report.json")
r110b = load("resisc45-screen-kd-resnet18-asf-cmp-v2/kd_screen_report.json")
r111 = load("resisc45-kd-final-seeds-1-2/kd_final_report.json")

def ms(vals):
    v = np.array(vals, dtype=float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}

DISPLAY = {
    "smallcnn": "B0 SmallCNN (from scratch)",
    "resnet18": "B1 ResNet18-CE",
    "mobilenetv3s": "B2 MobileNetV3-S",
    "efficientnetb0": "B3 EfficientNet-B0",
    "convnextt": "B4 ConvNeXt-T (teacher)",
    "ASF_A2fix_ent": "ASF-CE (proposed)",
    "CMP_C2": "CMP-CE (proposed)",
    "resnet18_kd": "ResNet18-KD",
    "asf_kd": "ASF-KD (proposed)",
}
master: dict = {"models": {}}

# ---- CE models: val 3-seed (060a/b), test 3-seed + precision/recall + latency (071b), best-val test (070b)
ce_src = {}
for e in r060a["results"]:
    ce_src.setdefault(e["model"], {})[e["seed"]] = e
for e in r060b["results"]:
    ce_src.setdefault(e["variant"], {})[e["seed"]] = e
test_ce = {}
for e in r071b["results"]:
    test_ce.setdefault(e["model"], {})[e["seed"]] = e
best_test = {k: v for k, v in r070b["test_results"].items()}

for key in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt", "ASF_A2fix_ent", "CMP_C2"):
    rows = ce_src[key]
    seeds = sorted(rows)
    t3 = test_ce[key]
    t3_seeds = sorted(t3)
    bv = max(rows.values(), key=lambda e: e["best_val_macro_f1"])
    bt = best_test[key]
    entry = {
        "display": DISPLAY[key],
        "params_total_m": round(rows[seeds[0]]["params"]["total"] / 1e6, 2),
        "val_f1_3seed": ms([rows[s]["best_val_macro_f1"] for s in seeds]),
        "val_acc_3seed": ms([rows[s]["best_val_acc"] for s in seeds]),
        "train_time_h_mean": round(float(np.mean([rows[s]["train_time_sec"] for s in seeds])) / 3600, 3),
        "executed_epochs": {s: rows[s]["executed_epochs"] for s in seeds},
        "test_3seed_acc": ms([t3[s]["test_acc"] for s in t3_seeds]),
        "test_3seed_f1": ms([t3[s]["test_macro_f1"] for s in t3_seeds]),
        "test_3seed_precision": ms([t3[s]["test_macro_precision"] for s in t3_seeds]),
        "test_3seed_recall": ms([t3[s]["test_macro_recall"] for s in t3_seeds]),
        "test_bestvalseed": {"seed": bt["seed"], "acc": bt["acc"], "f1": bt["macro_f1"],
                             "precision": bt["macro_precision"], "recall": bt["macro_recall"]},
        "latency_ms_bestvalseed": r070b["latency"][key]["median_ms"],
        "is_teacher": key == "convnextt",
    }
    master["models"][key] = entry

# ---- KD models: val 3-seed (110b s0 + 111 s1/s2), test 3-seed (112c), precision/recall từ 112c
kd_src = {"resnet18_kd": {}, "asf_kd": {}}
for e in r110b["results"]:
    if e["student"] in kd_src:
        kd_src[e["student"]][e["config"]["seed"]] = e
for e in r111["results"]:
    kd_src[e["student"]][e["seed"]] = e
test_kd = {}
for e in r112c["results"]:
    test_kd.setdefault(e["student"], {})[e["seed"]] = e

for key in ("resnet18_kd", "asf_kd"):
    rows = kd_src[key]
    seeds = sorted(rows)
    t3 = test_kd[key]
    t3_seeds = sorted(t3)
    bv = max(rows.values(), key=lambda e: e["best_val_macro_f1"])
    entry = {
        "display": DISPLAY[key],
        "params_total_m": round(rows[seeds[0]]["params"]["total"] / 1e6, 2),
        "val_f1_3seed": ms([rows[s]["best_val_macro_f1"] for s in seeds]),
        "val_acc_3seed": ms([rows[s]["best_val_acc"] for s in seeds]),
        "train_time_h_mean": round(float(np.mean([rows[s]["train_time_sec"] for s in seeds])) / 3600, 3),
        "executed_epochs": {s: rows[s]["executed_epochs"] for s in seeds},
        "test_3seed_acc": ms([t3[s]["test_acc"] for s in t3_seeds]),
        "test_3seed_f1": ms([t3[s]["test_macro_f1"] for s in t3_seeds]),
        "test_3seed_precision": ms([t3[s]["test_macro_precision"] for s in t3_seeds]),
        "test_3seed_recall": ms([t3[s]["test_macro_recall"] for s in t3_seeds]),
        "latency_ms_3seed_mean": round(float(np.mean([t3[s]["latency"]["median_ms"] for s in t3_seeds])), 2),
        "kd": {"alpha": 0.5, "T": 4.0, "teacher": "ConvNeXt-T s2 (val F1 0.9691, sha 9627ca43)"},
    }
    # best-val seed test metrics từ 112c
    bvs = max(t3_seeds, key=lambda s: rows[s]["best_val_macro_f1"])
    entry["test_bestvalseed"] = {"seed": bvs, "acc": t3[bvs]["test_acc"], "f1": t3[bvs]["test_macro_f1"],
                                 "precision": t3[bvs]["test_macro_precision"], "recall": t3[bvs]["test_macro_recall"]}
    entry["latency_ms_bestvalseed"] = t3[bvs]["latency"]["median_ms"]
    master["models"][key] = entry

# ---- KD overhead + teacher history ----
master["kd_overhead"] = {
    "teacher_inference_overhead": "KD epoch ~100-104s vs CE ~49-53s (teacher forward online)",
    "teacher_training_cost_history": {"model": "B4 ConvNeXt-T", "train_time_h_per_run": master["models"]["convnextt"]["train_time_h_mean"],
                                      "note": "đã chi trong v1, tái sử dụng làm teacher — không train lại"},
}
master["audit"] = {
    "fairness": "LOẠI A — 12 trường protocol GIỐNG hệt giữa CE/KD; khác biệt duy nhất kd_alpha/kd_T (biến điều tra). notes/v4_audit.json",
    "seed0_merge": "HỢP LỆ — config s0 vs s1/s2 chỉ khác epoch_saved; teacher hash 9627ca43 xác nhận logs cả 2 kernel; code nhúng notebook 110b đúng (không nhiễm bug _gate_input)",
}
master["protocol"] = {
    "split_sha256": "cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a",
    "epochs_max": 15, "early_stop_patience": 5, "batch": 128, "optimizer": "AdamW lr 1e-4/1e-3 wd 1e-4 cosine warmup 5%",
    "checkpoint_rule": "val macro-F1", "seeds": [0, 1, 2], "amp": True, "input": "256x256 ImageNet-norm RRC+hflip+vflip",
    "test_caveat": "test đã quan sát qua v1/v2 — v3 là nghiên cứu mở rộng, không chọn cấu hình bằng test",
}

(OUT / "results_master.json").write_text(json.dumps(master, indent=1, ensure_ascii=False))

# CSV phẳng
rows = []
for key, e in master["models"].items():
    rows.append({
        "model": key, "display": e["display"], "params_M": e["params_total_m"],
        "val_f1_mean": e["val_f1_3seed"]["mean"], "val_f1_std": e["val_f1_3seed"]["std"],
        "test_acc_mean": e["test_3seed_acc"]["mean"], "test_acc_std": e["test_3seed_acc"]["std"],
        "test_f1_mean": e["test_3seed_f1"]["mean"], "test_f1_std": e["test_3seed_f1"]["std"],
        "test_precision_mean": e["test_3seed_precision"]["mean"],
        "test_recall_mean": e["test_3seed_recall"]["mean"],
        "test_f1_bestvalseed": e["test_bestvalseed"]["f1"],
        "latency_ms": e.get("latency_ms_bestvalseed") or e.get("latency_ms_3seed_mean"),
        "train_h": e["train_time_h_mean"], "teacher": e.get("teacher") is not None,
    })
pd.DataFrame(rows).to_csv(OUT / "results_master.csv", index=False)
print(f"saved {OUT / 'results_master.json'} + results_master.csv")
for r in rows:
    print(f"  {r['display']:<28} testF1 {r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} acc {r['test_acc_mean']:.4f} lat {r['latency_ms']}ms")
