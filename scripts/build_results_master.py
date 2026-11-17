#!/usr/bin/env python3
"""Build bảng số liệu chuẩn (master) từ artifacts — nguồn duy nhất cho báo cáo/slides/hình.

Output: deliverables/results_master.json + results_master.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from common import ROOT, load_output_json, mean_std_dict

OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)

r010 = load_output_json("resisc45-data-audit-split-manifest/audit_report.json")
r060a = load_output_json("resisc45-final-baselines-b0-b4-3-seeds/final_baselines_report.json")
r060b = load_output_json("resisc45-final-proposed-asf-cmp-3-seeds/final_proposed_report.json")
r070b = load_output_json("resisc45-final-evaluation-v2-offline/test_report.json")
r071b = load_output_json("resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json")
r112c = load_output_json("resisc45-eval-kd-3-seeds-v2-fixed-sources/test_kd_3seed_report.json")
r110b = load_output_json("resisc45-screen-kd-resnet18-asf-cmp-v2/kd_screen_report.json")
r111 = load_output_json("resisc45-kd-final-seeds-1-2/kd_final_report.json")
r120 = load_output_json("resisc45-h1a-screen-cmp-exp-asf-lite/h1a_screen_report.json")
r121_val = load_output_json("resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_final_report.json")
r121_test = load_output_json("resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_test_3seed_report.json")

DISPLAY = {
    "smallcnn": "B0 SmallCNN",
    "resnet18": "B1 ResNet18-CE",
    "mobilenetv3s": "B2 MobileNetV3-S",
    "efficientnetb0": "B3 EfficientNet-B0",
    "convnextt": "B4 ConvNeXt-T",
    "ASF_A2fix_ent": "ASF-CE",
    "CMP_C2": "CMP-CE (v1)",
    "cmp_exp_ce": "CMP-exp-CE (v4)",
    "resnet18_kd": "ResNet18-KD",
    "asf_kd": "ASF-KD",
    "cmp_exp_kd": "CMP-exp-KD (v4)",
}

# Thời gian k-means init (s) ghi từ logs kernel 120 & 121 (tách riêng khỏi train_model)
KMEANS_SEC = {
    "cmp_exp_ce": {0: 44.8, 1: 48.1, 2: 41.2},
    "cmp_exp_kd": {0: 42.9, 1: 41.4, 2: 41.6},
}

master: dict = {"models": {}}

# ---- CE models: val 3-seed (060a/b), test 3-seed + P/R + 3-seed latency (071b), best-val test/lat (070b)
ce_src: dict = {}
for e in r060a["results"]:
    ce_src.setdefault(e["model"], {})[e["seed"]] = e
for e in r060b["results"]:
    ce_src.setdefault(e["variant"], {})[e["seed"]] = e
test_ce: dict = {}
for e in r071b["results"]:
    test_ce.setdefault(e["model"], {})[e["seed"]] = e
best_test = dict(r070b["test_results"])

for key in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt", "ASF_A2fix_ent", "CMP_C2"):
    rows = ce_src[key]
    seeds = sorted(rows)
    t3 = test_ce[key]
    t3_seeds = sorted(t3)
    bt = best_test[key]
    train_h = round(float(np.mean([rows[s]["train_time_sec"] for s in seeds])) / 3600, 3)
    master["models"][key] = {
        "display": DISPLAY[key],
        "group": "proposed_ce" if key in ("ASF_A2fix_ent", "CMP_C2") else "baseline",
        "params_total_m": round(rows[seeds[0]]["params"]["total"] / 1e6, 2),
        "val_f1_3seed": mean_std_dict([rows[s]["best_val_macro_f1"] for s in seeds]),
        "val_acc_3seed": mean_std_dict([rows[s]["best_val_acc"] for s in seeds]),
        "train_time_h_mean": train_h,
        "kmeans_init_h_mean": 0.0,
        "total_train_h_mean": train_h,
        "executed_epochs": {s: rows[s]["executed_epochs"] for s in seeds},
        "test_3seed_acc": mean_std_dict([t3[s]["test_acc"] for s in t3_seeds]),
        "test_3seed_f1": mean_std_dict([t3[s]["test_macro_f1"] for s in t3_seeds]),
        "test_3seed_precision": mean_std_dict([t3[s]["test_macro_precision"] for s in t3_seeds]),
        "test_3seed_recall": mean_std_dict([t3[s]["test_macro_recall"] for s in t3_seeds]),
        "test_bestvalseed": {
            "seed": bt["seed"],
            "acc": bt["acc"],
            "f1": bt["macro_f1"],
            "precision": bt["macro_precision"],
            "recall": bt["macro_recall"],
        },
        "latency_ms_3seed_mean": round(float(np.mean([t3[s]["latency"]["median_ms"] for s in t3_seeds])), 2),
        "latency_ms_bestvalseed": round(float(r070b["latency"][key]["median_ms"]), 2),
        "latency_source": "071b (3-seed mean of medians, T4 cuda, batch=1, n_warmup=10, n_iter=50)",
        "is_teacher": key == "convnextt",
    }

# ---- KD models (v3): val 3-seed (110b s0 + 111 s1/s2), test 3-seed (112c)
kd_src: dict = {"resnet18_kd": {}, "asf_kd": {}}
for e in r110b["results"]:
    if e["student"] in kd_src:
        kd_src[e["student"]][e["config"]["seed"]] = e
for e in r111["results"]:
    kd_src[e["student"]][e["seed"]] = e
test_kd: dict = {}
for e in r112c["results"]:
    test_kd.setdefault(e["student"], {})[e["seed"]] = e

for key in ("resnet18_kd", "asf_kd"):
    rows = kd_src[key]
    seeds = sorted(rows)
    t3 = test_kd[key]
    t3_seeds = sorted(t3)
    bvs = max(t3_seeds, key=lambda s: rows[s]["best_val_macro_f1"])
    train_h = round(float(np.mean([rows[s]["train_time_sec"] for s in seeds])) / 3600, 3)
    master["models"][key] = {
        "display": DISPLAY[key],
        "group": "kd_control" if key == "resnet18_kd" else "kd_extension",
        "params_total_m": round(rows[seeds[0]]["params"]["total"] / 1e6, 2),
        "val_f1_3seed": mean_std_dict([rows[s]["best_val_macro_f1"] for s in seeds]),
        "val_acc_3seed": mean_std_dict([rows[s]["best_val_acc"] for s in seeds]),
        "train_time_h_mean": train_h,
        "kmeans_init_h_mean": 0.0,
        "total_train_h_mean": train_h,
        "executed_epochs": {s: rows[s]["executed_epochs"] for s in seeds},
        "test_3seed_acc": mean_std_dict([t3[s]["test_acc"] for s in t3_seeds]),
        "test_3seed_f1": mean_std_dict([t3[s]["test_macro_f1"] for s in t3_seeds]),
        "test_3seed_precision": mean_std_dict([t3[s]["test_macro_precision"] for s in t3_seeds]),
        "test_3seed_recall": mean_std_dict([t3[s]["test_macro_recall"] for s in t3_seeds]),
        "latency_ms_3seed_mean": round(float(np.mean([t3[s]["latency"]["median_ms"] for s in t3_seeds])), 2),
        "latency_ms_bestvalseed": round(float(t3[bvs]["latency"]["median_ms"]), 2),
        "latency_source": "112c (3-seed mean of medians, T4 cuda, batch=1, n_warmup=10, n_iter=50)",
        "kd": {"alpha": 0.5, "T": 4.0, "teacher": "ConvNeXt-T s2 (val F1 0.9691, sha 9627ca43)"},
        "test_bestvalseed": {
            "seed": bvs,
            "acc": t3[bvs]["test_acc"],
            "f1": t3[bvs]["test_macro_f1"],
            "precision": t3[bvs]["test_macro_precision"],
            "recall": t3[bvs]["test_macro_recall"],
        },
    }

# ---- CMP-exp models (v4 Hướng A): val 3-seed (120 s0 + 121 s1/s2), test 3-seed (121)
cmp_exp_src: dict = {"cmp_exp_ce": {}, "cmp_exp_kd": {}}
for e in r120["results"]:
    if e["name"] in cmp_exp_src:
        cmp_exp_src[e["name"]][e["config"]["seed"]] = e
for e in r121_val["results"]:
    cmp_exp_src[e["variant"]][e["seed"]] = e
test_cmp_exp: dict = {}
for e in r121_test["results"]:
    test_cmp_exp.setdefault(e["variant"], {})[e["seed"]] = e

rn18_same_machine_ms = round(float(r121_test["same_machine_resnet18_latency"]["median_ms"]), 2)
for key in ("cmp_exp_ce", "cmp_exp_kd"):
    rows = cmp_exp_src[key]
    seeds = sorted(rows)
    t3 = test_cmp_exp[key]
    t3_seeds = sorted(t3)
    bvs = max(t3_seeds, key=lambda s: rows[s]["best_val_macro_f1"])
    train_h = round(float(np.mean([rows[s]["train_time_sec"] for s in seeds])) / 3600, 3)
    km_sec_mean = round(float(np.mean([KMEANS_SEC[key][s] for s in seeds])), 1)
    km_h = round(km_sec_mean / 3600, 3)
    entry = {
        "display": DISPLAY[key],
        "group": "proposed_ce" if key == "cmp_exp_ce" else "kd_extension",
        "params_total_m": round(rows[seeds[0]]["params"]["total"] / 1e6, 2),
        "val_f1_3seed": mean_std_dict([rows[s]["best_val_macro_f1"] for s in seeds]),
        "val_acc_3seed": mean_std_dict([rows[s]["best_val_acc"] for s in seeds]),
        "train_time_h_mean": train_h,
        "kmeans_init_sec_per_seed": KMEANS_SEC[key],
        "kmeans_init_sec_mean": km_sec_mean,
        "kmeans_init_h_mean": km_h,
        "total_train_h_mean": round(train_h + km_h, 3),
        "executed_epochs": {s: rows[s]["executed_epochs"] for s in seeds},
        "test_3seed_acc": mean_std_dict([t3[s]["test_acc"] for s in t3_seeds]),
        "test_3seed_f1": mean_std_dict([t3[s]["test_macro_f1"] for s in t3_seeds]),
        "test_3seed_precision": mean_std_dict([t3[s]["test_macro_precision"] for s in t3_seeds]),
        "test_3seed_recall": mean_std_dict([t3[s]["test_macro_recall"] for s in t3_seeds]),
        "latency_ms_3seed_mean": round(float(np.mean([t3[s]["latency"]["median_ms"] for s in t3_seeds])), 2),
        "latency_ms_bestvalseed": round(float(t3[bvs]["latency"]["median_ms"]), 2),
        "same_machine_resnet18_latency_ms": rn18_same_machine_ms,
        "latency_source": (
            "121 (3-seed mean of medians, T4 cuda, batch=1, n_warmup=10, n_iter=50; RN18 cùng máy: 4.11 ms)"
        ),
        "kappa_mean": round(float(np.mean([t3[s]["kappa"] for s in t3_seeds])), 2),
        "test_bestvalseed": {
            "seed": bvs,
            "acc": t3[bvs]["test_acc"],
            "f1": t3[bvs]["test_macro_f1"],
            "precision": t3[bvs]["test_macro_precision"],
            "recall": t3[bvs]["test_macro_recall"],
        },
    }
    if key == "cmp_exp_kd":
        entry["kd"] = {"alpha": 0.5, "T": 4.0, "teacher": "ConvNeXt-T s2 (val F1 0.9691, sha 9627ca43)"}
    master["models"][key] = entry


# ---- Paired comparisons (tính trên F1 theo seed chưa làm tròn)
def paired_f1_summary(a_map: dict, b_map: dict, a_field: str = "test_macro_f1", b_field: str = "test_macro_f1") -> dict:
    diffs_pct = [(a_map[s][a_field] - b_map[s][b_field]) * 100.0 for s in (0, 1, 2)]
    return {
        "diffs_pct_points_per_seed": [round(d, 2) for d in diffs_pct],
        "mean_diff_pct_points": round(float(np.mean(diffs_pct)), 2),
        "wins": f"{sum(d > 0 for d in diffs_pct)}/3",
    }


master["paired_comparisons_test_f1"] = {
    "asf_ce_vs_b2": paired_f1_summary(test_ce["ASF_A2fix_ent"], test_ce["mobilenetv3s"]),
    "asf_ce_vs_b1": paired_f1_summary(test_ce["ASF_A2fix_ent"], test_ce["resnet18"]),
    "cmp_v1_vs_b2": paired_f1_summary(test_ce["CMP_C2"], test_ce["mobilenetv3s"]),
    "cmp_v1_vs_b1": paired_f1_summary(test_ce["CMP_C2"], test_ce["resnet18"]),
    "cmp_exp_ce_vs_b2": paired_f1_summary(test_cmp_exp["cmp_exp_ce"], test_ce["mobilenetv3s"]),
    "cmp_exp_ce_vs_b1": paired_f1_summary(test_cmp_exp["cmp_exp_ce"], test_ce["resnet18"]),
    "cmp_exp_ce_vs_cmp_v1": paired_f1_summary(test_cmp_exp["cmp_exp_ce"], test_ce["CMP_C2"]),
    "resnet18_kd_vs_ce": paired_f1_summary(test_kd["resnet18_kd"], test_ce["resnet18"]),
    "asf_kd_vs_ce": paired_f1_summary(test_kd["asf_kd"], test_ce["ASF_A2fix_ent"]),
    "cmp_exp_kd_vs_ce": paired_f1_summary(test_cmp_exp["cmp_exp_kd"], test_cmp_exp["cmp_exp_ce"]),
}

master["audit_010"] = {
    "n_images": r010["n_images"],
    "n_classes": r010["n_classes"],
    "decode_errors": r010["sizes"]["decode_errors"],
    "exact_dup_files": r010["exact_dup_files"],
    "exact_dup_groups": r010["exact_dup_groups"],
    "neardup_candidate_pairs": r010["neardup_candidate_pairs"],
    "neardup_candidate_same_class": r010["neardup_candidate_same_class"],
    "neardup_candidate_cross_class": r010["neardup_candidate_cross_class"],
    "neardup_mutual_pairs": r010["neardup_mutual_pairs"],
    "neardup_mutual_same_class": r010["neardup_mutual_same_class"],
    "neardup_mutual_cross_class": r010["neardup_mutual_cross_class"],
    "split_counts": r010["split_counts"],
    "split_sha256": r010["split_sha256"],
}

# ---- KD overhead + teacher history ----
master["kd_overhead"] = {
    "teacher_inference_overhead": (
        "Cả CE và KD đều chạy đủ 15 epoch ở cả 3 seed; thời gian ~2× (~100s vs ~47–50s/epoch) "
        "là do teacher ConvNeXt-T forward online mỗi batch."
    ),
    "teacher_training_cost_history": {
        "model": "B4 ConvNeXt-T",
        "train_time_h_per_run": master["models"]["convnextt"]["train_time_h_mean"],
        "note": "đã chi trong v1, tái sử dụng làm teacher — không train lại",
    },
}
master["audit"] = {
    "fairness": "LOẠI A — 12 trường protocol giống hệt giữa CE và KD (notes/v4_audit.json).",
    "seed0_merge": "HỢP LỆ — s0 (110b) vs s1/s2 (111) cùng config & teacher hash 9627ca43.",
    "cmp_exp_v4": (
        "Cấu hình V4 (κ=exp(θ)∈[10,64] + hard-max + k-means init): CMP-exp-CE đạt 95.48% F1 test "
        "(vượt CMP v1 +0.39 điểm 3/3 seed và B1 +0.11 điểm 2/3 seed); CMP-exp-KD hội tụ sạch (95.74%±0.05%)."
    ),
}
master["protocol"] = {
    "split_sha256": r010["split_sha256"],
    "epochs_max": 15,
    "executed_epochs_all_runs": 15,
    "early_stop_patience": 5,
    "batch": 128,
    "optimizer": "AdamW lr 1e-4/1e-3 wd 1e-4 cosine warmup 5%",
    "checkpoint_rule": "val macro-F1",
    "seeds": [0, 1, 2],
    "amp": True,
    "input": "256x256 ImageNet-norm RRC+hflip+vflip",
    "latency_protocol": (
        "Tesla T4 (cuda), batch=1, 256x256, n_warmup=10, n_iter=50, median (ms); "
        "bảng tổng hợp gồm nhiều phiên đo (071b, 112c, 121)"
    ),
    "test_caveat": "test đã quan sát qua v1/v2/v3 — v3/v4 là nghiên cứu mở rộng theo final_plan_v3/v4 đã khóa",
}

(OUT / "results_master.json").write_text(json.dumps(master, indent=1, ensure_ascii=False), encoding="utf-8")

# CSV phẳng
csv_rows = []
for key, e in master["models"].items():
    csv_rows.append({
        "model": key,
        "display": e["display"],
        "group": e["group"],
        "params_M": e["params_total_m"],
        "val_acc_mean": e["val_acc_3seed"]["mean"],
        "val_acc_std": e["val_acc_3seed"]["std"],
        "val_f1_mean": e["val_f1_3seed"]["mean"],
        "val_f1_std": e["val_f1_3seed"]["std"],
        "test_acc_mean": e["test_3seed_acc"]["mean"],
        "test_acc_std": e["test_3seed_acc"]["std"],
        "test_f1_mean": e["test_3seed_f1"]["mean"],
        "test_f1_std": e["test_3seed_f1"]["std"],
        "test_precision_mean": e["test_3seed_precision"]["mean"],
        "test_recall_mean": e["test_3seed_recall"]["mean"],
        "test_f1_bestvalseed": e["test_bestvalseed"]["f1"],
        "latency_ms": e["latency_ms_3seed_mean"],
        "latency_bestval_ms": e["latency_ms_bestvalseed"],
        "train_h": e["train_time_h_mean"],
        "kmeans_h": e["kmeans_init_h_mean"],
        "total_train_h": e["total_train_h_mean"],
        "teacher": e.get("kd") is not None,
    })
pd.DataFrame(csv_rows).to_csv(OUT / "results_master.csv", index=False)
print(f"saved {OUT / 'results_master.json'} + results_master.csv")
for r in csv_rows:
    print(
        f"  {r['display']:<20} testAcc {r['test_acc_mean']:.4f} testF1 {r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f} "
        f"lat {r['latency_ms']}ms train {r['train_h']}h (+km {r['kmeans_h']}h)"
    )
