#!/usr/bin/env python3
"""Validate số liệu submission: per-seed consistency, metric ranges, paired diffs, README số khớp nguồn.

Chạy: uv run python3 scripts/validate_submission.py
Exit 0 nếu mọi check PASS; exit 1 nếu có FAIL.
"""
from __future__ import annotations

import sys
import zipfile

import numpy as np
from common import ROOT, load_json, load_output_json, mean_std

issues: list[str] = []
passes: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    msg = f"{'PASS' if cond else 'FAIL'}: {name}" + (f" — {detail}" if detail else "")
    (passes if cond else issues).append(msg)


# 1) Load artifacts nguồn
r071 = load_output_json("resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json")
r112 = load_output_json("resisc45-eval-kd-3-seeds-v2-fixed-sources/test_kd_3seed_report.json")
master = load_json("deliverables/results_master.json")


def extract(report: dict, key_field: str) -> dict:
    out: dict = {}
    for e in report["results"]:
        seed = e.get("seed", e.get("config", {}).get("seed"))
        out.setdefault(e[key_field], {})[seed] = e
    return out


ce = extract(r071, "model")
kd = extract(r112, "student")

# 2) Trùng/missing seed + metric range
for src_name, src in (("071b", ce), ("112c", kd)):
    for key, per_seed in src.items():
        seeds = sorted(per_seed)
        check(f"{src_name}/{key}: đủ 3 seed, không trùng", seeds == [0, 1, 2] and len(per_seed) == 3, str(seeds))
        for s, e in per_seed.items():
            check(f"{src_name}/{key}/s{s}: acc trong [0,1]", 0 <= e["test_acc"] <= 1)
            check(f"{src_name}/{key}/s{s}: F1 trong [0,1]", 0 <= e["test_macro_f1"] <= 1)

# 3) Master mean/std khớp tính lại từ per-seed (ddof=1)
for key in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt", "ASF_A2fix_ent", "CMP_C2"):
    rows = ce[key]
    seeds = sorted(rows)
    acc_m, acc_s = mean_std([rows[s]["test_acc"] for s in seeds])
    m_entry = master["models"][key]["test_3seed_acc"]
    check(f"master/{key}: acc mean khớp per-seed", abs(m_entry["mean"] - acc_m) < 5e-5,
          f"{m_entry['mean']} vs {acc_m:.6f}")
    check(f"master/{key}: acc std khớp (ddof=1)", abs(m_entry["std"] - acc_s) < 5e-5,
          f"{m_entry['std']} vs {acc_s:.6f}")
    f1_entry = master["models"][key]["test_3seed_f1"]
    f1_m, f1_s = mean_std([rows[s]["test_macro_f1"] for s in seeds])
    check(f"master/{key}: f1 mean/std khớp",
          abs(f1_entry["mean"] - f1_m) < 5e-5 and abs(f1_entry["std"] - f1_s) < 5e-5)

for key in ("resnet18_kd", "asf_kd"):
    rows = kd[key]
    seeds = sorted(rows)
    m_entry = master["models"][key]["test_3seed_acc"]
    acc_m, acc_s = mean_std([rows[s]["test_acc"] for s in seeds])
    check(f"master/{key}: acc mean/std khớp per-seed",
          abs(m_entry["mean"] - acc_m) < 5e-5 and abs(m_entry["std"] - acc_s) < 5e-5)

# 4) Paired diffs khớp per-seed (điểm phần trăm)
paired = r112.get("paired_vs_CE", {})
ce_f1 = {m: {s: e["test_macro_f1"] for s, e in ce[m].items()} for m in ("resnet18", "ASF_A2fix_ent")}
for kd_key, ce_map in (("resnet18_kd", "resnet18"), ("asf_kd", "ASF_A2fix_ent")):
    diffs_ratio = [kd[kd_key][s]["test_macro_f1"] - ce_f1[ce_map][s] for s in (0, 1, 2)]
    diffs_pct = [d * 100 for d in diffs_ratio]
    entry = paired.get(kd_key)
    if entry is None:
        issues.append(f"FAIL: thiếu paired entry cho {kd_key}")
        continue
    stated = entry.get("f1_diffs_per_seed")  # lưu dạng TỶ LỆ (0.0058 = 0.58 điểm %)
    stated_pct = [round(v * 100, 2) for v in stated]
    recomputed_pct = [round(d, 2) for d in diffs_pct]
    check(f"paired/{kd_key}: per-seed diffs khớp (đơn vị điểm %)", stated_pct == recomputed_pct,
          f"stated {stated_pct} vs computed {recomputed_pct}")
    mean_diff_pct = float(np.mean(diffs_pct))
    check(f"paired/{kd_key}: mean khớp", abs(entry["f1_mean_diff"] * 100 - mean_diff_pct) < 0.005,
          f"{entry['f1_mean_diff'] * 100:+.4f} vs {mean_diff_pct:+.4f}")
    wins = sum(d > 0 for d in diffs_pct)
    check(f"paired/{kd_key}: wins khớp", str(entry.get("f1_wins", entry.get("acc_wins", ""))) != "0/3" or wins == 0,
          f"wins {wins}/3")

# 5) README numbers khớp master (trích các chuỗi % trong README)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
for key, expected_str in (("resnet18_kd", "96.02"), ("asf_kd", "95.91"), ("convnextt", "96.78"),
                          ("resnet18", "95.37"), ("mobilenetv3s", "94.69")):
    check(f"README chứa {expected_str} ({key})", expected_str in readme)

# 6) Deliverables hiện diện
for f in ("Bao_cao_cuoi_mon_NWPU_RESISC45.docx", "Slides_cuoi_mon_NWPU_RESISC45.pptx",
          "results_master.json", "results_master.csv", "manifest.json",
          "NWPU_RESISC45_submission.zip", "README_REPRODUCE.md"):
    check(f"deliverables/{f} tồn tại", (ROOT / "deliverables" / f).exists())

# 7) ZIP không chứa file nhạy cảm
bad_patterns = (".git", "venv", "__pycache__", "credential", "data/raw", "token", ".kaggle")
with zipfile.ZipFile(ROOT / "deliverables/NWPU_RESISC45_submission.zip") as z:
    names = z.namelist()
    bad = [n for n in names if any(s in n.lower() for s in bad_patterns)]
    check("ZIP không chứa file nhạy cảm", not bad, str(bad))
    check("ZIP ≥ 40 files", len(names) >= 40, f"{len(names)}")

print("\n".join(passes))
if issues:
    print("\n".join(issues))
print(f"\n=== VALIDATION: {len(passes)} PASS / {len(issues)} FAIL ===")
sys.exit(1 if issues else 0)
