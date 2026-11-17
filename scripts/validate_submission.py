#!/usr/bin/env python3
"""Validate số liệu submission: duplicate (model, seed), seed counts, metric ranges, paired diffs,
bảng chính trong README & báo cáo markdown, và cấu trúc ZIP sạch.

Chạy: uv run python3 scripts/validate_submission.py
Exit 0 nếu mọi check PASS; exit 1 nếu có FAIL.
"""
from __future__ import annotations

import re
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
r121 = load_output_json("resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_test_3seed_report.json")
r060a = load_output_json("resisc45-final-baselines-b0-b4-3-seeds/final_baselines_report.json")
r060b = load_output_json("resisc45-final-proposed-asf-cmp-3-seeds/final_proposed_report.json")
r110b = load_output_json("resisc45-screen-kd-resnet18-asf-cmp-v2/kd_screen_report.json")
r111 = load_output_json("resisc45-kd-final-seeds-1-2/kd_final_report.json")
r120 = load_output_json("resisc45-h1a-screen-cmp-exp-asf-lite/h1a_screen_report.json")
r010 = load_output_json("resisc45-data-audit-split-manifest/audit_report.json")
master = load_json("deliverables/results_master.json")


def extract(report: dict, key_field: str, report_name: str) -> dict:
    """Trích xuất theo (model, seed), phát hiện dòng trùng TRƯỚC khi đưa vào dict."""
    out: dict = {}
    seen: set[tuple[str, object]] = set()
    dups: list[tuple[str, object]] = []
    for e in report["results"]:
        seed = e.get("seed", e.get("config", {}).get("seed"))
        pair = (e[key_field], seed)
        if pair in seen:
            dups.append(pair)
            continue
        seen.add(pair)
        out.setdefault(e[key_field], {})[seed] = e
    check(f"{report_name}: không trùng (model, seed) trước khi lưu dict", not dups, f"duplicates={dups}")
    return out


ce = extract(r071, "model", "071b")
kd = extract(r112, "student", "112c")
cmp_exp = extract(r121, "variant", "121")
extract(r060a, "model", "060a")
extract(r060b, "variant", "060b")
extract(r110b, "student", "110b")
extract(r111, "student", "111")
extract(r120, "name", "120")

# 2) Đủ 3 seed & metric range
for src_name, src in (("071b", ce), ("112c", kd), ("121", cmp_exp)):
    for key, per_seed in src.items():
        seeds = sorted(per_seed)
        check(f"{src_name}/{key}: đủ 3 seed [0,1,2]", seeds == [0, 1, 2], str(seeds))
        for s, e in per_seed.items():
            check(f"{src_name}/{key}/s{s}: acc trong [0,1]", 0 <= e["test_acc"] <= 1)
            check(f"{src_name}/{key}/s{s}: F1 trong [0,1]", 0 <= e["test_macro_f1"] <= 1)

# 3) Master mean/std & executed_epochs = 15 & audit_010
all_sources = [
    ("071b", ce, ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt", "ASF_A2fix_ent", "CMP_C2")),
    ("112c", kd, ("resnet18_kd", "asf_kd")),
    ("121", cmp_exp, ("cmp_exp_ce", "cmp_exp_kd")),
]
for _src_name, src, keys in all_sources:
    for key in keys:
        rows = src[key]
        seeds = sorted(rows)
        acc_m, acc_s = mean_std([rows[s]["test_acc"] for s in seeds])
        m_acc = master["models"][key]["test_3seed_acc"]
        check(f"master/{key}: acc mean/std khớp per-seed",
              abs(m_acc["mean"] - acc_m) < 5e-5 and abs(m_acc["std"] - acc_s) < 5e-5,
              f"{m_acc} vs {acc_m:.4f}±{acc_s:.4f}")
        f1_m, f1_s = mean_std([rows[s]["test_macro_f1"] for s in seeds])
        m_f1 = master["models"][key]["test_3seed_f1"]
        check(f"master/{key}: f1 mean/std khớp per-seed",
              abs(m_f1["mean"] - f1_m) < 5e-5 and abs(m_f1["std"] - f1_s) < 5e-5,
              f"{m_f1} vs {f1_m:.4f}±{f1_s:.4f}")
        ep = master["models"][key].get("executed_epochs", {})
        check(f"master/{key}: executed_epochs = 15 ở cả 3 seed",
              all(ep.get(str(s), ep.get(s)) == 15 for s in (0, 1, 2)), str(ep))

a10 = master.get("audit_010", {})
check("audit_010: candidate=20, mutual=16 khớp raw audit_report.json",
      a10.get("neardup_candidate_pairs") == r010["neardup_candidate_pairs"] == 20
      and a10.get("neardup_mutual_pairs") == r010["neardup_mutual_pairs"] == 16,
      f"master={a10.get('neardup_candidate_pairs')}/{a10.get('neardup_mutual_pairs')}")

# 4) Paired diffs khớp per-seed (điểm phần trăm) & khớp results_master.json
paired = r112.get("paired_vs_CE", {})
ce_f1 = {
    m: {s: e["test_macro_f1"] for s, e in ce[m].items()}
    for m in ("resnet18", "ASF_A2fix_ent", "CMP_C2", "mobilenetv3s")
}
for kd_key, ce_map in (("resnet18_kd", "resnet18"), ("asf_kd", "ASF_A2fix_ent")):
    diffs_pct = [(kd[kd_key][s]["test_macro_f1"] - ce_f1[ce_map][s]) * 100 for s in (0, 1, 2)]
    entry = paired.get(kd_key)
    if entry is None:
        issues.append(f"FAIL: thiếu paired entry cho {kd_key}")
        continue
    stated_pct = [round(v * 100, 2) for v in entry.get("f1_diffs_per_seed")]
    recomputed_pct = [round(d, 2) for d in diffs_pct]
    check(f"paired/{kd_key}: per-seed diffs khớp 112c (điểm %)", stated_pct == recomputed_pct,
          f"stated {stated_pct} vs computed {recomputed_pct}")
    mean_diff_pct = float(np.mean(diffs_pct))
    check(f"paired/{kd_key}: mean khớp 112c", abs(entry["f1_mean_diff"] * 100 - mean_diff_pct) < 0.005,
          f"{entry['f1_mean_diff'] * 100:+.4f} vs {mean_diff_pct:+.4f}")

expected_paired = {
    "asf_ce_vs_b2": [ce_f1["ASF_A2fix_ent"][s] - ce_f1["mobilenetv3s"][s] for s in (0, 1, 2)],
    "asf_ce_vs_b1": [ce_f1["ASF_A2fix_ent"][s] - ce_f1["resnet18"][s] for s in (0, 1, 2)],
    "cmp_v1_vs_b2": [ce_f1["CMP_C2"][s] - ce_f1["mobilenetv3s"][s] for s in (0, 1, 2)],
    "cmp_v1_vs_b1": [ce_f1["CMP_C2"][s] - ce_f1["resnet18"][s] for s in (0, 1, 2)],
    "cmp_exp_ce_vs_b2": [cmp_exp["cmp_exp_ce"][s]["test_macro_f1"] - ce_f1["mobilenetv3s"][s] for s in (0, 1, 2)],
    "cmp_exp_ce_vs_b1": [cmp_exp["cmp_exp_ce"][s]["test_macro_f1"] - ce_f1["resnet18"][s] for s in (0, 1, 2)],
    "cmp_exp_ce_vs_cmp_v1": [cmp_exp["cmp_exp_ce"][s]["test_macro_f1"] - ce_f1["CMP_C2"][s] for s in (0, 1, 2)],
    "resnet18_kd_vs_ce": [kd["resnet18_kd"][s]["test_macro_f1"] - ce_f1["resnet18"][s] for s in (0, 1, 2)],
    "asf_kd_vs_ce": [kd["asf_kd"][s]["test_macro_f1"] - ce_f1["ASF_A2fix_ent"][s] for s in (0, 1, 2)],
    "cmp_exp_kd_vs_ce": [
        cmp_exp["cmp_exp_kd"][s]["test_macro_f1"] - cmp_exp["cmp_exp_ce"][s]["test_macro_f1"]
        for s in (0, 1, 2)
    ],
}
pc = master.get("paired_comparisons_test_f1", {})
for pname, raw_diffs in expected_paired.items():
    m_entry = pc.get(pname, {})
    recomp = [round(d * 100, 2) for d in raw_diffs]
    recomp_mean = round(float(np.mean([d * 100 for d in raw_diffs])), 2)
    check(f"master paired/{pname} khớp",
          m_entry.get("diffs_pct_points_per_seed") == recomp and m_entry.get("mean_diff_pct_points") == recomp_mean,
          f"master={m_entry} vs {recomp}/{recomp_mean}")


# 5) Parse markdown tables để kiểm tra số nằm ĐÚNG hàng & ĐÚNG cột
def clean_md_cell(s: str) -> str:
    return re.sub(r"[*`]", "", s).strip()


def parse_md_tables(md_text: str) -> list[tuple[list[str], list[list[str]]]]:
    tables = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            headers = [clean_md_cell(c) for c in lines[i].strip().strip("|").split("|")]
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([clean_md_cell(c) for c in lines[j].strip().strip("|").split("|")])
                j += 1
            tables.append((headers, rows))
            i = j
        else:
            i += 1
    return tables


def verify_md_table(doc_name: str, text: str, col_acc: str, col_f1: str, row_aliases: dict[str, str]) -> None:
    tables = parse_md_tables(text)
    found: dict[str, dict[str, str]] = {}
    for headers, rows in tables:
        if col_acc in headers and col_f1 in headers and headers[0] == "Model":
            idx_acc, idx_f1 = headers.index(col_acc), headers.index(col_f1)
            for r in rows:
                if len(r) > max(idx_acc, idx_f1):
                    found[r[0]] = {col_acc: r[idx_acc], col_f1: r[idx_f1]}
    for mkey, label_prefix in row_aliases.items():
        matched = [v for row_label, v in found.items() if row_label.startswith(label_prefix)]
        if not matched:
            issues.append(f"FAIL: {doc_name} thiếu hàng bảng cho {mkey} ({label_prefix})")
            continue
        cell = matched[0]
        exp_acc = f"{master['models'][mkey]['test_3seed_acc']['mean'] * 100:.2f}"
        exp_f1 = f"{master['models'][mkey]['test_3seed_f1']['mean'] * 100:.2f}"
        check(f"{doc_name} bảng [{label_prefix}]: Acc={exp_acc} & F1={exp_f1} đúng hàng/cột",
              exp_acc in cell[col_acc] and exp_f1 in cell[col_f1],
              f"got acc={cell[col_acc]!r}, f1={cell[col_f1]!r}")


ALL_ROW_ALIASES = {
    "smallcnn": "B0 SmallCNN",
    "resnet18": "B1 ResNet18-CE",
    "mobilenetv3s": "B2 MobileNetV3-S",
    "efficientnetb0": "B3 EfficientNet-B0",
    "convnextt": "B4 ConvNeXt-T",
    "ASF_A2fix_ent": "ASF-CE",
    "CMP_C2": "CMP-CE",
    "cmp_exp_ce": "CMP-exp-CE",
    "resnet18_kd": "ResNet18-KD",
    "asf_kd": "ASF-KD",
    "cmp_exp_kd": "CMP-exp-KD",
}

readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
verify_md_table("README.md", readme_text, "Test Acc %", "Test Macro-F1 %", ALL_ROW_ALIASES)

bao_cao_md = (ROOT / "deliverables" / "Bao_cao_ket_qua_nghien_cuu.md").read_text(encoding="utf-8")
verify_md_table("Bao_cao_ket_qua_nghien_cuu.md", bao_cao_md, "Acc % ± std", "F1 % ± std", ALL_ROW_ALIASES)

# 6) Deliverables hiện diện
for f in ("Bao_cao_cuoi_mon_NWPU_RESISC45.docx", "Slides_cuoi_mon_NWPU_RESISC45.pptx",
          "Bao_cao_ket_qua_nghien_cuu.md", "results_master.json", "results_master.csv", "manifest.json",
          "README_REPRODUCE.md", "demo_smoke.json"):
    check(f"deliverables/{f} tồn tại", (ROOT / "deliverables" / f).exists())

# 7) ZIP check (khi chạy trong repo có ZIP; khi giải nén vào thư mục sạch thì bản ZIP có thể nằm ở nơi user giải nén)
zip_path = ROOT / "deliverables" / "NWPU_RESISC45_submission.zip"
if zip_path.exists():
    bad_patterns = (".git", "venv", "__pycache__", "credential", "data/raw", "token", ".kaggle", ".env", ".pt")
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        bad = [n for n in names if any(s in n.lower() for s in bad_patterns) and ".env.example" not in n.lower()]
        check("ZIP không chứa file nhạy cảm hoặc checkpoint .pt", not bad, str(bad))
        check("ZIP ≥ 70 files", len(names) >= 70, f"{len(names)}")
        for req in ("README.md", "pyproject.toml", "uv.lock", ".python-version", "demo/app.py", "src/models.py",
                    "deliverables/Bao_cao_cuoi_mon_NWPU_RESISC45.docx",
                    "deliverables/Slides_cuoi_mon_NWPU_RESISC45.pptx",
                    "deliverables/results_master.json",
                    "deliverables/README_REPRODUCE.md",
                    "notebooks/120-h1a-screen/nb.py",
                    "notebooks/121-cmp-exp-final-eval/nb.py",
                    "outputs/resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json",
                    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_test_3seed_report.json"):
            check(f"ZIP chứa {req}", req in names)

print("\n".join(passes))
if issues:
    print("\n".join(issues))
print(f"\n=== VALIDATION: {len(passes)} PASS / {len(issues)} FAIL ===")
sys.exit(1 if issues else 0)
