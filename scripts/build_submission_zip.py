#!/usr/bin/env python3
"""Cập nhật deliverables/manifest.json và đóng gói sạch deliverables/NWPU_RESISC45_submission.zip.

Chạy: uv run python3 scripts/build_submission_zip.py
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from common import ROOT

DELIV = ROOT / "deliverables"

MANIFEST_FILES = [
    "outputs/resisc45-final-baselines-b0-b4-3-seeds/ckpt_resnet18_s2.pt",
    "outputs/resisc45-final-proposed-asf-cmp-3-seeds/ckpt_ASF_A2fix_ent_s1.pt",
    "outputs/resisc45-final-proposed-asf-cmp-3-seeds/ckpt_CMP_C2_s1.pt",
    "outputs/resisc45-final-baselines-b0-b4-3-seeds/ckpt_convnextt_s2.pt",
    "outputs/resisc45-screen-kd-resnet18-asf-cmp-v2/ckpt_resnet18_kd.pt",
    "outputs/resisc45-kd-final-seeds-1-2/ckpt_asf_kd_s1.pt",
    "outputs/resisc45-h1a-screen-cmp-exp-asf-lite/ckpt_cmp_exp_kd.pt",
    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/ckpt_cmp_exp_ce_s1.pt",
    "outputs/resisc45-data-audit-split-manifest/split_manifest.csv",
    "outputs/resisc45-data-audit-split-manifest/class_order.json",
    "outputs/resisc45-final-evaluation-v2-offline/test_report.json",
    "outputs/resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json",
    "outputs/resisc45-eval-kd-3-seeds-v2-fixed-sources/test_kd_3seed_report.json",
    "outputs/resisc45-screen-kd-resnet18-asf-cmp-v2/kd_screen_report.json",
    "outputs/resisc45-kd-final-seeds-1-2/kd_final_report.json",
    "outputs/resisc45-h1a-screen-cmp-exp-asf-lite/h1a_screen_report.json",
    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_final_report.json",
    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_test_3seed_report.json",
]

EVIDENCE_OUTPUT_FILES = [
    "outputs/resisc45-data-audit-split-manifest/audit_report.json",
    "outputs/resisc45-data-audit-split-manifest/class_order.json",
    "outputs/resisc45-data-audit-split-manifest/near_dup_pairs.csv",
    "outputs/resisc45-data-audit-split-manifest/split_manifest.csv",
    "outputs/resisc45-data-audit-split-manifest/resisc45-data-audit-split-manifest.log",
    "outputs/resisc45-final-baselines-b0-b4-3-seeds/final_baselines_report.json",
    "outputs/resisc45-final-proposed-asf-cmp-3-seeds/final_proposed_report.json",
    "outputs/resisc45-final-evaluation-v2-offline/test_report.json",
    "outputs/resisc45-final-evaluation-v2-offline/robustness_table.csv",
    "outputs/resisc45-expanded-eval-3-seeds-v2-fixed/test_3seed_report.json",
    "outputs/resisc45-screen-kd-resnet18-asf-cmp-v2/kd_screen_report.json",
    "outputs/resisc45-kd-final-seeds-1-2/kd_final_report.json",
    "outputs/resisc45-eval-kd-3-seeds-v2-fixed-sources/test_kd_3seed_report.json",
    "outputs/resisc45-h1a-screen-cmp-exp-asf-lite/h1a_screen_report.json",
    "outputs/resisc45-h1a-screen-cmp-exp-asf-lite/resisc45-h1a-screen-cmp-exp-asf-lite.log",
    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_final_report.json",
    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/cmp_exp_test_3seed_report.json",
    "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/resisc45-cmp-exp-3-seeds-and-test-eval.log",
]


def sha256_file(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": h.hexdigest()}


def build_manifest() -> None:
    manifest = {rel: sha256_file(ROOT / rel) for rel in MANIFEST_FILES if (ROOT / rel).exists()}
    (DELIV / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {len(manifest)} entries")


def build_zip() -> None:
    zip_path = DELIV / "NWPU_RESISC45_submission.zip"
    root_files = (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "AGENTS.md",
        "CLAUDE.md",
        "src/models.py",
        "demo/app.py",
        "deliverables/Bao_cao_cuoi_mon_NWPU_RESISC45.docx",
        "deliverables/Bao_cao_ket_qua_nghien_cuu.md",
        "deliverables/Slides_cuoi_mon_NWPU_RESISC45.pptx",
        "deliverables/results_master.json",
        "deliverables/results_master.csv",
        "deliverables/manifest.json",
        "deliverables/demo_smoke.json",
        "deliverables/README_REPRODUCE.md",
        *EVIDENCE_OUTPUT_FILES,
    )
    entries: list[tuple[Path, str]] = [(ROOT / rel, rel) for rel in root_files]
    for pat in ("*.py", "*.sh"):
        for p in sorted((ROOT / "scripts").glob(pat)):
            entries.append((p, f"scripts/{p.name}"))
    for p in sorted((ROOT / "tests").glob("*.py")):
        entries.append((p, f"tests/{p.name}"))
    for ext in ("*.md", "*.csv", "*.json"):
        for p in sorted((ROOT / "notes").glob(ext)):
            entries.append((p, f"notes/{p.name}"))
    for p in sorted((ROOT / "notes" / "figures").glob("*.png")):
        entries.append((p, f"notes/figures/{p.name}"))
    for slug_dir in sorted((ROOT / "notebooks").iterdir()):
        if not slug_dir.is_dir():
            continue
        for fname in ("nb.py", "kernel-metadata.json"):
            p = slug_dir / fname
            if p.exists():
                entries.append((p, f"notebooks/{slug_dir.name}/{fname}"))

    added = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            if src.exists():
                zf.write(src, arc)
                added += 1
    print(f"zip: {zip_path} ({added} files, {zip_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build_manifest()
    build_zip()
