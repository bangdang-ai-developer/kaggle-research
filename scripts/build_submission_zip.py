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
    entries: list[tuple[Path, str]] = [
        (ROOT / "README.md", "README.md"),
        (DELIV / "Bao_cao_cuoi_mon_NWPU_RESISC45.docx", "Bao_cao/Bao_cao_cuoi_mon_NWPU_RESISC45.docx"),
        (DELIV / "Bao_cao_ket_qua_nghien_cuu.md", "Bao_cao/Bao_cao_ket_qua_nghien_cuu.md"),
        (DELIV / "Slides_cuoi_mon_NWPU_RESISC45.pptx", "Slides/Slides_cuoi_mon_NWPU_RESISC45.pptx"),
        (DELIV / "results_master.json", "So_lieu/results_master.json"),
        (DELIV / "results_master.csv", "So_lieu/results_master.csv"),
        (DELIV / "manifest.json", "So_lieu/manifest_artifacts.json"),
        (DELIV / "demo_smoke.json", "Demo/demo_smoke.json"),
        (DELIV / "README_REPRODUCE.md", "README_REPRODUCE.md"),
        (ROOT / "demo" / "app.py", "Demo/app.py"),
        (ROOT / "src" / "models.py", "src/models.py"),
        (ROOT / "AGENTS.md", "AGENTS.md"),
        (ROOT / "CLAUDE.md", "CLAUDE.md"),
    ]
    for p in sorted((ROOT / "scripts").glob("*.py")):
        entries.append((p, f"scripts/{p.name}"))
    for p in sorted((ROOT / "tests").glob("*.py")):
        entries.append((p, f"tests/{p.name}"))
    for ext in ("*.md", "*.csv", "*.json"):
        for p in sorted((ROOT / "notes").glob(ext)):
            entries.append((p, f"notes/{p.name}"))
    for p in sorted((ROOT / "notebooks").glob("*/nb.py")):
        entries.append((p, f"notebooks/{p.parent.name}/nb.py"))
    for p in sorted((ROOT / "notes" / "figures").glob("*.png")):
        entries.append((p, f"Hinh/figures/{p.name}"))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            if src.exists():
                zf.write(src, arc)
    print(f"zip: {zip_path} ({len(entries)} files, {zip_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build_manifest()
    build_zip()
