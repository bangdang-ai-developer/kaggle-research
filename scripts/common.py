"""Shared artifact repository & statistics helpers cho scripts/."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_json(path: str | Path) -> Any:
    """Đọc JSON an toàn bằng context manager; path tương đối tính từ ROOT."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def load_output_json(rel_path: str) -> Any:
    """Đọc JSON report từ thư mục outputs/."""
    return load_json(ROOT / "outputs" / rel_path)


def mean_std(vals: Sequence[float]) -> tuple[float, float]:
    """Tính (mean, sample std ddof=1) cho tập seed."""
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def mean_std_dict(vals: Sequence[float], digits: int = 4) -> dict[str, float]:
    """Trả về {"mean": ..., "std": ...} làm tròn chuẩn results_master."""
    m, s = mean_std(vals)
    return {"mean": round(m, digits), "std": round(s, digits)}
