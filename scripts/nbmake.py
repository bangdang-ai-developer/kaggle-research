#!/usr/bin/env python3
"""Build notebook.ipynb tu nguon nb.py (marker # %%).

Cach dung:
    uv run python3 scripts/nbmake.py notebooks/<slug>/

Nguon notebooks/<slug>/nb.py dung marker:
    # %%            -> code cell
    # %% [md]       -> markdown cell
    # %% embed:src/models.py  -> 1 code cell chua noi dung file do (tuong doi project root)

Output: notebooks/<slug>/notebook.ipynb (ghi de).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent


def parse(nb_py: Path) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    kind, buf = "code", []
    for line in nb_py.read_text(encoding="utf-8").splitlines():
        stripped = line.rstrip("\n")
        if stripped.startswith("# %%"):
            if buf and "\n".join(buf).strip():
                cells.append((kind, "\n".join(buf).strip("\n")))
            rest = stripped[4:].strip()
            if rest.startswith("[md]"):
                kind, buf = "markdown", []
            elif rest.startswith("embed:"):
                rel = rest[len("embed:") :].strip()
                src = ROOT / rel
                if not src.exists():
                    sys.exit(f"nbmake: embed source not found: {src}")
                body = src.read_text(encoding="utf-8")
                cells.append(("code", f"# ===== embedded from {rel} (single source of logic) =====\n{body}"))
                kind, buf = "code", []
            else:
                kind, buf = "code", []
        else:
            buf.append(line)
    if buf and "\n".join(buf).strip():
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    slug_dir = Path(sys.argv[1])
    nb_py = slug_dir / "nb.py"
    if not nb_py.exists():
        sys.exit(f"nbmake: {nb_py} not found")
    nb = nbf.v4.new_notebook()
    # Kaggle papermill bắt buộc kernelspec ("No kernel name found" nếu thiếu)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "markdown" else nbf.v4.new_code_cell(s) for k, s in parse(nb_py)]
    out = slug_dir / "notebook.ipynb"
    nbf.write(nb, out)
    print(f"nbmake: wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
