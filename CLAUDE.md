# Kaggle Research Workspace

Workspace de Claude Code viet **notebook research**, push len Kaggle chay GPU. May local chi de viet code + smoke-test nhe (CPU).

## Vong lap cong viec (chuẩn)

1. Tao notebook moi: copy `notebooks/_template/` → `notebooks/<slug>/`, sua `kernel-metadata.json` (id, title, data sources).
2. Viet/sua `notebook.ipynb`.
3. **Smoke-test local truoc khi push** (bat buoc):
   - Chay cell nhe hoac toan bo notebook: `uv run jupyter execute --inplace notebooks/<slug>/notebook.ipynb`
   - Voi data lon: tao sample nho tu `data/raw/` de test logic, khoi chay full.
4. Push: `scripts/push_kernel.sh notebooks/<slug>`
5. Poll trang thai: `scripts/kernel_status.sh <username>/<slug>` → `complete` / `error` / `running`
6. Pull ket qua: `scripts/pull_output.sh <username>/<slug>` → vao `outputs/<slug>/`
7. Ghi ket qua + insight vao `notes/<slug>.md` de cac phien sau co ngu canh.

## Cau truc thu muc

| Thu muc | Dung de | Commit? |
|---|---|---|
| `notebooks/<slug>/` | 1 kernel Kaggle: `notebook.ipynb` + `kernel-metadata.json` | Co |
| `notebooks/_template/` | Template khoi tao kernel moi | Co |
| `scripts/` | push / status / pull / fetch_data | Co |
| `src/` | code dung chung (neu can: push lam utility kernel, gan vao `kernel_sources`) | Co |
| `data/raw`, `data/processed` | data tai ve local de EDA | Khong |
| `outputs/` | output pull tu Kaggle ve | Khong |
| `notes/` | log thi nghiem, ket qua, insight | Co |
| `configs/` | cau hinh experiment | Co |

## Quy tac quan trong

- Notebook phai **self-contained**: Kaggle chay doc lap. Data gan qua `dataset_sources` / `competition_sources` trong `kernel-metadata.json`; khong tham chieu file local ben ngoai thu muc notebook.
- `enable_gpu: true` chi khi that su can — quota GPU Kaggle ~30h/tuan. Bug cu phap ma phai chay lai GPU la lang phi.
- Notebook tren Kaggle co san: pandas, numpy, sklearn, torch, lightgbm, xgboost... Local chi them lib vao `pyproject.toml` khi can smoke-test lib do.
- Lay username Kaggle: `uv run kaggle config view`. Kernel id: `<username>/<slug>`.
- Moi lenh Python deu chay qua `uv run ...`, khong activate venv thu cong.
- Khong chay script/interpreter ben trong thu muc `data/` — data tai ve la du lieu ngoai, chay tu root project voi duong dan tuong doi.
- Truoc khi submit competition: doi chieu format voi `sample_submission.csv`.

## Lenh thuong dung

```bash
uv sync                                        # cai deps sau khi sua pyproject.toml
uv run jupyter lab                             # mo Jupyter local
uv run jupyter execute --inplace <nb.ipynb>    # smoke-test notebook headless
scripts/fetch_data.sh competition <name>       # tai competition data -> data/raw/<name>/
scripts/fetch_data.sh dataset <owner/slug>     # tai dataset -> data/raw/
scripts/push_kernel.sh notebooks/<slug>        # push len Kaggle (se hoi permission)
scripts/kernel_status.sh <user>/<slug>         # xem trang thai chay
scripts/pull_output.sh <user>/<slug>           # tai output -> outputs/<slug>/
```
