# Reproduction — hướng dẫn tái lập

**Yêu cầu:** tài khoản Kaggle (GPU quota ~11h cho toàn bộ final; pilot/screening thêm ~4h), uv 0.13+, Python 3.12 local (3.13 trên Kaggle).

## 0. Một nguồn logic

- Toàn bộ model/train/eval: [src/models.py](../src/models.py)
- Notebook đều được sinh từ `nb.py` bằng: `uv run python3 scripts/nbmake.py notebooks/<slug>/`
- **Không sửa notebook.ipynb trực tiếp** — sửa nb.py / src/models.py rồi rebuild.

## 1. Workspace

```bash
git clone <repo> && cd kaggle-research
uv sync                                          # deps local (smoke CPU)
uv run kaggle config view                        # kiểm tra username + auth (~/.kaggle/access_token)
```

## 2. Thứ tự kernel (mỗi kernel: build → smoke local → push → poll → pull)

| # | Kernel | Slug Kaggle | GPU | Thời gian | Mục đích |
|---|---|---|---|---|---|
| 1 | notebooks/010-data-audit | resisc45-data-audit-split-manifest | CPU | 7–40' | audit + split manifest chuẩn (sha `cc70020a…`) |
| 2 | notebooks/020-pilot-v2 | resisc45-pilot-v2-local-stage | T4 | ~30' | timing + sanity forward/backward |
| 3 | notebooks/030-baselines | resisc45-screen-baselines-b0-b4 | T4 | ~50' | screening B0–B4 8 epoch |
| 4 | notebooks/040-asf-ablation → 040b-asf-fix | resisc45-screen-asf-* | T4 | ~25' ×2 | ablation ASF + vòng fix |
| 5 | notebooks/050-cmp-ablation → 050b-cmp-fix | resisc45-screen-cmp-* | T4 | ~30' ×2 | ablation CMP + vòng fix |
| 6 | notebooks/060a-final-baselines | resisc45-final-baselines-b0-b4-3-seeds | T4 | ~5h | final 5 baseline × 3 seed |
| 7 | notebooks/060b-final-proposed | resisc45-final-proposed-asf-cmp-3-seeds | T4 | ~1.5h | final ASF + CMP × 3 seed |
| 8 | notebooks/070b-final-evaluation | resisc45-final-evaluation-v2-offline | T4 | ~40' | test + latency + robustness (1 lần duy nhất) |
| 9 | notebooks/100b→110b→111→112c | resisc45-kd-* / resisc45-eval-kd-3-seeds-v2-fixed-sources | T4 | ~4.1h | KD pilot, screening, final 3-seed + test |
| 10 | notebooks/120-h1a-screen → 121-cmp-exp-final-eval | resisc45-h1a-screen-* / resisc45-cmp-exp-3-seeds-and-test-eval | T4 | ~2.5h | v4 Hướng A: CMP κ=exp screening + 3-seed + test |

```bash
./scripts/push_kernel.sh notebooks/<slug>                       # push (cần xác nhận theo permissions)
./scripts/kernel_status.sh dangbang1/<slug-kaggle>              # poll (slug = THEO TITLE, xem URL kernel)
./scripts/pull_output.sh dangbang1/<slug-kaggle>                # pull artifacts → outputs/<slug>/
```

## 3. Lưu ý bắt buộc (mỗi cái từng gây fail — chi tiết notes/progress.md)

1. **010 phải chạy trước mọi kernel** — split manifest là nguồn duy nhất (fail-fast kiểm tra sha256 `cc70020a…`).
2. **Stage local**: mọi kernel huấn luyện đã tự copy dataset → `/tmp` (FUSE 17–35 ảnh/s là bottleneck; không stage = GPU 0%).
3. **Pretrained weights** cần `enable_internet: true` khi **train**; khi **chỉ load checkpoint** dùng `build_model(cfg, pretrained=False)` + `enable_internet: false` (070b).
4. Smoke local là CPU — **không bắt được lỗi device-mismatch CUDA**; trước push GPU, rà mọi `.to(device)` cho model kiểm tra.
5. Push khi kernel đang chạy/just-cancelled → **409**; chờ hoặc dùng slug mới.
6. `kernels status` dùng slug THEO TITLE (xem URL kernel), không phải `id` trong metadata.

## 4. Số liệu đích khi tái lập thành công

- Test F1 (macro): B0 ≈ 0.60 · B1 ≈ 0.953 · B2 ≈ 0.948 · B3 ≈ 0.960 · B4 ≈ 0.968 · **ASF ≈ 0.955** · CMP ≈ 0.952
- Chênh lệch ±0.3–0.5% giữa run là bình thường (seed, session I/O). Đối chiếu chi tiết: `notes/results_summary.md`.

## 5. Demo

Demo app (Streamlit/Gradio) — **chưa làm** (mục 14 spec, ngoài khối tài liệu này); checkpoint dùng được: `outputs/resisc45-final-*-3-seeds/ckpt_*.pt` + `class_order.json` từ kernel 010.
