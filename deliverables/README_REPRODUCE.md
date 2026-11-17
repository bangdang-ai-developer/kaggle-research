# NWPU-RESISC45 — Hướng dẫn tái lập và cấu trúc gói nộp

Toàn bộ lệnh dưới đây chạy trực tiếp từ thư mục gốc sau khi giải nén `NWPU_RESISC45_submission.zip` (hoặc từ root repo `kaggle-research`). Mọi bảng số liệu báo cáo được sinh tự động từ `deliverables/results_master.json` bằng `scripts/build_results_master.py`.

## 1. Cấu trúc gói nộp (khớp 100% đường dẫn repo)

```text
├── README.md                    # Tóm tắt kết quả, bảng đối chiếu & quick start
├── pyproject.toml, uv.lock      # Dependencies khóa cho Python 3.12 (uv sync)
├── .python-version              # 3.12
├── src/models.py                # Nguồn logic duy nhất: models, KD, train, eval
├── demo/app.py                  # Streamlit offline demo (7 models)
├── scripts/                     # nbmake, build_results_master, validate_submission, demo_smoke, make_*, ...
├── tests/                       # Unit & checkpoint regression tests (pytest)
├── notebooks/<slug>/            # nb.py + kernel-metadata.json cho toàn bộ 22+ kernels
├── notes/                       # Plans, registry, diagnostics, limitations, figures/*.png
├── deliverables/
│   ├── Bao_cao_cuoi_mon_NWPU_RESISC45.docx
│   ├── Slides_cuoi_mon_NWPU_RESISC45.pptx
│   ├── Bao_cao_ket_qua_nghien_cuu.md
│   ├── results_master.json / .csv
│   ├── manifest.json            # SHA256 của 18 artifact nguồn (checkpoints + reports)
│   ├── demo_smoke.json          # Kết quả chạy smoke thực tế trên 7 checkpoints
│   └── README_REPRODUCE.md      # File này
└── outputs/                     # Kèm sẵn reports JSON/CSV nhẹ (010/060/070b/071b/110b/111/112c/120/121)
                                 # giúp validate_submission.py & build_results_master.py chạy ngay
```

## 2. Chạy kiểm tra ngay từ thư mục giải nén

```bash
uv sync                                        # cài Python 3.12 + dependencies (cần thêm streamlit nếu chạy UI)
uv run pytest                                  # chạy 22 unit tests
uv run python3 scripts/validate_submission.py  # kiểm tra toàn vẹn số liệu, seed, bảng và ZIP
uv run python3 scripts/build_results_master.py # tái tạo results_master từ reports trong outputs/
```

## 3. Tải các tệp lớn nằm ngoài ZIP (Checkpoints `.pt` & Dataset)

Để giữ `NWPU_RESISC45_submission.zip` nhẹ, các checkpoint `.pt` (~45–111 MB/file) và ảnh raw không đóng gói trong ZIP. Để chạy `demo/app.py` hoặc `scripts/demo_smoke.py`:

1. **Dataset NWPU-RESISC45** (`punfake/resisc45`, ~427 MB): tải vào `data/raw/punfake---resisc45/NWPU-RESISC45` qua `./scripts/fetch_data.sh dataset punfake/resisc45`.
2. **7 checkpoints dùng cho Demo** (lưu đúng đường dẫn tương đối dưới `outputs/`, SHA256 ghi trong `deliverables/manifest.json`):

| Lựa chọn trong `demo/app.py` | Kernel Kaggle (`dangbang1/...`) | Đường dẫn lưu cục bộ dưới `outputs/` |
|---|---|---|
| B1 ResNet18-CE s2 | `resisc45-final-baselines-b0-b4-3-seeds` | `resisc45-final-baselines-b0-b4-3-seeds/ckpt_resnet18_s2.pt` |
| B1 ResNet18-KD s0 | `resisc45-screen-kd-resnet18-asf-cmp-v2` | `resisc45-screen-kd-resnet18-asf-cmp-v2/ckpt_resnet18_kd.pt` |
| ASF-CE s1 | `resisc45-final-proposed-asf-cmp-3-seeds` | `resisc45-final-proposed-asf-cmp-3-seeds/ckpt_ASF_A2fix_ent_s1.pt` |
| ASF-KD s1 | `resisc45-kd-final-seeds-1-2` | `resisc45-kd-final-seeds-1-2/ckpt_asf_kd_s1.pt` |
| CMP-CE v1 s1 | `resisc45-final-proposed-asf-cmp-3-seeds` | `resisc45-final-proposed-asf-cmp-3-seeds/ckpt_CMP_C2_s1.pt` |
| CMP-exp-CE s1 | `resisc45-cmp-exp-3-seeds-and-test-eval` | `resisc45-cmp-exp-3-seeds-and-test-eval/ckpt_cmp_exp_ce_s1.pt` |
| CMP-exp-KD s0 | `resisc45-h1a-screen-cmp-exp-asf-lite` | `resisc45-h1a-screen-cmp-exp-asf-lite/ckpt_cmp_exp_kd.pt` |

*(Ghi chú: teacher checkpoint `ckpt_convnextt_s2.pt` nằm tại `outputs/resisc45-final-baselines-b0-b4-3-seeds/`, SHA256 `9627ca43...`, cũng có thể lấy từ cùng kernel).*

Kéo output từ Kaggle (yêu cầu tài khoản Kaggle & biến `KAGGLE_API_TOKEN` trong `.env` hoặc `~/.kaggle/access_token`):

```bash
./scripts/pull_output.sh dangbang1/resisc45-data-audit-split-manifest
./scripts/pull_output.sh dangbang1/resisc45-final-baselines-b0-b4-3-seeds
./scripts/pull_output.sh dangbang1/resisc45-final-proposed-asf-cmp-3-seeds
./scripts/pull_output.sh dangbang1/resisc45-screen-kd-resnet18-asf-cmp-v2
./scripts/pull_output.sh dangbang1/resisc45-kd-final-seeds-1-2
./scripts/pull_output.sh dangbang1/resisc45-h1a-screen-cmp-exp-asf-lite
./scripts/pull_output.sh dangbang1/resisc45-cmp-exp-3-seeds-and-test-eval
```

Sau khi có checkpoints, chạy demo & smoke test:

```bash
uv run python3 scripts/demo_smoke.py
uv run --with streamlit streamlit run demo/app.py
```

## 4. Thứ tự tái lập training trên Kaggle GPU (~18.5h Tesla T4)

| # | Thư mục `notebooks/` | Slug Kaggle (`dangbang1/...`) | Mục đích |
|---|---|---|---|
| 1 | `010-data-audit` | `resisc45-data-audit-split-manifest` | Audit + split manifest chuẩn (SHA `cc70020a…`) |
| 2 | `020-pilot-v2` | `resisc45-pilot-v2-local-stage` | Timing + sanity forward/backward |
| 3 | `030 → 040/040b → 050/050b` | `resisc45-screen-*` | Screening baselines + ablation + vòng fix |
| 4 | `060a + 060b` | `resisc45-final-*` | Final v1: 7 mô hình × 3 seed × 15 epoch |
| 5 | `070b`, `071b` | `resisc45-final-evaluation-v2-offline`, `resisc45-expanded-eval-3-seeds-v2-fixed` | Test + robustness (070b) & Test 3-seed CE (071b) |
| 6 | `080 + 090` | `resisc45-screen-*-v2/v3` | Screening v2 |
| 7 | `100b → 110b → 111 → 112c` | `resisc45-kd-*`, `resisc45-eval-kd-3-seeds-v2-fixed-sources` | KD pilot → screening → final → test 3-seed |
| 8 | `120-h1a-screen → 121-cmp-exp-final-eval` | `resisc45-h1a-screen-cmp-exp-asf-lite`, `resisc45-cmp-exp-3-seeds-and-test-eval` | V4 Hướng A: CMP-exp (`κ = exp(θ)`) screening → 3-seed + test |

Build notebook tự chứa từ nguồn `nb.py` trước khi push: `uv run python3 scripts/nbmake.py notebooks/<slug>`.
