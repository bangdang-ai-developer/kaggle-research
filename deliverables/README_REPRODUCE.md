# NWPU-RESISC45 — Hướng dẫn tái lập (tóm tắt)

Mọi chi tiết đầy đủ: `notes/reproduction.md` trong repo. Mọi số trong báo cáo/slides được sinh tự động
từ `deliverables/results_master.json` (xây từ artifacts checkpoint/report bằng `scripts/build_results_master.py`).

## Cấu trúc gói nộp

```
deliverables/
├── Bao_cao_cuoi_mon_NWPU_RESISC45.docx    # báo cáo đầy đủ (19 mục)
├── Slides_cuoi_mon_NWPU_RESISC45.pptx     # 19 slides + speaker notes
├── results_master.json / .csv             # bảng số liệu chuẩn (nguồn duy nhất)
├── demo_smoke.json                        # bằng chứng kiểm tra demo
└── README_REPRODUCE.md                    # file này
```

Ngoài ZIP (nằm trong repo `kaggle-research`):
- `src/models.py` — toàn bộ logic model/train/eval/KD (nguồn duy nhất)
- `notebooks/<slug>/nb.py` — nguồn các notebook Kaggle (build bằng `scripts/nbmake.py`)
- `outputs/` — artifacts đã pull (checkpoint, report JSON, log, confusion matrix, predictions)
- `notes/` — registry mọi run, diagnostics v2/v3, plans, limitations, changelog

## Thứ tự tái lập trên Kaggle (tài khoản GPU, ~16h tổng qua 3 vòng)

| # | Kernel (thư mục `notebooks/`) | Slug Kaggle | Mục đích |
|---|---|---|---|
| 1 | 010-data-audit | resisc45-data-audit-split-manifest | audit + split manifest chuẩn (sha `cc70020a…`) |
| 2 | 020-pilot-v2 | resisc45-pilot-v2-local-stage | timing + sanity |
| 3 | 030 → 040/040b → 050/050b | resisc45-screen-* | baseline + ablation + vòng fix |
| 4 | 060a + 060b | resisc45-final-* | final v1: 7 mô hình × 3 seed |
| 5 | 070b | resisc45-final-evaluation-v2-offline | test + robustness v1 |
| 6 | 080 + 090 | resisc45-screen-*-v2/v3 | screening v2 |
| 7 | 100b → 110b → 111 | resisc45-kd-* | KD pilot → screening → final |
| 8 | 112c | resisc45-eval-kd-3-seeds-v2-fixed-sources | test eval KD 3-seed |
| 9 | 120 → 121 | resisc45-h1a-screen-* / resisc45-cmp-exp-3-seeds-and-test-eval | v4 Hướng A: CMP κ=exp screening → 3-seed + test eval |

Quy tắc: kernel chạy theo thứ tự phụ thuộc (kernel sau attach output kernel trước qua `kernel_sources`);
mọi kernel huấn luyện tự stage dataset sang disk local; checkpoint chọn theo validation macro-F1; test chỉ
đánh giá theo kế hoạch đã khóa (`notes/final_plan*.json`).

## Demo local

```bash
uv sync
uv run streamlit run demo/app.py
```

Yêu cầu `outputs/` chứa checkpoint (đã pull từ Kaggle như bảng trên).
