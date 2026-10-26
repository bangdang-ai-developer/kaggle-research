# Dataset Card — NWPU-RESISC45

**Ngày audit đầy đủ:** 2026-10-26 — notebook `010-data-audit` v8 chạy trên Kaggle (`dangbang1/resisc45-data-audit-split-manifest` v8, COMPLETE), artifacts pull + verify độc lập local.

## Thông tin chung — ĐÃ XÁC MINH trên Kaggle

| Thuộc tính | Giá trị | Nguồn xác minh |
|---|---|---|
| Số ảnh | 31.500 | audit v8 + local |
| Số lớp × ảnh/lớp | 45 × 700 | audit v8 |
| Kích thước | 100% 256×256 RGB (0 ảnh lệch) | audit v8 |
| Decode lỗi | 0 | audit v8 |
| Exact dup (MD5) | 16 ảnh / 8 nhóm, cùng nhãn | audit v8 (pixel-check PSNR n/a — MD5 identical) |
| Near-dup candidate (dHash ≤6) | 20 cặp (13 cùng lớp / 7 khác lớp) | audit v8 — khớp local rerun độc lập |
| Mutual-NN (được gộp) | 16 cặp (12 cùng / 4 khác lớp) | audit v8 — khớp local |
| Nhóm đa lớp | 4 nhóm × 2 ảnh (airplane↔runway, beach↔runway, harbor↔runway, mobile_home_park↔rectangular_farmland) — pixel MSE 0.024–0.067 = chỉ tương tự cấu trúc, KHÔNG phải dup thật; vẫn co-split an toàn | audit v8 + pixel-check local |
| Nguồn Kaggle | `punfake/resisc45` (apache-2.0) | `kaggle datasets metadata` |
| GSD | 0.2–30 m/px (3 mức, không có metadata per-image) | paper arXiv:1703.00121 |

## Manifest chuẩn (duy nhất cho toàn project)

- File: `outputs/resisc45-data-audit-split-manifest/split_manifest.csv` — 31.500 dòng
- Cột: `image_id, class_name, class_index, dup_group, dup_kind (single/exact/near), needs_manual_verification, split`
- **SHA256:** `cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a`
- Split seed: 42 · Kernel Kaggle mount: `/kaggle/input/datasets/punfake/resisc45/NWPU-RESISC45` (layout 2026-10)
- Notebook sau đọc qua kernel_sources = `dangbang1/resisc45-data-audit-split-manifest`; KHÔNG sinh split mới.

## Phân bố split (thực đo)

- Tổng: train 23.626 / val 3.149 / test 4.725 (= 31.500 ✓)
- Per-class: train 524–526, val 69–70, test 104–107 (lệch ≤2 so với chuẩn 525/70/105 — do 4 nhóm đa lớp gán global theo deficit; chống leakage được ưu tiên theo protocol)
- Leakage: 0 dup_group trải >1 split (verify cả trong kernel lẫn local)

## Quy tắc split (đã áp)

1. Exact-dup (MD5): gộp nhóm tự động (đã xác minh).
2. Near-dup: chỉ gộp cặp **mutual nearest-neighbor** (hamming ≤6) — không nối chuỗi; candidate còn lại chỉ báo cáo (`near_dup_pairs.csv` + 2 PNG review).
3. Nhóm đa lớp: nguyên nhóm 1 split, gán global deficit-fill, per-class lệch được ghi thật.
4. Test locked: không dùng cho bất kỳ lựa chọn nào; chỉ đánh giá cuối sau `final_plan.json`.

