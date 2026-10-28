# Limitations — hạn chế và những gì KHÔNG được tuyên bố

**Ngày:** 2026-10-28 · Đối chiếu `notes/results_summary.md` (số chính thức).

## 1. Về kết quả ASF (thắng B1)

1. **Độ lớn lợi ích nhỏ:** TEST +0.19% F1 (0.9550 vs 0.9531), val paired mean +0.39%. Nằm gần biên nhiễu seed (std ~0.002–0.004). Paired 3/3 seed dương + t≈4.5 (p≈0.05, df=2) hỗ trợ hướng đúng nhưng **chỉ 3 seed** — không khẳng định phổ quát.
2. **Chỉ trên 1 dataset:** mọi kết quả là NWPU-RESISC45. Không có EuroSAT/dataset thứ hai (ngoài phạm vi thời gian + quota).
3. **So sánh với baseline lớn:** ASF KHÔNG vượt B3 EfficientNet-B0 (0.9602) và B4 ConvNeXt-T (0.9678) — ASF thắng backbone gốc và B2, không phải SOTA tổng thể.
4. **Gate sau entropy reg phân bố gần đều** (max_mean 0.14–0.18 so đều 0.125): chưa có bằng chứng gate học "chọn theo nội dung ảnh" mạnh; chỉ chứng minh không sụp và model không tệ hơn. Cơ chế giải thích lợi ích +0.2% **chưa được chứng minh nhân quả** (gate distribution ≠ causality).
5. **Latency +45%** so B1 (3.72 vs 2.56 ms) — nếu ứng dụng biên độ cực hạn, chi phí phải được tính.

## 2. Về kết quả CMP (chưa thắng)

1. **Negative result chính thức:** 5 cấu hình (C1/C2/C3/C2fix/C3fix), tốt nhất 0.9441 (screen) — TEST 0.9515, hòa B1 (−0.15%), không thắng.
2. **Prototype collapse bền vững** (inter-cos = 1.0 mọi seed, mọi fix): k-means init + diversity 0.01 không đủ. Hướng tiếp (chưa chạy, đề xuất): diversity weight mạnh hơn hoặc hinge trực tiếp trên cặp prototype, orthogonal init, per-prototype gating.
3. **Confusion margin:** 0.3 gây hại rõ (−0.44%), 0.1 trung tính (−0.35%) — cơ chế weight-loss (phương án thứ 2 của spec) chưa được thử.

## 3. Về protocol và dữ liệu

1. **Duplicate gần nhau:** 4 nhóm cross-class được co-split dựa mutual-NN dHash — pixel-check cho thấy chúng chỉ "tương tự cấu trúc" (không trùng thật). Nếu sai, ảnh hưởng ≤8 ảnh/4725 test (≤0.17%).
2. **Không có metadata địa lý** — không tuyên bố generalization không gian.
3. **Robustness là synthetic corruption** (blur/JPEG/brightness trong phòng) — không chứng minh độ bền deployment ngoài đời.
4. **1 lần test duy nhất** đúng protocol; không có khoảng tin cậy test (chỉ có val qua 3 seed).
5. **Per-epoch history không được lưu vào artifact** của 060 (chỉ best + s/epoch); learning curves tái tạo một phần từ kernel log nếu cần — v1 thiết kế thiếu, đã ghi để sửa cho các dự án sau.
6. **A3 (gate consistency) và ablation đa seed** không chạy — quota ưu tiên final 3 seed.
7. **Không kiểm chứng dataset thứ hai** — đề xuất bước tiếp theo số 1 nếu làm tiếp.

## 4. Về triển khai

- Smoke local CPU không bắt được lỗi device-mismatch CUDA (đã gây 2 lần rerun) — cần smoke GPU ngắn trong tương lai.
- Kaggle quirks đã xử lý: kernelspec bắt buộc, mount path `/kaggle/input/datasets/<owner>/<slug>`, slug theo title, 409 sau cancel, FUSE I/O biến động (stage local bắt buộc).
