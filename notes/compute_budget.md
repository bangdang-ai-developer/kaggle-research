# Compute Budget — dự toán trước pilot

**Trần:** Kaggle GPU quota ~30h/tuần (T4 ×2 có sẵn nhưng 1 kernel dùng 1 GPU). **Chưa kiểm tra số dư thực tế** (Kaggle không có API công khai đọc GPU quota — kiểm tra bằng `skill status` khi có competition hoặc user xem web UI). Mọi số dưới đây là **ước tính chờ đo thật ở pilot**.

## Tham số ước tính

- Train 23.625 ảnh (525×45), 256×256, batch 128, AMP, T4.
- Ước 45–90 s/epoch với ResNet18-size (đo thật ở pilot 020 trước khi chốt).

## Kế hoạch chi tiêu (đơn vị: giờ GPU, ước tính)

| Hạng mục | Kernel | Runs | Ước/run | Tổng |
|---|---|---|---|---|
| Data audit + pilot + đo thời gian | 010 + 020 | 2 | 0.5 | 1.0 |
| Baseline screening (B0–B4, 8 epoch, 1 seed) | 030 | 5 | 0.3 | 1.5 |
| ASF ablation screening (A1, A2, ± spatial-branch control, 8 epoch) | 040 | 3–4 | 0.3 | 1.2 |
| CMP ablation screening (C1, C2, C3, 8 epoch + probe graph) | 050 | 3–4 | 0.4 | 1.6 |
| Final: 5 baseline × 3 seed × 15 epoch | 060 | 15 | 0.6 | 9.0 |
| Final: 2 proposed × 3 seed × 15 epoch (+ graph CMP) | 060 | 6 | 0.7 | 4.2 |
| Final eval: test metrics + latency + robustness | 070 | 1–2 | 0.5 | 1.0 |
| Dự phòng lỗi/rerun | — | — | — | ~5.4 |
| **Tổng dự toán** | | | | **≈ 25h** |

Giữ 15–20% dự phòng → **ngân sách mục tiêu ≤ 25h/tuần**, trong trần 30h.

## Quy tắc cắt giảm nếu vượt (theo thứ tự, phải ghi log)

1. Giảm ablation: bỏ A3/C4 trước.
2. Baseline mạnh nhất (B4 ConvNeXt-T) 3 seed → 2 seed (ghi rõ).
3. Screening subset 50% train.
4. KHÔNG cắt: seed 3 của 2 proposed, đối chứng trực tiếp A0/C0/C1, final test, integrity checks.

## Nguyên tắc vận hành

- Mỗi kernel có giới hạn epoch/runtime + smoke mode trước.
- Poll status chu kỳ ≥ 5 phút; không rerun tự động vô hạn.
- Mỗi run có `run_id`; ghi `experiment_registry.csv` ngay khi submit.
- Kernel `complete` ≠ experiment thành công: phải parse metrics + artifact.
