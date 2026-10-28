# Compute Budget — CHỐT TỪ SỐ ĐO THẬT (pilot v2, T4, 2026-10-26)

## ĐỐI CHIẾU THỰC TẾ (2026-10-28, sau khi hoàn tất toàn bộ thí nghiệm)

| Hạng mục | Dự toán | Thực tế |
|---|---|---|
| Pilot (020 v1 hủy + manual run) | 1.0 | ~1.2 |
| Screening 030/040/050 + fix 040b/050b | 2.15 | ~2.55 (thêm vòng fix duyệt) |
| Final 060a baselines ×3 seed | 4.7 | 5.0 |
| Final 060b proposed ×3 seed | 0.9 | 1.4 |
| Eval 070 (v1 fail infra + 070b) | 0.5 | 0.7 |
| Stage/boot/overhead | 0.3 | ~0.5 |
| **Tổng GPU** | **~9.7** | **~11.3** |
| CPU (010 audit ×8 phiên) | 0 | ~3h (miễn phí quota) |

Vượt dự toán ~15% do 1 vòng screening-fix duyệt thêm + 2 kernel fail hạ tầng. **Tổng vẫn dưới trần 30h/tuần** (khoảng 37% quota tuần).

**Trần:** Kaggle GPU quota ~30h/tuần. **Số liệu nguồn:** `outputs/resisc45-pilot-v2-local-stage/pilot_report.json` (manual draft run trên T4, stage local dataset, split `cc70020a`).

## Số đo thật (T4, batch 128, AMP, staged local)

| Thành phần | Giá trị đo |
|---|---|
| ResNet18 epoch (23.625 ảnh) | **42–44s** + val eval 4s |
| ASF subset epoch (4.725 ảnh) | 12.2s → full ≈ 52s/epoch (+22% so RN18) |
| CMP subset epoch | 11.1s → full ≈ 50s/epoch |
| Peak GPU mem | 2.19–2.63 GiB (nhiều headroom trên T4 15GiB) |
| Val eval (3.149 ảnh) | 4.0s |
| Stage dataset local (26.775 ảnh, 16 threads) | 18–105s tùy session |
| Checkpoint | 42.8 MB/model |
| Latency batch-1 | smallcnn 1.6 / RN18 2.6 / CMP 2.6 / ASF 4.7 / MBV3 5.1 / ConvNeXt-T 6.1 / EffB0 7.8 ms |
| Confusion graph probe (full, ngoại suy) | ≈ 4–5 phút/seed (6 epoch trên 21.3k fit + predict 1.05k) |
| Sanity chất lượng | B1 val acc 91.1% / F1 91.2% sau 2 epoch — hợp lý với RESISC45 |

## Kế hoạch chi (giờ GPU, ước tính từ số đo trên; epoch = train+val)

| Hạng mục | Tính | Giờ |
|---|---|---|
| Screening 030: B0–B4, 8 epoch, 1 seed | 8×(30+39+35+94+134)s ≈ 43m | 0.7 |
| Screening 040: ASF A1/A2/spatial-control, 8 epoch | 3×8×52s ≈ 21m | 0.35 |
| Screening 050: CMP C1/C2/C3 (+probe C3), 8 epoch | ≈ 20m + 5m | 0.45 |
| Final 060: 5 baseline × 3 seed × 15 epoch | 3×15×(30+39+35+94+134)s ≈ 4.7h | 4.7 |
| Final 060: ASF ×3 seed | 3×15×52s ≈ 39m (+A3 nếu đủ) | 0.75 |
| Final 060: CMP ×3 seed + graph probe mỗi seed | 37m + 15m | 0.9 |
| Eval 070: test metrics + latency + robustness | | 0.5 |
| Stage + boot 5–6 kernel | | 0.3 |
| **Tổng mục tiêu** | | **≈ 8.7h** |
| Dự phòng 25% (rerun, lỗi) | | 2.2h |
| **Trần chi tiêu** | | **≈ 11h / 30h** ✓ |

Quota dư ~19h → có thể mở rộng: A3 (gate consistency) + C4 (diversity) ×3 seed (+~1.5h), seed thứ 3 cho ablation (+~1h), robustness mở rộng. Quyết định sau khi xem kết quả screening.

## Quy tắc cắt giảm (nếu phát sinh lỗi chuỗi)

1. Bỏ A3/C4 trước → 2. B4 ConvNeXt-T 3 seed → 2 seed (ghi rõ) → 3. KHÔNG cắt: 3 seed của ASF/CMP, đối chứng trực tiếp, final test.

## Vận hành (bài học từ pilot)

- **BẮT BUỘC stage dataset local** đầu mỗi kernel huấn luyện (`stage_dataset()` trong src/models.py) — FUSE trực tiếp 17 ảnh/s làm GPU 0%.
- Mỗi kernel có smoke mode + `[train]`/`[stage]` progress log; kiểm tra model-on-device trước khi forward (CPU smoke không bắt được CUDA mismatch).
- Kernel `complete` ≠ thành công; poll 5–10 phút; push đè khi đang chạy → 409, dùng kernel mới nếu cần.
- run_id trong report ghi "020-pilot-v1" (config không đổi nhãn) — ánh xạ thật: kernel `resisc45-pilot-v2-local-stage`, draft run sau ver3 fix (commit a2c0998).
