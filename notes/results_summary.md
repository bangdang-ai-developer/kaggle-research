# Results Summary — số liệu chính thức (đã verify từ artifacts)

**Ngày:** 2026-10-28 · **Dataset:** NWPU-RESISC45 31.500 ảnh, split `cc70020a` (train 23.626 / val 3.149 / test 4.725)
**Protocol:** 15 epoch, early-stop 5, checkpoint val macro-F1, 3 seeds {0,1,2}; test chạy 1 lần duy nhất sau khi `final_plan.json` khóa.
**Nguồn:** `outputs/resisc45-final-baselines-b0-b4-3-seeds/`, `outputs/resisc45-final-proposed-asf-cmp-3-seeds/`, `outputs/resisc45-final-evaluation-v2-offline/`

## 1. Validation (3 seeds, mean ± std)

| Model | val F1 | val Acc | params (M) | train h/run |
|---|---|---|---|---|
| B0 SmallCNN | 0.5821 ± 0.0032 | 0.5889 ± 0.0046 | 0.29 | 0.34 |
| B1 ResNet18 | 0.9504 ± 0.0021 | 0.9504 ± 0.0021 | 11.20 | 0.20 |
| B2 MobileNetV3-S | 0.9424 ± 0.0010 | 0.9423 ± 0.0010 | 1.56 | 0.18 |
| B3 EfficientNet-B0 | 0.9580 ± 0.0005 | 0.9580 ± 0.0005 | 4.07 | 0.33 |
| B4 ConvNeXt-T | 0.9671 ± 0.0019 | 0.9672 ± 0.0019 | 27.85 | 0.62 |
| **ASF A2fix-ent** | **0.9543 ± 0.0036** | **0.9543 ± 0.0036** | 11.60 | 0.21 |
| CMP C2 | 0.9499 ± 0.0010 | 0.9501 ± 0.0010 | 11.49 | 0.20 |

## 2. TEST chính thức (4.725 ảnh, best-val seed — lưu trong test_report.json)

| Model | seed | TEST Acc | TEST F1 (macro) | Latency ms (median) | Throughput imgs/s |
|---|---|---|---|---|---|
| B0 SmallCNN | 0 | 0.6008 | 0.5955 | 2.52 | 310 |
| B1 ResNet18 | 2 | 0.9530 | 0.9531 | 2.56 | 368 |
| B2 MobileNetV3-S | 0 | 0.9475 | 0.9476 | 4.60 | 384 |
| B3 EfficientNet-B0 | 0 | 0.9602 | 0.9601 | 7.36 | 355 |
| B4 ConvNeXt-T | 2 | 0.9678 | 0.9677 | 5.96 | 136 |
| **ASF A2fix-ent** | 2 | **0.9549** | **0.9550** | 3.72 | 348 |
| CMP C2 | 1 | 0.9515 | 0.9515 | 2.56 | 330 |

## 3. Trả lời tiêu chí đề tài

### ASF — ĐẠT: thắng backbone gốc B1
- Val: paired Δ = +0.24% / +0.38% / +0.54% theo seed — **3/3 dương**, mean +0.39% (paired t≈4.5, p≈0.05).
- **TEST: 0.9550 vs B1 0.9531 (+0.19%)** — nhất quán với val. Đồng thời thắng B2 (+0.74%).
- Chi phí: +3.6% params (11.60M vs 11.20M), latency +45% (3.72 vs 2.56 ms) — rẻ hơn B3/B4 đáng kể về thời gian huấn luyện (0.21h vs 0.33h/0.62h).
- Gate hoạt động thật (không collapse sau entropy reg; max_mean 0.14–0.18/seed, phân hóa theo ảnh).
- **Kết luận RQ1 (trong phạm vi thí nghiệm):** gate entropy + wavelet branch cho +0.2% F1 với chi phí tham số/nhiễu nhỏ; LỢI ÍCH NHỎ — cần cẩn trọng tuyên bố (±0.2% gần biên nhiễu, nhưng paired 3/3 seed dương hỗ trợ hướng đúng).

### CMP — CHƯA ĐẠT: hòa B1
- Val paired Δ = −0.05% (1/3 seed dương); TEST 0.9515 vs 0.9531 (−0.15%).
- Nguyên nhân có bằng chứng cơ học: prototype collapse bền vững (inter-cos = 1.0 ở MỌI seed và mọi biến thể fix); confusion margin 0.3 gây hại, 0.1 trung tính.
- **Kết luận RQ2:** multi-prototype + confusion margin KHÔNG cải thiện trong cấu hình đã thử — negative result chính thức (5 cấu hình đã thử: C1/C2/C3/C2fix/C3fix).

### RQ3 chi phí
- ASF: +0.19% test F1 đổi +3.6% params, +45% latency, +5% train time — tương xứng ở mức ứng dụng nhẹ; không cạnh tranh EffB0/B4 về F1 thuần.
- CMP: cùng chi phí B1, không có lợi ích — không tương xứng.
- ConvNeXt-T: F1 cao nhất (0.9677) nhưng train đắt 3.8× B1, throughput thấp nhất 136 imgs/s.

## 4. Robustness (test acc, corruption synthetic — không chứng minh deployment)

| Model | clean | blur σ1 | blur σ2 | JPEG q30 | JPEG q60 | bright ×0.6 | ×1.4 |
|---|---|---|---|---|---|---|---|
| B1 ResNet18 | 0.9530 | 0.9310 | 0.7458 | 0.9346 | 0.9378 | 0.9335 | 0.9263 |
| **ASF** | 0.9549 | 0.9376 | 0.7780 | 0.9359 | 0.9407 | 0.9380 | 0.9283 |
| CMP | 0.9515 | 0.9249 | 0.7215 | 0.9319 | 0.9333 | 0.9274 | 0.9263 |
| B3 EffB0 | 0.9602 | 0.9458 | 0.6214 | 0.9416 | 0.9494 | 0.9494 | 0.9505 |
| B4 ConvNeXt-T | 0.9678 | 0.9608 | 0.8732 | 0.9568 | 0.9644 | 0.9657 | 0.9628 |

- ASF ≥ B1 ở 5/6 corruption (blur σ2: −17.7% vs −20.7%) — tín hiệu nhỏ cùng hướng wavelet.
- EffB0 sụp mạnh ở blur σ2 (−34%) — stride lớn nhạy motion blur; ConvNeXt-T robust nhất mọi loại.

## 5. Ablation (val, 8 epoch, seed 0 — screening, không phải final)

- ASF: A1 fixed 0.9469 / A2 adaptive 0.9456 (gate collapse) → sau fix: A2fix-ent 0.9472 (gate sống).
- CMP: C1 0.9422 / C2 0.9441 / C3(+margin 0.3) 0.9404 / C2fix 0.9398 / C3fix 0.9413 — collapse bền vững.
- Chi tiết: `notes/experiment_registry.csv` + reports trong `outputs/`.

## 6. Bảng kiểm thắng/thua (mục 16.11)

| Proposed | Thắng B1 (backbone gốc)? | Thắng baseline khác? | Bằng chứng |
|---|---|---|---|
| ASF A2fix-ent | **CÓ** — TEST +0.19%, val paired 3/3 seed | B2 (+0.74% test) | §2 + paired seeds |
| CMP C2 | KHÔNG (−0.15% test, hòa thống kê) | B2 (+0.39% test) | §2 + collapse analysis |

## 7. Phần chưa hoàn thành / hạn chế (chi tiết limitations.md)

- CMP không thắng B1 — 5 cấu hình, collapse bền vững; hướng tiếp: hinge margin trực tiếp, orthogonal init, per-prototype gating.
- A3 (gate consistency) và mở rộng seed cho ablation chưa chạy — quota ưu tiên final.
- Không kiểm chứng dataset thứ hai (EuroSAT) — ngoài phạm vi thời gian.
- ±0.2% lợi ích ASF cần cẩn trọng tuyên bố; không có kiểm định thống kê ngoài paired-seed.
