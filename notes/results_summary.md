# Results Summary — số liệu chính thức (đã verify từ artifacts)

## V4 (Hướng A, 2026-11-16..17) — Sửa `κ = exp(θ)` cho CMP (`CMP-exp`)

**Thiết lập:** Kế hoạch khóa [final_plan_v4_cmp_exp.json](final_plan_v4_cmp_exp.json); sửa `CMPResNet18` sang `cmp_scale_mode="exp"` (`κ = exp(θ) ∈ [10, 64]`) + `hardmax` + `k-means init`. Seed 0 từ `120-h1a-screen`, seeds 1–2 và test eval từ `121-cmp-exp-final-eval`. `ASF-Lite` (`0.9504` CE / `0.9555` KD, latency `4.07–4.28 ms`) dừng ở screening.

| Model | Val F1 (3-seed) | TEST Acc (3-seed) | TEST F1 (3-seed) | Latency ms (cùng máy 121) |
|---|---|---|---|---|
| **CMP-exp-CE** | **0.9544 ± 0.0010** | **0.9549 ± 0.0022** | **0.9548 ± 0.0021** | **3.69 ± 0.02** *(RN18: 4.11)* |
| **CMP-exp-KD** | **0.9557 ± 0.0005** | **0.9573 ± 0.0005** | **0.9574 ± 0.0005** | **3.71 ± 0.00** *(RN18: 4.11)* |

**Paired theo seed `{0, 1, 2}` trên TEST F1:**
- `CMP-exp-CE − CMP-CE v1`: `[+0.67%, +0.11%, +0.38%]` → mean **+0.39% (3/3 dương)**.
- `CMP-exp-CE − B1 ResNet18-CE`: `[+0.42%, −0.34%, +0.26%]` → mean **+0.11% (2/3 dương)** *(Val: +0.40%, 3/3 dương)*.
- `CMP-exp-KD − CMP-exp-CE`: `[+0.16%, +0.48%, +0.13%]` → mean **+0.26% (3/3 dương)**, `KD loss 15.21 → 0.508` (hết kẹt `18.46` ở v3).

---

## V3 — Knowledge Distillation từ ConvNeXt-T (nghiên cứu mở rộng, 2026-10-29)

**Setup:** teacher ConvNeXt-T s2 (val F1 0.9691, hash `9627ca43`, frozen); KD T=4, α=0.5; 15 epoch; students khởi tạo ImageNet pretrained; teacher chỉ trong training. Test đã quan sát v1/v2 → v3 là nghiên cứu mở rộng (công khai).

### TEST 3-seed (kernel 112c, `test_kd_3seed_report.json`)

| Model | TEST Acc | TEST F1 | Latency ms |
|---|---|---|---|
| **ResNet18-KD** | **0.9602 ± 0.0020** | **0.9601 ± 0.0019** | 2.53 |
| **ASF-KD** | 0.9591 ± 0.0014 | 0.9591 ± 0.0014 | 3.76 |
| B1 ResNet18 (CE) | 0.9537 ± 0.0019 | 0.9537 ± 0.0019 | 2.52 |
| ASF (CE) | 0.9520 ± 0.0026 | 0.9520 ± 0.0026 | 3.82 |
| B3 EffB0 | 0.9608 ± 0.0015 | 0.9607 ± 0.0015 | 7.40 |
| B4 ConvNeXt-T | 0.9678 ± 0.0002 | 0.9677 ± 0.0002 | 6.87 |

### Paired KD − CE trên test (F1, theo seed)

| So sánh | Diffs theo seed | Mean | Wins |
|---|---|---|---|
| RN18-KD − RN18-CE | [+0.58%, +0.55%, +0.81%] | **+0.65%** | **3/3** |
| ASF-KD − ASF-CE | [+0.91%, +0.72%, +0.48%] | **+0.70%** | **3/3** |
| ASF-KD − RN18-KD | (per-seed) | ≈ −0.10% | kiến trúc ASF ≈ RN18 dưới KD |

### Trả lời RQ v3

- **RQ1: KD CÓ cải thiện** — +0.65% (RN18) và +0.70% (ASF) test F1, nhất quán 3/3 seed, val cũng tăng cùng hướng (RN18-KD 0.9552±0.0003; ASF-KD 0.9576±0.0011 — val F1 cao nhất project).
- **RQ2: dưới KD, ASF ≈ RN18-KD trên test** (−0.10%); trên val ASF-KD vẫn +0.24%. Lợi ích kiến trúc ASF thu hẹp mạnh dưới KD — đóng góp chính của v3 là distillation, không phải kiến trúc.
- **RQ3:** chi phí KD chỉ ở training (teacher online, ~2× s/epoch); inference latency gần như không đổi (3.76 vs 3.82 ms ASF; 2.53 vs 2.52 RN18) — student độc lập teacher. Teacher training cost lịch sử ghi riêng 060a.
- CMP: không đưa KD final (KD loss không hội tụ tốt trên hard-max head — screening 110b, negative result).
- **Models tốt nhất nhóm nhỏ:** ResNet18-KD (0.9602) vượt B1-CE +0.65% và gần B3 EffB0 (0.9608) với latency 1/3; ASF-KD 0.9591. B4 vẫn trần (0.9678).



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

## 2. TEST chính thức

### 2a. Best-val seed (bản công bố ban đầu — `test_report.json` từ 070b)

| Model | seed | TEST Acc | TEST F1 (macro) | Latency ms (median) | Throughput imgs/s |
|---|---|---|---|---|---|
| B0 SmallCNN | 0 | 0.6008 | 0.5955 | 2.52 | 310 |
| B1 ResNet18 | 2 | 0.9530 | 0.9531 | 2.56 | 368 |
| B2 MobileNetV3-S | 0 | 0.9475 | 0.9476 | 4.60 | 384 |
| B3 EfficientNet-B0 | 0 | 0.9602 | 0.9601 | 7.36 | 355 |
| B4 ConvNeXt-T | 2 | 0.9678 | 0.9677 | 5.96 | 136 |
| ASF A2fix-ent | 2 | 0.9549 | 0.9550 | 3.72 | 348 |
| CMP C2 | 1 | 0.9515 | 0.9515 | 2.56 | 330 |

⚠️ Cảnh báo: bảng 2a chọn seed theo val → có selection effect; bảng 2b dưới đây là đánh giá công bằng hơn.

### 2b. TEST mở rộng đủ 3 seeds (`test_3seed_report.json` từ 071b, `final_plan_v2.json`)

| Model | TEST Acc mean ± std | TEST F1 mean ± std | Latency ms |
|---|---|---|---|
| B0 SmallCNN | 0.5946 ± 0.0065 | 0.5904 ± 0.0064 | 1.54 |
| B1 ResNet18 | **0.9537 ± 0.0019** | **0.9537 ± 0.0019** | 2.52 |
| B2 MobileNetV3-S | 0.9469 ± 0.0015 | 0.9469 ± 0.0015 | 4.91 |
| B3 EfficientNet-B0 | 0.9608 ± 0.0015 | 0.9607 ± 0.0015 | 7.40 |
| B4 ConvNeXt-T | 0.9678 ± 0.0002 | 0.9677 ± 0.0002 | 6.87 |
| ASF A2fix-ent | 0.9520 ± 0.0026 | 0.9520 ± 0.0026 | 3.82 |
| CMP C2 | 0.9510 ± 0.0012 | 0.9509 ± 0.0012 | 2.80 |

**Paired theo seed (F1):** ASF vs B1 = [−0.12%, −0.56%, +0.19%] mean **−0.16%** (1/3 seed dương); CMP vs B1 = [−0.25%, −0.45%, −0.12%] mean **−0.27%** (0/3).

**Phát hiện selection effect:** con số test 0.9550 của ASF (bảng 2a) là của seed-2 — seed tốt nhất theo val; khi lấy đủ 3 seed, ASF mean 0.9520 và **không thắng B1 trên test**. Lợi ích val (+0.39%, 3/3 seed) không khái quát sang test. Đây là kết quả trung thực của đánh giá mở rộng, cập nhật sau bản công bố 2a.

## 3. Trả lời tiêu chí đề tài

### ASF — ĐẠT trên val; trên TEST 3-seed ≈ B1 (không khái quát)
- Val: paired Δ = +0.24% / +0.38% / +0.54% theo seed — **3/3 dương**, mean +0.39% (paired t≈4.5, p≈0.05).
- **TEST 3-seed: 0.9520 ± 0.0026 vs B1 0.9537 ± 0.0019 — mean −0.16%, chỉ 1/3 seed dương → lợi ích val KHÔNG khái quát sang test.** Bản công bố best-val seed (0.9550 vs 0.9531, +0.19%) có selection effect — xem §2a/§2b.
- ASF vẫn thắng B2 rõ ràng trên cả val lẫn test 3-seed (+0.51%).
- Chi phí: +3.6% params (11.60M vs 11.20M), latency +52% (3.82 vs 2.52 ms).
- **Kết luận RQ1:** cải thiện val có thật (3/3 seed) nhưng quy mô +0.2–0.4% và không được xác nhận trên test — tuyên bố trung thực: "cải thiện nhỏ trên validation, không vượt baseline gốc trên test".

### CMP — CHƯA ĐẠT: dưới B1 trên cả val và test
- Val paired Δ = −0.05% (1/3 seed dương); **TEST 3-seed 0.9509 ± 0.0012 vs B1 0.9537 ± 0.0019 — paired 0/3 seed dương, mean −0.27%.**
- Nguyên nhân có bằng chứng cơ học: prototype collapse bền vững (inter-cos = 1.0 ở MỌI seed và mọi biến thể fix); confusion margin 0.3 gây hại, 0.1 trung tính.
- **Kết luận RQ2:** multi-prototype + confusion margin KHÔNG cải thiện trong cấu hình đã thử — negative result chính thức (5 cấu hình v1 + 2 cấu hình v2: hard-max phá collapse cơ học nhưng F1 vẫn không tăng).

**Bổ sung v3 (κ cap — đo trên 3.149 ảnh validation, `notes/v3b_kappa_check.json`):** CMP-KD không hội tụ có nguyên nhân tham số hóa xác định được. Logits CMP bị chặn κ·cos ∈ [−κ, κ] với κ học được 3.823 (< cap ln 100), T=4 → **trần soft-max student = 13.32%** cho bất kỳ lớp nào. Trong khi teacher soft targets (T=4) median max-prob 0.5985 (p90 0.74): **99.43% mẫu có teacher soft target vượt trần student** → student không thể khớp teacher targets, giải thích KD loss kẹt ~18.5. Đây là hạn chế tham số hóa (cosine bounded + κ thấp), không phải của KD hay prototype nói chung — nếu cải thiện CMP sau này: ưu tiên đổi tham số hóa scale (κ = exp(θ) kiểu ArcFace, scale 30–64) trước khi thêm loss/kiến trúc mới.

### RQ3 chi phí
- ASF: chi phí +3.6% params, +52% latency nhưng không có lợi ích test xác nhận → không tương xứng (theo 3-seed).
- CMP: cùng chi phí B1, không có lợi ích — không tương xứng.
- ConvNeXt-T: F1 cao nhất (0.9677 ± 0.0002) nhưng train đắt 3.8× B1, throughput thấp nhất.

## 4. Robustness (test acc, corruption synthetic — không chứng minh deployment)

| Model | clean | blur σ1 | blur σ2 | JPEG q30 | JPEG q60 | bright ×0.6 | ×1.4 |
|---|---|---|---|---|---|---|---|
| B1 ResNet18 | 0.9530 | 0.9310 | 0.7458 | 0.9346 | 0.9378 | 0.9335 | 0.9263 |
| **ASF** | 0.9549 | 0.9376 | 0.7780 | 0.9359 | 0.9407 | 0.9380 | 0.9283 |
| CMP | 0.9515 | 0.9249 | 0.7215 | 0.9319 | 0.9333 | 0.9274 | 0.9263 |
| B3 EffB0 | 0.9602 | 0.9458 | 0.6214 | 0.9416 | 0.9494 | 0.9494 | 0.9505 |
| B4 ConvNeXt-T | 0.9678 | 0.9608 | 0.8732 | 0.9568 | 0.9644 | 0.9657 | 0.9628 |

- ASF ≥ B1 ở 5/6 corruption theo delta (6/6 theo accuracy tuyệt đối); blur σ2: −17.7% vs −20.7%.
- EffB0 sụp mạnh nhất ở blur σ2 (−33.9 điểm phần trăm) — chỉ là quan sát, chưa có thí nghiệm xác định nguyên nhân; ConvNeXt-T bền nhất mọi loại corruption.

## 5. Ablation (val, 8 epoch, seed 0 — screening, không phải final)

- ASF: A1 fixed 0.9469 / A2 adaptive 0.9456 (gate collapse) → sau fix: A2fix-ent 0.9472 (gate sống).
- CMP: C1 0.9422 / C2 0.9441 / C3(+margin 0.3) 0.9404 / C2fix 0.9398 / C3fix 0.9413 — collapse bền vững.
- Chi tiết: `notes/experiment_registry.csv` + reports trong `outputs/`.

## 6. Bảng kiểm thắng/thua (mục 16.11 — cập nhật theo test 3-seed đầy đủ)

| Proposed | Thắng B1 (backbone gốc)? | Thắng baseline khác? | Bằng chứng |
|---|---|---|---|
| ASF A2fix-ent | **VAL: CÓ** (paired 3/3, +0.39%); **TEST: KHÔNG** (mean −0.16%, 1/3 seed) | B2: **CÓ** (test 3-seed +0.51%) | §1, §2b, §3 |
| CMP C2 | KHÔNG (val −0.05%; test −0.27%, 0/3 seed) | B2: **CÓ** (test 3-seed +0.40%) | §2b + collapse analysis |

Đề bài yêu cầu "thắng ít nhất một baseline": **cả hai ĐẠT** (thắng B2 rõ trên test 3-seed). Tuy nhiên kết luận trung thực đầy đủ: ASF chỉ thắng B1 trên validation, không giữ được trên test; CMP không thắng B1 ở bất kỳ tập nào.

## 7. Phần chưa hoàn thành / hạn chế (chi tiết limitations.md)

- CMP không thắng B1 — 5 cấu hình, collapse bền vững; hướng tiếp: hinge margin trực tiếp, orthogonal init, per-prototype gating.
- A3 (gate consistency) và mở rộng seed cho ablation chưa chạy — quota ưu tiên final.
- Không kiểm chứng dataset thứ hai (EuroSAT) — ngoài phạm vi thời gian.
- ±0.2% lợi ích ASF cần cẩn trọng tuyên bố; không có kiểm định thống kê ngoài paired-seed.
