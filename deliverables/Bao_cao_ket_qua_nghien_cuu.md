# Báo cáo kết quả nghiên cứu chi tiết — NWPU-RESISC45: ASF, CMP và Knowledge Distillation

**Dự án:** Phân loại cảnh ảnh viễn thám bằng lựa chọn đặc trưng đa tỉ lệ, học phân biệt các lớp dễ nhầm lẫn và knowledge distillation
**Thời gian thực hiện:** 26/10 – 17/11/2026 · **Nền tảng:** Kaggle Tesla T4, ~18.5 giờ GPU (measured) + ~3 giờ CPU
**Trạng thái:** hoàn thành 4 vòng nghiên cứu (v1/v2/v3/v4 Hướng A), 22 kernel Kaggle, có artifact đầy đủ
**Phạm vi tài liệu:** tổng kết chi tiết toàn bộ kết quả đã đo, phân tích và kết luận kèm bằng chứng truy vết. Số liệu đồng bộ với `deliverables/results_master.json` (sinh tự động từ artifacts).

---

## 1. Tổng quan dự án

### 1.1. Bài toán
Phân loại 45 cảnh ảnh viễn thám trên bộ NWPU-RESISC45 — bài toán có nhiều cặp lớp chỉ phân biệt được bằng bố cục không gian chi tiết (church↔palace, circular↔rectangular farmland, dense↔medium residential, wetland↔lake).

### 1.2. Hai kiến trúc đề xuất (gắn trên cùng backbone ResNet18 ImageNet-pretrained)

**ASF (Adaptive Scale–Frequency ResNet18):** giữ nguyên backbone; tách Haar wavelet mức 1 trên feature maps layer2 (128ch, 32×32) và layer4 (512ch, 8×8) tạo 8 thành phần (mỗi stage: nhánh ORIG — feature map gốc không qua DWT + 3 subband LH/HL/HH); mỗi thành phần qua 1×1 conv → BN → ReLU → GAP cho vector f_i (128-d); gate softmax theo từng ảnh (MLP 1024→8) tổng hợp Σ w_i·f_i; hợp nhất residual qua Linear khởi tạo nhỏ (std 1e-3) cộng vào đặc trưng GAP(layer4) trước head 45 lớp. Entropy regularizer λ·(log 8 − H(w))/log 8, λ = 0.05 giữ gate khỏi sụp 1-hot.

**CMP-exp (V4 — cấu hình chính bảng kết quả):** GAP → embedding Linear 512-d → L2 normalize; `K = 2` prototype/lớp khởi tạo bằng k-means trên tập train; `hard-max` cosine similarity; scale mũ `κ = exp(θ)` khởi tạo ở `10.0` (`θ = ln 10`) và chặn trên tại `64.0`. Các thành phần confusion graph (xây train-only 90/10, 6 epoch), margin và pair loss được khảo sát trong phần ablation (`C3`/`CMP-PG`) và không dùng trong V4 chính.

### 1.3. Bốn vòng nghiên cứu
- **V1:** audit dữ liệu + protocol chống rò rỉ + final 7 mô hình (5 baseline + 2 proposed) × 3 seed + test + robustness (1 checkpoint best-val/model).
- **V2:** chẩn đoán bằng số trên checkpoint thật, 4 candidate sửa lỗi cơ học, mở rộng đánh giá test đủ 3 seed × 7 mô hình (phát hiện selection effect).
- **V3:** nghiên cứu mở rộng Knowledge Distillation từ ConvNeXt-T cho 3 student, final 2 mô hình KD × seeds 1–2, test eval 3-seed.
- **V4 (Hướng A):** cấu hình `CMP-exp` (`κ = exp(θ) ∈ [10, 64]` + `hardmax` + `k-means init`), screening `ASF-Lite` (dừng), và đánh giá `CMP-exp-CE`/`CMP-exp-KD` đủ 3 seed + TEST.

### 1.4. Đóng góp chính (kết quả cuối cùng trên TEST, 3 seeds)
1. **Hai mô hình tự chế ở chế độ CE chuẩn đáp ứng yêu cầu vượt baseline:** **`ASF-CE` (95.20% Acc / 95.20% macro-F1)** vượt `B2 MobileNetV3-S` (`94.69%`) `+0.51` điểm phần trăm (chưa vượt `B1 ResNet18-CE` `95.37%` trên test, dù thắng `3/3` seed trên validation `+0.39` điểm); **`CMP-exp-CE` (95.49% Acc / 95.48% macro-F1)** vượt `B2` `+0.79` điểm, vượt `CMP-CE (v1)` `+0.39` điểm (`3/3` seed) và vượt nhẹ `B1` trung bình `+0.11` điểm (dương ở `2/3` seed).
2. **Mở rộng Knowledge Distillation (v3/v4):** `ResNet18-KD` (đối chứng KD trên backbone chuẩn) đạt **96.02% Acc / 96.01% macro-F1** (`+0.65` điểm F1 vs `B1`, `3/3` seed); `ASF-KD` đạt **95.91% Acc / 95.91% macro-F1** (`+0.70` điểm F1 vs `ASF-CE`, `3/3` seed); `CMP-exp-KD` đạt **95.73% Acc / 95.74% macro-F1** (`+0.26` điểm F1 vs `CMP-exp-CE`, `3/3` seed, và tăng `94.48% → 95.61%` trên **validation seed 0** so với `CMP-KD` cũ).
3. Quy trình chẩn đoán bằng số trên checkpoint phát hiện đúng các lỗi cơ học (`gate collapse`, `margin` ngược chiều, `κ`-cap ở `CMP-KD` cũ với `κ = 3.823` tạo trần softmax `13.32%` ở `T = 4`) để sửa dứt điểm ở V4.

---

## 2. Dataset và protocol

### 2.1. Audit dữ liệu
Nguồn Kaggle `punfake/resisc45`. Kiểm toàn bộ 31.500 ảnh (`010-data-audit`): 0 lỗi decode, 100% 256×256 RGB, đủ 45 lớp × 700 ảnh. Trùng lặp (`audit_report.json`, `near_dup_pairs.csv`): **8 nhóm trùng tuyệt đối** (16 file MD5 identical, cùng lớp) và **20 cặp gần trùng dHash** (Hamming ≤ 6: 13 cùng lớp / 7 khác lớp), trong đó **16 cặp là mutual nearest-neighbor** (12 cùng lớp / 4 khác lớp) được gộp vào nhóm; 4 nhóm chéo lớp được co-split để loại rò rỉ (kiểm tra pixel cho thấy chỉ tương tự cấu trúc — MSE 0.02–0.07 — không phải trùng thật).

### 2.2. Split
75/10/15 phân tầng, seed 42, chia theo nhóm trùng: **train 23.626 / val 3.149 / test 4.725**; per-class 524–526 / 69–70 / 104–107. Manifest SHA256 `cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a` — không chia lại ở bất kỳ vòng nào.

### 2.3. Protocol huấn luyện (đồng nhất mọi mô hình/mọi vòng)
Tối đa 15 epoch, early-stop patience 5 trên validation macro-F1; AdamW (lr backbone 1e-4, head 1e-3, weight decay 1e-4, cosine + warmup 5%); batch 128; AMP; augmentation train: RandomResizedCrop(0.7–1.0) + hflip + vflip; eval: resize 256 + chuẩn hóa ImageNet. 3 seeds {0,1,2}. Dataset được copy sang disk local trước training (loại bottleneck network mount: 17–35 ảnh/s → local 1.500+ ảnh/s).

### 2.4. Quy tắc test
Test chỉ đánh giá khi kế hoạch khóa (`final_plan*.json`); checkpoint chọn theo validation; v1 test một lần (best-val seed); v2 mở rộng đủ 3 seed theo kế hoạch cố định; v3 công khai là nghiên cứu mở rộng vì test đã quan sát.

---

## 3. Kết quả vòng V1 (CE thuần)

### 3.1. Validation macro-F1 (3 seeds)

| Model | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| B0 SmallCNN | 0.5854 | 0.5790 | 0.5819 | 0.5821 ± 0.0032 |
| B1 ResNet18-CE | 0.9483 | 0.9505 | 0.9524 | 0.9504 ± 0.0021 |
| B2 MobileNetV3-S | 0.9436 | 0.9420 | 0.9417 | 0.9424 ± 0.0010 |
| B3 EfficientNet-B0 | 0.9584 | 0.9581 | 0.9574 | 0.9580 ± 0.0005 |
| B4 ConvNeXt-T | 0.9669 | 0.9654 | 0.9691 | 0.9671 ± 0.0019 |
| ASF A2fix-ent | 0.9507 | 0.9543 | 0.9578 | 0.9543 ± 0.0036 |
| CMP C2 | 0.9503 | 0.9507 | 0.9488 | 0.9499 ± 0.0010 |

### 3.2. Test best-val seed (bản công bố đầu — có selection effect, xem §5)

| Model | seed | Acc | F1 |
|---|---|---|---|
| B1 | 2 | 0.9530 | 0.9531 |
| ASF | 2 | 0.9549 | 0.9550 |
| CMP | 1 | 0.9515 | 0.9515 |

### 3.3. V2 mở rộng: TEST đủ 3 seeds — phát hiện selection effect

| Model | Acc mean ± std | F1 mean ± std | F1 per seed |
|---|---|---|---|
| B1 | 0.9537 ± 0.0019 | 0.9537 ± 0.0019 | 0.9521 / 0.9558 / 0.9531 |
| ASF | 0.9520 ± 0.0026 | 0.9520 ± 0.0026 | 0.9509 / 0.9502 / 0.9550 |
| CMP | 0.9510 ± 0.0012 | 0.9509 ± 0.0012 | 0.9496 / 0.9513 / 0.9519 |

> **Sửa lỗi (v4):** cột per-seed trong bản trước viết tay sai (không khớp mean±std/paired diffs — mean±std và paired diffs luôn đúng vì sinh từ artifacts). Bảng trên lấy trực tiếp từ `test_3seed_report.json`.

**Paired F1 theo seed (test):** ASF − B1 = [−0.12, −0.56, +0.19] điểm phần trăm → mean **−0.16** (1/3 seed dương); CMP − B1 = [−0.25, −0.45, −0.12] → mean **−0.27** (0/3).

**Selection effect:** trên validation, ASF > B1 paired 3/3 seed (+0.24/+0.38/+0.54 điểm, mean +0.39, t≈4.5, p≈0.05). Trên test đủ seed, khoảng cách đảo dấu (−0.16). Con số test +0.19% của bản công bố đầu đo bằng seed tốt nhất theo val và không khái quát. Kết luận được điều chỉnh: **ASF cải thiện validation nhất quán nhưng xấp xỉ B1 trên test.** Cả hai proposed vẫn vượt B2 trên test 3-seed (ASF +0.51, CMP +0.40 điểm).

### 3.4. Robustness v1 (test accuracy, corruption tổng hợp — single best-val checkpoint)

| Model | clean | blur σ1 | blur σ2 | JPEG q30 | JPEG q60 | bright ×0.6 | ×1.4 |
|---|---|---|---|---|---|---|---|
| B1 (seed 2) | 0.9530 | 0.9310 | 0.7458 | 0.9346 | 0.9378 | 0.9335 | 0.9263 |
| ASF (seed 2) | 0.9549 | 0.9376 | 0.7780 | 0.9359 | 0.9407 | 0.9380 | 0.9283 |
| CMP (seed 1) | 0.9515 | 0.9249 | 0.7215 | 0.9319 | 0.9333 | 0.9274 | 0.9263 |
| B3 (seed 0) | 0.9602 | 0.9458 | 0.6214 | 0.9416 | 0.9494 | 0.9494 | 0.9505 |
| B4 (seed 2) | 0.9678 | 0.9608 | 0.8732 | 0.9568 | 0.9644 | 0.9657 | 0.9628 |

`ASF-CE` (seed 2) có accuracy tuyệt đối cao hơn `B1` (seed 2) ở **6/6** corruption, và mức giảm accuracy (`Δ vs clean`) nhỏ hơn ở **5/6** corruption (trừ `JPEG q30`: `-1.90` điểm vs `-1.84` điểm của `B1`); ConvNeXt-T bền nhất; EfficientNet-B0 giảm mạnh nhất ở blur σ2 (`-33.88` điểm phần trăm — nguyên nhân chưa có thí nghiệm chứng minh).

---

## 4. Screening (8 epoch, seed 0 — không phải final benchmark)

### 4.1. ASF ablation (v1 + vòng fix v2)

| Cấu hình | val F1 | Gate diagnostics |
|---|---|---|
| A1 fixed fusion (zero-init) | 0.9469 | — |
| A2 adaptive (zero-init) | 0.9456 | **sụp 1-hot LL4: max_mean 1.0, std 0** |
| A2-conv (detail học được, control) | 0.9457 | sụp tương tự → không phải lỗi Haar |
| A2fix-small (init std 1e-3) | 0.9453 | **vẫn sụp** |
| **A2fix-ent (init 1e-3 + entropy 0.05)** | **0.9472** | **sống: max_mean 0.16, std > 0** |

Kết luận: entropy regularizer là nhân tố quyết định để nhánh gate tồn tại; energy descriptor (v2) có căn cứ từ diagnostics nhưng không vượt GAP-gate ở vòng v3.

### 4.2. CMP ablation (v1 + vòng fix v2)

| Cấu hình | val F1 | inter-prototype cos |
|---|---|---|
| C1 cosine K=1 | 0.9422 | — |
| C2 K=2 (LSE) | 0.9441 | **1.0000** |
| C3 K=2 + margin 0.3 | 0.9404 | 1.0000 |
| C2fix k-means + diversity 0.01 | 0.9398 | 1.0000 |
| C3fix margin 0.1 | 0.9413 | 1.0000 |
| CMP-P hard-max + k-means (v2) | 0.9394 | 0.7254 |
| CMP-PG + pair loss đúng chiều (v2) | 0.9416 | 0.7373 |

---

## 5. Vòng V2 — diagnostics và kết quả

### 5.1. Diagnostics bằng số (script `scripts/v2_diagnostics.py`, FP32, trên checkpoint thật + synthetic)

| ID | Nghi vấn | Kết quả | Hệ quả |
|---|---|---|---|
| D1 | Margin v1 tác động gì? | **Sai chiều**: CE mẫu khó giảm 0.714→0.604; gradient lớp khó 0.401→0.331 | Đó là down-weight mẫu khó, không phải discriminative margin |
| D2 | Collapse tái lập được? | Synthetic LSE/hard-max (κ=10, embedding cố định): **KHÔNG** — prototype tách nhau (−0.85…−0.92) | Cơ chế đầy đủ chưa xác định (nghi parallel-drift qua LSE murette + embedding chuyển động) — ghi minh bạch |
| D3 | Checkpoint thật? | **CÓ**: 45/45 lớp pair-cos 1.00000000, euclid 0.00035; cross-class −0.023 (metric đúng) | Collapse là hiện tượng thật, không phải lỗi đo |
| D4a | "Gate tắt ≡ ResNet18"? | **BÁC BỎ**: residual/backbone norm = 0.37; max\|Δlogit\| = 8.50 | Sửa nhận định v1 |
| D4b | Signed-GAP triệt tiêu? | Đo TRƯỚC projection trên subband thô: \|E[comp]\| LL4 1.025 vs LH4 0.054 (19×); RMS 1.426 vs 0.328 (4.3×) | Chỉ là thống kê hệ số gợi ý khảo sát energy descriptor — chưa đo tín hiệu sau proj/ReLU tại điểm head; thử nghiệm ASF-E không cải thiện cấu hình được chọn |

### 5.2. Screening v2 kết quả
- **CMP-P hard-max + k-means**: 0.9394; **CMP-PG + pair loss đúng chiều**: 0.9416. Hard-max phá collapse cơ học (inter-cos 1.0 → 0.73/0.74; pair loss giảm đều 0.457 → 0.048) — xác nhận chẩn đoán — nhưng F1 không tăng; minority ratio ≈ 0 (prototype yếu bị hard-max bỏ hoàn toàn). → K=2 thực chất vẫn K=1 về hiệu năng.
- **ASF-E energy gate**: vượt control fixed cùng recipe (+0.39%) nhưng không vượt A2fix-ent GAP-gate (0.9464 vs 0.9472); ASF-EA anneal 0.9454. → giữ ASF v1.

### 5.3. Kết luận v2
Không candidate nào vượt cấu hình v1 tương ứng → **mô hình nộp chính giữ v1** (ASF A2fix-ent, CMP C2); toàn bộ v2 là diagnostics + negative results + mở rộng đánh giá test đủ seed.

---

## 6. Vòng V3 — Knowledge Distillation

### 6.1. Thiết lập
Teacher: **ConvNeXt-T seed 2** — best val macro-F1 0.9691 trong 3 seed của v1 (chọn theo val), checkpoint SHA256 `9627ca43…` xác minh khi load, val accuracy verify 0.9692; frozen eval + requires_grad False + forward trong no_grad; BN running stats xác nhận bất biến qua vòng train (K5). Students khởi tạo ImageNet pretrained (không warm-start từ checkpoint task). Loss: L = (1 − α)·CE + α·T²·KL(softmax(z_t/T) ‖ log_softmax(z_s/T)), **α = 0.5, T = 4**, batchmean, FP32; teacher/student cùng tensor x (same-view), không cache logits.

### 6.2. Diagnostics KD (K1–K5, PASS toàn bộ)
Chiều KL khớp tính tay (0.949311); teacher grad None qua no_grad; T² scale có mặt + FP32 ổn định; α=0 trả về đúng CE (sai số < 1e-6); teacher weights + BN stats bất biến khi train qua `train_model` thật.

### 6.3. Pilot (100b, 2 epoch)
Teacher online: KD epoch **100–104s** (~2× CE epoch ~49–53s); KD aux loss 9.38 → 2.48; peak mem 3.49 GiB. Stage local 125s (215 ảnh/s).

### 6.4. Screening (110b, 15 epoch, seed 0)

| Student | CE s0 | KD s0 | Δ | KD loss theo epoch |
|---|---|---|---|---|
| ResNet18-KD | 0.9483 | **0.9550** | **+0.67** | 14.96 → 0.65 (hội tụ đẹp) |
| ASF-KD | 0.9507 | **0.9565** | **+0.58** | 13.99 → 0.53 |
| CMP-KD (hard-max) | 0.9503* | 0.9448 | *—* | 24.03 → 18.46 (không hội tụ tốt) |

KD control có tín hiệu rõ (loại trừ nghi ngờ implementation/teacher quality). **CMP-KD negative**: KD loss không hội tụ trên hard-max head → CMP không đưa KD vào final. Phân tích bổ sung (đo trên 3.149 ảnh validation): logits CMP bị chặn κ·cos với κ học được 3.823 → trần soft-max student 13.32%/lớp, trong khi teacher soft targets (T=4) median max-prob 0.5985 — **99.43% teacher targets vượt trần student**, giải thích định lượng việc KD loss kẹt ~18.5. Đây là hạn chế tham số hóa: nếu cải thiện CMP, ưu tiên tham số hóa scale kiểu ArcFace (κ = exp(θ), scale 30–64) trước khi thêm thành phần khác. Lưu ý so sánh: CE 0.9503 là đầu LSE còn KD là đầu hard-max — khác cả kiến trúc head lẫn loss, nên KHÔNG kết luận "KD làm CMP giảm 0.55"; kết luận đúng: "cấu hình hard-max-KD được thử không cải thiện". Control hard-max-CE (không KD, 8 epoch) = 0.9394 — khác budget, chỉ tham chiếu.

### 6.5. KD Final 3-seed validation (111)

| Student | seed 0 | seed 1 | seed 2 | mean ± std | Paired KD − CE cùng seed |
|---|---|---|---|---|---|
| ResNet18-KD | 0.9550 | 0.9555 | 0.9552 | **0.9552 ± 0.0003** | [+0.67, +0.50, +0.28] → mean **+0.48** (3/3 dương) |
| ASF-KD | 0.9565 | 0.9587 | 0.9576 | **0.9576 ± 0.0011** | [+0.58, +0.44, −0.02] → mean **+0.33** (2/3 dương) |

Gate ASF-KD phân hóa mạnh dưới KD (max_mean 0.46; std LL4 0.26) — khác rõ trạng thái gần-đều của A2fix-ent CE.

### 6.6. KD TEST 3-seed (112c, kế hoạch khóa trước)

| Model | Acc mean ± std | F1 mean ± std | Latency ms |
|---|---|---|---|
| ResNet18-KD | **0.9602 ± 0.0020** | **0.9601 ± 0.0019** | 2.53 |
| ASF-KD | 0.9591 ± 0.0014 | 0.9591 ± 0.0014 | 3.76 |

**Paired vs CE 3-seed trên test:**
- ResNet18-KD − ResNet18-CE: [+0.58, +0.55, +0.81] điểm phần trăm → mean **+0.65, 3/3 seed dương**
- ASF-KD − ASF-CE: [+0.91, +0.72, +0.48] → mean **+0.70, 3/3 seed dương**

---

## 6b. Vòng V4 (Hướng A) — Sửa lỗi `κ = exp(θ)` cho CMP & Screening ASF-Lite

### 6b.1. Cấu hình V4 và Screening (kernel `120`, seed 0, 15 epoch)
Trong `src/models.py`, `logit_scale` khởi tạo `math.log(10.0)` nhưng bản v1–v3 thiếu `.exp()` khi tính `kappa = logit_scale.clamp(max=ln 100)`, khiến `κ` bị khóa `≤ 4.605` (ở checkpoint `CMP-KD` v3 cũ có `κ = 3.823`, tạo trần softmax `13.32%` tại `T = 4` như §6.4 đã chứng minh). Vòng v4 sử dụng gói cấu hình mới (`κ = exp(θ)` khởi tạo `10.0`, chặn trên `64.0`, kết hợp `hardmax` + `k-means init` — viết gọn là `CMP-exp`) cùng một biến thể đối chứng `ASF-Lite` ở seed 0:

| Cấu hình | Val Acc (s0) | Val Macro-F1 (s0) | Δ F1 vs ResNet18 s0 | Latency T4 (ms) | Ghi chú |
|---|---|---|---|---|---|
| `cmp_exp_ce` (cấu hình V4 CE) | **0.9543** | **0.9542** | **+0.59%** (`0.9483` → `0.9542`) | 3.77 | `κ = 14.26`, `inter-cos = 0.553` (không collapse) |
| `cmp_exp_kd` (cấu hình V4 + KD) | **0.9562** | **0.9561** | **+0.11%** (`0.9550` → `0.9561`) | 3.77 | `κ = 19.38`, `KD loss: 15.21 → 0.508` (vs `CMP-KD` cũ `0.9448` / `18.46` trên val s0) |
| `asf_lite_ce` (`layer2+layer3` DWT, RMS) | 0.9505 | 0.9504 | +0.21% | 4.07 | Dừng ở screening (chậm hơn ASF v1) |
| `asf_lite_kd` (như trên + KD) | 0.9555 | 0.9555 | +0.05% | 4.28 | Thấp hơn ASF v1 KD (`0.9565`) |

Theo kế hoạch khóa [final_plan_v4_cmp_exp.json](../notes/final_plan_v4_cmp_exp.json), `asf_lite` dừng ở screening; `cmp_exp_ce` và `cmp_exp_kd` được đẩy lên đủ 3 seeds + đánh giá TEST (kernel `121`). Mức tăng của `CMP-exp` so với `CMP v1` (`CMP_C2`) là kết quả tổng hợp của **gói cấu hình V4** (scale mũ `κ = exp(θ)` + `hardmax` + `k-means init`), không quy riêng cho từng thành phần.

### 6b.2. Kết quả 3-seed Validation và TEST của `CMP-exp` (kernel `120` + `121`)

| Mô hình | Val F1 (3-seed) | Test Acc (3-seed) | Test Macro-F1 (3-seed) | Latency (ms, cùng máy 121) | Train h/run (+k-means) |
|---|---|---|---|---|---|
| `CMP-exp-CE` | **0.9544 ± 0.0010** | **0.9549 ± 0.0022** | **0.9548 ± 0.0021** | **3.69 ± 0.02** *(RN18 cùng máy: 4.11)* | 0.198h *(`+44.7s` = 0.012h k-means → tổng 0.210h)* |
| `CMP-exp-KD` | **0.9557 ± 0.0005** | **0.9573 ± 0.0005** | **0.9574 ± 0.0005** | **3.71 ± 0.00** *(RN18 cùng máy: 4.11)* | 0.390h *(`+42.0s` = 0.012h k-means → tổng 0.402h)* |

**So sánh ghép cặp theo seed `{0, 1, 2}` trên TEST macro-F1 (tính từ số chưa làm tròn):**
- `CMP-exp-CE − CMP-CE (v1)`: `[+0.67, +0.11, +0.38]` điểm phần trăm → mean **+0.39, 3/3 seed dương**.
- `CMP-exp-CE − B1 ResNet18-CE`: `[+0.42, −0.34, +0.26]` → mean **+0.11, 2/3 seed dương** *(trên Validation thắng **3/3 seed**, mean **+0.40**, std `0.0010` bằng nửa B1)*. Đây là cấu hình CE tự chế đầu tiên vượt `B1 ResNet18-CE` trên cả 3-seed Val lẫn 3-seed Test.
- `CMP-exp-KD − CMP-exp-CE`: `[+0.16, +0.48, +0.13]` → mean **+0.26, 3/3 seed dương**, độ ổn định giữa các seed cao nhất nhóm KD (`std = 0.0005`), nhưng vẫn thấp hơn `ResNet18-KD` (`96.01%`) trên test.

---

## 7. Bảng TEST tổng hợp cuối (3 seeds — nguồn `deliverables/results_master.csv`)

| Model | Params M | Acc % ± std | F1 % ± std | P macro % | R macro % | Latency ms (3-seed) | Train h (+k-means) |
|---|---|---|---|---|---|---|---|
| B0 SmallCNN | 0.29 | 59.46 ± 0.65 | 59.04 ± 0.64 | 60.64 | 59.46 | 1.54 *(071b)* | 0.344 |
| B1 ResNet18-CE | 11.20 | 95.37 ± 0.19 | 95.37 ± 0.19 | 95.44 | 95.37 | 2.52 *(071b)* / 4.11 *(121)* | 0.200 |
| B2 MobileNetV3-S | 1.56 | 94.69 ± 0.15 | 94.69 ± 0.15 | 94.78 | 94.69 | 4.91 *(071b)* | 0.180 |
| B3 EfficientNet-B0 | 4.07 | 96.08 ± 0.15 | 96.07 ± 0.15 | 96.12 | 96.08 | 7.40 *(071b)* | 0.331 |
| B4 ConvNeXt-T | 27.85 | 96.78 ± 0.02 | 96.77 ± 0.02 | 96.81 | 96.78 | 6.87 *(071b)* | 0.616 |
| **ASF-CE** *(v1)* | 11.60 | 95.20 ± 0.26 | 95.20 ± 0.26 | 95.29 | 95.20 | 3.82 *(071b)* | 0.209 |
| CMP-CE *(v1)* | 11.49 | 95.10 ± 0.12 | 95.09 ± 0.12 | 95.15 | 95.10 | 2.80 *(071b)* | 0.197 |
| **CMP-exp-CE** *(v4)* | 11.49 | **95.49 ± 0.22** | **95.48 ± 0.21** | 95.54 | 95.48 | 3.69 *(121, RN18: 4.11)* | 0.198 *(+0.012 km)* |
| **ResNet18-KD** *(v3)* | 11.20 | **96.02 ± 0.20** | **96.01 ± 0.19** | 96.07 | 96.02 | 2.53 *(112c)* | 0.384 |
| **ASF-KD** *(v3)* | 11.60 | 95.91 ± 0.14 | 95.91 ± 0.14 | 95.96 | 95.91 | 3.76 *(112c)* | 0.398 |
| **CMP-exp-KD** *(v4)* | 11.49 | 95.73 ± 0.05 | 95.74 ± 0.05 | 95.79 | 95.73 | 3.71 *(121, RN18: 4.11)* | 0.390 *(+0.012 km)* |

Mọi giá trị khớp trực tiếp với `deliverables/results_master.json` / `.csv`. **Phạm vi latency:** toàn bộ là trung bình của 3 giá trị median qua 3 seed (`Tesla T4 cuda`, `batch = 1`, `256×256`, `n_warmup = 10`, `n_iter = 50`); do bảng tổng hợp gộp từ 3 phiên đánh giá (`071b`, `112c`, `121`), khi so `CMP-exp` với `ResNet18` dùng phép đo đối chứng cùng máy ở `121` (`3.69–3.71 ms` vs `4.11 ms`). **Phạm vi thời gian huấn luyện:** toàn bộ các run CE và KD ở cả 3 seed đều có `executed_epochs = 15` (`060a/b`, `110b/111`, `120/121`); thời gian KD tăng ~`2×` (`~0.38–0.40h` vs `~0.20–0.21h`) là do teacher ConvNeXt-T forward online. Với `CMP-exp`, `train_model` chưa gồm bước `init_prototypes_kmeans` (`~44.7s` cho CE, `~42.0s` cho KD — cộng thêm `0.012h/run`).

---

## 8. Trả lời các câu hỏi nghiên cứu

**RQ1 (ASF có cải thiện?):** Có trên validation — paired 3/3 seed (`+0.39` điểm F1, `p ≈ 0.05`). Trên test đủ seed: `ASF-CE` (`95.20%`) vượt `B2` (`94.69%`, `+0.51` điểm) nhưng chưa vượt `B1` (`95.37%`, `−0.16` điểm, `1/3` seed). Với KD, `ASF-KD` (`95.91%`) không vượt `ResNet18-KD` (`96.01%`). `ASF-Lite` (v4 screening) không vượt `ASF v1 KD` và chậm hơn (`4.07–4.28 ms`) nên dừng.

**RQ2 (CMP có cải thiện?):** Ở v1–v2 (7 cấu hình cũ), `CMP-CE (v1)` (`95.09%`) thua `B1` (`95.37%`) và `CMP-KD` v3 cũ (`κ = 3.823`, trần softmax `13.32%` tại `T = 4`) kẹt ở `94.48%` val s0. Ở v4, gói cấu hình **`CMP-exp`** (`κ = exp(θ) ∈ [10, 64]` + `hardmax` + `k-means init`) đưa **`CMP-exp-CE` đạt 95.44% val (3/3 thắng B1) và 95.48% test macro-F1** — vượt `B2` `+0.79` điểm, vượt `CMP v1` `+0.39` điểm (`3/3` seed) và vượt nhẹ `B1` `+0.11` điểm (`2/3` seed); `CMP-exp-KD` tăng từ `94.48%` lên `95.61%` (val s0) và đạt `95.74% ± 0.05%` trên test (`3/3` seed thắng `CMP-exp-CE`), nhưng vẫn thấp hơn `ResNet18-KD` (`96.01%`).

**RQ3 (chi phí):** Toàn bộ CE và KD đều chạy `15` epoch ở cả 3 seed. KD tăng thời gian train ~`2×` (`~100s` vs `~47–50s/epoch`) do teacher online; inference latency giữ nguyên (`2.53` vs `2.52 ms` với RN18). `CMP-exp-CE` chạy cùng tốc độ train `B1` (`0.198h` + `0.012h` k-means) và latency cùng máy tương đương (`3.69 ms` vs `4.11 ms`). `ASF-CE` thêm `+3.6%` params (`11.60M`) và `+52%` latency (`3.82 ms` vs `2.52 ms`).

---

## 9. Đóng góp và kết luận theo từng vòng

| Vòng | Đóng góp | Bằng chứng |
|---|---|---|
| V1 | Protocol chống leakage có kiểm chứng; benchmark 7 mô hình × 3 seed; robustness matrix | `test_3seed_report.json` (071b), split sha `cc70020a` |
| V2 | 5 diagnostics định lượng (margin sai chiều; collapse thật; residual không chết; signed-GAP triệt tiêu; selection effect); 4 candidate negative có phân tích | `v2_diagnostics.json`, registry 040/040b/050/050b/090 |
| V3 | KD chuẩn được kiểm chứng K1–K5; cải thiện +0.65/+0.70 điểm test nhất quán; quy trình stage-local + logging per-epoch | `v3_diagnostics.json`, `test_kd_3seed_report.json` |
| V4 | Sửa `κ = exp(θ)` cho CMP: `CMP-exp-CE` đạt `95.48%` F1 test (vượt CMP v1 & B1); `CMP-exp-KD` hội tụ sạch đạt `95.74% ± 0.05%` | `h1a_screen_report.json` (120), `cmp_exp_test_3seed_report.json` (121) |

**Negative results được bảo toàn:** gate sụp 1-hot (040); margin v1 sai chiều (D1); prototype collapse bền vững v1/v2 (050/050b); energy gate không vượt GAP gate (090); CMP-KD v3 kẹt trần softmax 13.32% (110b/v3b); ASF lợi ích không khái quát test (071b); ASF-Lite chậm hơn ASF v1 (120).

---

## 10. Hạn chế

1. Một dataset; chưa kiểm chứng EuroSAT.
2. Test đã quan sát qua các vòng — mọi kết quả test v3 là nghiên cứu mở rộng; không chọn cấu hình bằng test nhưng khả năng vô thức thiên lệch vẫn tồn tại.
3. 3 seed là mẫu nhỏ; không thực hiện kiểm định tương đương nên không tuyên bố hai mô hình "như nhau".
4. Robustness là corruption tổng hợp trong phòng.
5. Per-epoch history v1 không lưu đầy đủ vào artifact (chỉ v2/v3).
6. Cơ chế prototype collapse đầy đủ chưa được tái lập trong synthetic — chỉ xác nhận hiện tượng và một hướng sửa hữu hiệu về cơ chế.
7. Không so sánh với SOTA chuyên biệt ngoài 5 baseline đã chọn.
