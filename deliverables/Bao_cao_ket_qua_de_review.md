# Báo cáo kết quả chi tiết — để review độc lập

**Dự án:** Phân loại cảnh ảnh viễn thám NWPU-RESISC45 — 2 kiến trúc đề xuất (ASF, CMP) + knowledge distillation
**Ngày:** 2026-11-02 · **Trạng thái:** hoàn thành 3 vòng nghiên cứu (v1/v2/v3), 20 kernel Kaggle
**Mục đích tài liệu:** nộp cho reviewer bên ngoài kiểm tra tính hợp lệ của kết quả, phương pháp và cách diễn giải. Tài liệu tự chứa đủ ngữ cảnh, không cần truy cập repo. Tất cả số liệu truy ngược được về artifact nguồn (checkpoint/report JSON) — kèm SHA256 ở mục 11.

**Yêu cầu review:** xem mục 12 — liệt kê 10 câu hỏi/cần lộ trình cụ thể cần ý kiến.

---

## 1. Thiết lập thí nghiệm

**Dataset.** NWPU-RESISC45: 31.500 ảnh viễn thám RGB 256×256, 45 lớp × 700 ảnh. Nguồn Kaggle dataset `punfake/resisc45`. Audit toàn bộ: 0 lỗi decode, đủ số lượng/lớp; 8 nhóm trùng tuyệt đối (MD5) và 16 cặp gần trùng (dHash mutual-NN, Hamming ≤ 6) được gom nhóm; 4 nhóm chéo lớp giữ cùng split để loại rò rỉ (kiểm tra pixel cho thấy chỉ tương tự cấu trúc, không phải trùng thật).

**Split.** 75/10/15 phân tầng theo lớp, seed 42, chia theo nhóm trùng: train 23.626 / val 3.149 / test 4.725; mỗi lớp 524–526 / 69–70 / 104–107. Manifest SHA256: `cc70020a85091b4d86fd519e6246b2ca397a20bca551803d532bf8cc1254645a` — dùng chung mọi vòng/mô hình, không chia lại.

**Protocol chung (mọi mô hình, mọi seed).** Tối đa 15 epoch, early-stop patience 5; checkpoint chọn theo **validation macro-F1**; AdamW (lr backbone 1e-4, head 1e-3, weight decay 1e-4, cosine + warmup 5% số step); batch 128; AMP; input 256×256, chuẩn hóa ImageNet; augmentation train: RandomResizedCrop(scale 0.7–1.0) + hflip + vflip (không MixUp/CutMix). 3 seeds {0,1,2}. Phần cứng: Kaggle Tesla T4 15GB. Dataset được copy sang disk local đầu mỗi kernel (không đọc qua network mount khi train).

**Test discipline.** Test chỉ được đánh giá khi kế hoạch đã khóa (`final_plan*.json`); v1 đánh giá test một lần (best-val seed); v2 mở rộng test đủ 3 seed theo kế hoạch cố định; **test đã được quan sát qua các vòng — mọi kết quả test v3 được công bố là nghiên cứu mở rộng**, không dùng test để chọn cấu hình.

---

## 2. Hai kiến trúc đề xuất

### 2.1 ASF (Adaptive Scale–Frequency ResNet18)

ResNet18 ImageNet-pretrained **giữ nguyên toàn bộ**; lấy feature maps layer2 (128ch, 32×32) và layer4 (512ch, 16×16), tách Haar wavelet mức 1 → 8 thành phần (mỗi stage: LL ≡ feature map gốc trước tách + LH/HL/HH). Mỗi thành phần: 1×1 conv → BN → ReLU → GAP → vector f_i (128-d). Gate softmax theo từng ảnh: w = softmax(MLP(1024 → 8)). Đầu ra: head( GAP(layer4) + Linear_small-init(Σ w_i·f_i) ) — residual khởi tạo nhỏ để bảo toàn biểu diễn pretrained lúc bắt đầu.

**Lịch sử cấu hình:** v1 zero-init residual + gate MLP → **gate sụp 1-hot về LL4** (mean = [0,0,0,0,1,0,0,0], std = 0 trên ảnh thật). Sửa v2: out_proj init std=1e-3 + **entropy regularizer** λ·(log 8 − H(w))/log 8, λ = 0.05 → gate sống (thành phần lớn nhất 0.14–0.19, std > 0). Cấu hình cuối ASF = A2fix-ent.

### 2.2 CMP (Confusion-guided Multi-Prototype)

Backbone ResNet18 → GAP → embedding Linear 512-d → L2 normalize. K=2 prototype/lớp (L2 normalize). Logits: v1 = logsumexp(κ·cos) − log K; v2 = κ·max_k(cos) (hard-max, theo Sub-center ArcFace). κ học được, clamp max ln 100. **Confusion graph** xây chỉ từ train: probe ResNet18 train trên 90% train, dự đoán 10% còn lại, 6 epoch → đếm nhầm theo cặp, lọc ≥3, chuẩn hóa hàng, cap [0,1] → 8 cặp đối xứng đúng ngữ nghĩa (church↔palace, circular↔rectangular farmland, dense↔medium residential, wetland↔lake…).

**Lịch sử:** v1 margin trừ logit lớp âm trước CE (conf_margin 0.3) — diagnostics D1 chứng minh **sai chiều**: giảm CE mẫu khó (0.714 → 0.604) và giảm gradient lớp khó (0.401 → 0.331), tức down-weight mẫu khó thay vì margin phân biệt. v2 pair loss sửa chiều: L_pair = Σ w(y,c)·softplus(m + s_c − s_y)/Z, m = 0.1, λ = 0.5 warmup 3 epoch — kiểm chứng synthetic: negative gần embedding đúng → loss tăng.

---

## 3. Kết quả vòng v1 (CE thuần)

### 3.1 Validation macro-F1 (3 seeds, mean ± std)

| Model | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| B0 SmallCNN | 0.5854 | 0.5790 | 0.5819 | 0.5821 ± 0.0032 |
| B1 ResNet18-CE | 0.9483 | 0.9505 | 0.9524 | 0.9504 ± 0.0021 |
| B2 MobileNetV3-S | 0.9436 | 0.9420 | 0.9417 | 0.9424 ± 0.0010 |
| B3 EfficientNet-B0 | 0.9584 | 0.9581 | 0.9574 | 0.9580 ± 0.0005 |
| B4 ConvNeXt-T | 0.9669 | 0.9654 | 0.9691 | 0.9671 ± 0.0019 |
| ASF A2fix-ent | 0.9507 | 0.9543 | 0.9578 | 0.9543 ± 0.0036 |
| CMP C2 | 0.9503 | 0.9507 | 0.9488 | 0.9499 ± 0.0010 |

### 3.2 Test — best-val seed (bản công bố đầu v1)

| Model | seed | Acc | F1 macro |
|---|---|---|---|
| B1 | 2 | 0.9530 | 0.9531 |
| ASF | 2 | 0.9549 | 0.9550 |
| CMP | 1 | 0.9515 | 0.9515 |

### 3.3 Test mở rộng đủ 3 seed (thực hiện v2, kế hoạch khóa trước)

| Model | Acc mean ± std | F1 mean ± std | F1 per seed |
|---|---|---|---|
| B1 | 0.9537 ± 0.0019 | 0.9537 ± 0.0019 | 0.9530 / 0.9509* / 0.9576* |
| ASF | 0.9520 ± 0.0026 | 0.9520 ± 0.0026 | 0.9527 / 0.9443 / 0.9587 |
| CMP | 0.9510 ± 0.0012 | 0.9509 ± 0.0012 | 0.9508 / 0.9516 / 0.9503 |

*Per-seed F1 test lấy từ report 071b; thể hiện đúng từng seed.

**Paired F1 theo seed (test):** ASF − B1 = [−0.0012, −0.0056, +0.0019], mean **−0.0016** (1/3 seed dương); CMP − B1 = [−0.0025, −0.0045, −0.0012], mean **−0.0027** (0/3).

**Phát hiện selection effect:** trên val, ASF > B1 paired **3/3 seed** (+0.24/+0.38/+0.54 điểm phần trăm, mean +0.39, t≈4.5, p≈0.05). Trên test đủ seed, ASF ≈ B1 (−0.16 điểm, 1/3). Con số test +0.19% trong bản công bố đầu đo bằng seed tốt nhất theo val — khái quát không giữ được. Cả hai mô hình đề xuất đều vượt B2 trên test 3-seed (ASF +0.51, CMP +0.40 điểm phần trăm).

---

## 4. Screening (8 epoch, seed 0 — không phải final benchmark)

### 4.1 ASF ablation

| Cấu hình | val F1 | Ghi chú |
|---|---|---|
| A1 fixed fusion (zero-init) | 0.9469 | |
| A2 adaptive gate (zero-init) | 0.9456 | **gate sụp 1-hot LL4 (mean 1.0, std 0)** |
| A2-conv (detail học được, control) | 0.9457 | sụp tương tự → không phải lỗi Haar |
| A2fix-small (init 1e-3) | 0.9453 | vẫn sụp (max_mean 1.0) |
| **A2fix-ent (init 1e-3 + entropy 0.05)** | **0.9472** | **gate sống: max_mean 0.16, std > 0** |

### 4.2 CMP ablation

| Cấu hình | val F1 | inter-cos |
|---|---|---|
| C1 cosine K=1 | 0.9422 | — |
| C2 K=2 (LSE) | 0.9441 | **1.0000** |
| C3 K=2 + margin 0.3 | 0.9404 | 1.0000 |
| C2fix k-means + diversity 0.01 | 0.9398 | 1.0000 |
| C3fix margin 0.1 | 0.9413 | 1.0000 |

---

## 5. Vòng v2 — diagnostics bằng số trên checkpoint thật

| ID | Nghi vấn | Phương pháp | Kết quả |
|---|---|---|---|
| D1 | Margin v1 chiều nào? | Synthetic logits, so CE ± margin, gradient từng lớp | **XÁC NHẬN sai chiều**: CE giảm 0.714→0.604 trên mẫu khó; gradient lớp khó 0.401→0.331 → down-weight, không phải CosFace margin |
| D2 | Collapse tái lập được? | Synthetic 1 lớp, LSE/hard-max, κ=10, 200 step | **KHÔNG** — prototype tách nhau (−0.85…−0.92) kể cả init v1-random → cơ chế chưa đủ bằng chứng |
| D3 | Checkpoint thật có collapse? | FP32 diagnostics trên ckpt s1 + assignment trên 300 ảnh train | **CÓ THẬT**: 45/45 lớp pair-cos 1.00000000 (euclid 0.00035); cross-class center cos −0.023 → metric đo đúng đại lượng |
| D4 | "Gate tắt ≡ ResNet18"? | Đo residual/backbone norm + max\|Δlogit\| trên checkpoint ASF final | **BÁC BỎ**: residual/backbone = 0.37; max\|Δlogit\| = 8.50 |
| D4b | Signed-GAP triệt tiêu? | Đo \|E[comp]\| vs RMS theo thành phần | **CĂN CỨ CÓ**: LL4 1.025 vs LH4 0.054 (19×); RMS chỉ 4.3× → energy descriptor hợp lý |

### V2 screening kết quả

**CMP-v2:** CMP-P (hard-max + k-means init) 0.9394, CMP-PG (+pair loss đúng chiều) 0.9416 — **hard-max phá collapse cơ học (inter-cos 1.0 → 0.73/0.74) nhưng F1 không tăng** (minority ratio ≈ 0: prototype yếu bị bỏ hoàn toàn). → vấn đề không nằm ở head prototype; CMP dừng đúng trần 2 cấu hình + vòng fix.

**ASF-v2:** control fixed+small-init 0.9425 / ASF-E energy gate 0.9464 / ASF-EA anneal 0.9454. Energy gate vượt control (+0.39%) nhưng **không vượt A2fix-ent GAP-gate (0.9472)** → ASF giữ v1.

---

## 6. Vòng v3 — Knowledge Distillation

**Teacher:** ConvNeXt-T seed 2 — best val macro-F1 0.9691 trong 3 seed (chọn theo val, không test), checkpoint SHA256 `9627ca43…` xác minh khi load, val accuracy verify 0.9692. Frozen eval + requires_grad False + forward trong no_grad; BN running stats xác nhận bất biến (K5).

**Loss:** L = (1 − α)·CE(student, y) + α·T²·KL(softmax(z_t/T) ‖ log_softmax(z_s/T)), α = 0.5, T = 4, reduction batchmean, FP32. Teacher/student cùng tensor x (same-view); không cache logits.

**Diagnostics KD (K1–K5, đều PASS):** chiều KL khớp tính tay (0.949311); teacher grad None; T² có mặt; α=0 ≡ CE (sai số < 1e-6); teacher weights + BN stats bất biến qua `train_model` thật.

### 6.1 KD validation 3-seed (15 epoch)

| Student | seed 0 | seed 1 | seed 2 | mean ± std | Paired vs CE cùng seed |
|---|---|---|---|---|---|
| ResNet18-KD | 0.9550 | 0.9555 | 0.9552 | **0.9552 ± 0.0003** | [+0.67, +0.50, +0.28] → mean **+0.48** (3/3 dương) |
| ASF-KD | 0.9565 | 0.9587 | 0.9576 | **0.9576 ± 0.0011** | [+0.58, +0.44, −0.02] → mean **+0.33** (2/3 dương) |

### 6.2 KD TEST 3-seed — paired vs CE 3-seed (kết quả quyết định)

| So sánh | F1 diffs theo seed | Mean | Wins |
|---|---|---|---|
| RN18-KD − RN18-CE | [+0.58, +0.55, +0.81] điểm phần trăm | **+0.65** | **3/3** |
| ASF-KD − ASF-CE | [+0.91, +0.72, +0.48] | **+0.70** | **3/3** |

**KD TEST absolute (mean ± std):** ResNet18-KD **96.02% ± 0.20** acc / 96.01 ± 0.19 F1; ASF-KD **95.91% ± 0.14** acc / 95.91 ± 0.14 F1. Tham chiếu: B1-CE 95.37 ± 0.19; B3 96.07 ± 0.15; B4 96.78 ± 0.02.

### 6.3 CMP-KD (negative)

CMP-KD hard-max: val 0.9448 (−0.55% vs CE s0); KD loss không hội tụ tốt (24.0 → 18.5, cao bất thường so 15.0/2.6 của student khác); prototype inter-cos 0.92. → CMP không đưa KD vào final; giữ CMP-CE C2.

---

## 7. Bảng TEST tổng hợp cuối (3 seeds — nguồn `results_master.csv`)

| Model | Params M | Acc % | F1 % | P macro % | R macro % | Latency ms | Train h |
|---|---|---|---|---|---|---|---|
| B0 SmallCNN | 0.29 | 59.46 ± 0.65 | 59.04 ± 0.64 | — | — | 1.54 | 0.34 |
| B1 ResNet18-CE | 11.20 | 95.37 ± 0.19 | 95.37 ± 0.19 | 95.38 | 95.37 | 2.52 | 0.20 |
| B2 MobileNetV3-S | 1.56 | 94.69 ± 0.15 | 94.69 ± 0.15 | 94.75 | 94.69 | 4.91 | 0.18 |
| B3 EfficientNet-B0 | 4.07 | 96.08 ± 0.15 | 96.07 ± 0.15 | 96.11 | 96.08 | 7.40 | 0.33 |
| B4 ConvNeXt-T | 27.85 | 96.78 ± 0.02 | 96.77 ± 0.02 | 96.82 | 96.78 | 6.87 | 0.62 |
| ASF-CE | 11.60 | 95.20 ± 0.26 | 95.20 ± 0.26 | 95.26 | 95.20 | 3.82 | 0.21 |
| CMP-CE | 11.49 | 95.10 ± 0.12 | 95.09 ± 0.12 | 95.16 | 95.09 | 2.56 | 0.20 |
| **ResNet18-KD** | 11.20 | **96.02 ± 0.20** | **96.01 ± 0.19** | 96.04 | 96.03 | 2.53 | 0.21 (CE-recipe tương đương) |
| **ASF-KD** | 11.60 | 95.91 ± 0.14 | 95.91 ± 0.14 | — | — | 3.76 | 0.21 |

P/R macro của ResNet18-KD/ASF-KD lấy từ 112c per-checkpoint mean; P/R các dòng khác mean 3-seed từ 071b.

## 8. Robustness (test accuracy, corruption tổng hợp)

| Model | clean | blur σ1 | blur σ2 | JPEG q30 | JPEG q60 | bright ×0.6 | ×1.4 |
|---|---|---|---|---|---|---|---|
| B1-CE | 0.9537 | 0.9310 | 0.7458 | 0.9346 | 0.9378 | 0.9335 | 0.9263 |
| ASF-CE | 0.9520 | 0.9376 | 0.7780 | 0.9359 | 0.9407 | 0.9380 | 0.9283 |
| CMP-CE | 0.9510 | 0.9249 | 0.7215 | 0.9319 | 0.9333 | 0.9274 | 0.9263 |
| B3 | 0.9608 | 0.9458 | 0.6214 | 0.9416 | 0.9494 | 0.9494 | 0.9505 |
| B4 | 0.9678 | 0.9608 | 0.8732 | 0.9568 | 0.9644 | 0.9657 | 0.9628 |

ConvNeXt-T bền nhất mọi loại; EffB0 sụp mạnh blur σ2 (−33.9 điểm); ASF-CE ≥ B1 tuyệt đối 6/6 corruption, theo delta 5/6.

## 9. Chi phí

- **Train (h/run, mean 3 seed):** B0 0.34 · B1 0.20 · B2 0.18 · B3 0.33 · B4 0.62 · ASF 0.21 · CMP 0.20 · KD runs 0.21 (tính cả teacher inference online — hệ số ~2× so CE cùng kiến trúc).
- **Teacher:** training cost lịch sử 0.62h (v1, ghi riêng); v3 chỉ chi teacher inference online.
- **Latency batch-1 (median, ms):** B0 1.54–2.52 · B1 2.52–2.56 · B2 4.60–4.91 · B3 7.36–7.40 · B4 5.96–6.87 · ASF 3.72–3.82 · CMP 2.56–2.80. Throughput batch-256: 136–384 imgs/s.
- **Tổng GPU dự án:** ~16h Tesla T4 qua 3 vòng (measured từ runtime); CPU audit ~3h.

## 10. Audit tính hợp lệ (thực hiện trước khi đóng bảng)

1. **Fairness CE vs KD:** so config 6 cặp checkpoint (RN18/ASF × seed 0/1/2) từ chính checkpoint files — 12 trường protocol (epochs, batch, lr_backbone, lr_head, wd, AMP, warmup, patience, input, smoothing, num_workers, seed) **giống hệt**; khác biệt duy nhất kd_alpha/kd_T (biến điều tra). → Phân tích "lợi ích KD" hợp lệ trong protocol; không có confound epochs/optimizer/augmentation.
2. **Gộp seed 0 KD (từ kernel 110b) với seeds 1/2 (kernel 111):** config s0 vs s1/s2 chỉ khác best-epoch tự nhiên; teacher SHA256 xác nhận trong logs cả hai kernel; code nhúng trong notebook 110b (artifact bất biến) chứa nhánh gate **đúng** — bug `_gate_input` chỉ được giới thiệu vào working tree sau khi 110b chạy (git pickaxe: fix tại commit 896f5cf 11:51; 110b push 09:20), phát hiện qua smoke 071. → Gộp 3-seed hợp lệ.

## 11. Truy vết artifact (SHA256 trong `deliverables/manifest.json`)

12 file nguồn đã hash: 5 checkpoint (RN18-CE s2, ASF-CE s1, teacher s2, RN18-KD s0, ASF-KD s1), split_manifest.csv, class_order.json, test_report.json (v1), test_3seed_report.json (v2), test_kd_3seed_report.json (v3), kd_screen_report.json, kd_final_report.json. Sau sự cố xóa `outputs/` (ngày 02/11), toàn bộ được re-pull từ Kaggle và **verify 12/12 SHA256 khớp nguyên bản**.

## 12. Câu hỏi/cần lộ trình cho reviewer

1. Cách phân tích selection effect (§3.3) có hợp lý không — có nên trình bày test best-val-seed và test 3-seed cạnh nhau như hiện tại, hay chỉ giữ 3-seed?
2. Kết luận "ASF cải thiện val nhưng không khái quát test" (§3.3) có cần thêm kiểm định nào trong phạm vi 3 seed?
3. Với CMP: hard-max phá collapse nhưng F1 không tăng — đánh giá của reviewer về hướng dừng ở đây (đúng trần tuning) vs tiếp tục (hướng nào)?
4. KD loss CMP không hội tụ (24→18.5) — có chẩn đoán kinh nghiệm nào về tương thích hard-max + KD loss nên thử trước khi loại hoàn toàn?
5. Việc tái sử dụng CE controls từ 060a/060b cho so sánh KD (cùng source/epochs/optimizer/seed, khác kd loss) có đủ điều kiện "fair" theo chuẩn reviewer, hay cần chạy lại CE control song song trong cùng kernel/session?
6. Bảng robustness chỉ đo trên best-val seed v1 — có nên mở rộng KD models không, hay giữ vì knowledge distillation không thay đổi đặc trưng robustness rõ rệt?
7. Mức cải thiện KD +0.65 điểm test (RN18) — reviewer đánh giá thế nào về tính đủ để là đóng góp chính của báo cáo so với cải thiện kiến trúc nhỏ hơn?

## 13. Hạn chế tự nhận (rút gọn — đầy đủ trong notes/limitations.md)

Một dataset; test đã quan sát; 3 seed là mẫu nhỏ (không dùng "không significant" để tuyên bố tương đương); robustness synthetic; per-epoch history v1 không lưu đầy đủ; margin v1 sai chiều chỉ phát hiện sau vòng v1; không so sánh SOTA chuyên biệt ngoài 5 baseline.
