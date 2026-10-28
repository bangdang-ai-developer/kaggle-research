# Report Outline — khung báo cáo cuối môn (cho TV2 review/viết)

**Quy ước:** mọi số lấy từ `notes/results_summary.md` (đã verify) — KHÔNG ghi số từ screening như final. Hình tham chiếu `notes/figures/`.

## 1. Mở bài (0.5 trang)
- Bài toán: phân loại 45 cảnh ảnh viễn thám (NWPU-RESISC45, 31.500 ảnh 256×256).
- Đặt vấn đề: hai hướng cải thiện trên backbone phổ thông — (1) lựa chọn đặc trưng đa tỉ lệ/dải tần theo ảnh (ASF), (2) học phân biệt các lớp dễ nhầm bằng multi-prototype (CMP) — với kiểm soát chi phí (RQ3).
- Đóng góp: ASF thắng backbone gốc +0.19% test F1 (paired 3/3 seed); CMP negative result có phân tích cơ học; bộ protocol chống leakage + cost analysis.

## 2. Dataset và audit (1 trang)
- Nguồn Kaggle `punfake/resisc45`; audit: 31.500 ảnh, 45×700, 0 decode lỗi, 100% 256×256.
- Duplicate: 8 nhóm exact (MD5) + 16 mutual-NN dHash (12 cùng/4 khác lớp) — co-split chống leakage; 39→20 candidate sau khi fix thuật toán (minh bạch quá trình).
- Split 75/10/15 per-class 524–526/69–70/104–107, seed 42, sha256 manifest.
- Hình: `fig_class_dist.png`, mẫu ảnh 45 lớp, bảng audit.

## 3. Phương pháp
### 3.1 Baselines (0.5 trang)
B0 SmallCNN từ đầu; B1 ResNet18; B2 MobileNetV3-S; B3 EffNet-B0; B4 ConvNeXt-T — ImageNet pretrained (trừ B0), cùng augmentation/optimizer protocol.

### 3.2 ASF-ResNet18 (1.5 trang) — công thức + hình `fig_asf_arch`
- 8 thành phần = (LL ≡ feature map, LH, HL, HH) × {layer2, layer4}; Haar DWT level-1 trên feature map.
- Gate theo ảnh: softmax MLP(8F→8); fusion residual qua Linear (init std=1e-3) + **entropy regularizer λ=0.05**.
- Bối cảnh thiết kế: v1 zero-init gây gate collapse về LL4 (negative result 040) — entropy reg sửa (040b).

### 3.3 CMP-ResNet18 (1 trang) — công thức + hình `fig_cmp_arch`
- Embedding L2-norm; K=2 prototype/lớp; logits = LSE(κ·cos) − log K (chuẩn hóa theo K, κ học được).
- Confusion margin CosFace-style từ graph train-only (probe 90/10, 6 epoch); graph đúng ngữ nghĩa (church↔palace, farmland, residential, wetland↔lake).

### 3.4 Protocol (0.5 trang)
Split 75/10/15 group-aware seed 42; 15 epoch early-stop 5; checkpoint val macro-F1; AdamW 1e-4/1e-3 cosine; AMP batch 128; 3 seeds; test khóa 1 lần duy nhất sau final_plan.

## 4. Kết quả (2 trang)
- Bảng val 3-seed + bảng TEST (results_summary §1–2).
- **ASF thắng B1: val paired 3/3 (+0.39%, p≈0.05); TEST +0.19%** — kèm paired-diff bar.
- CMP hòa B1 — negative result + collapse analysis (inter-cos = 1.0 mọi seed).
- Cost: bảng params/latency/throughput/train-time; trade-off figure `fig_test_f1_latency`.
- Hình: `fig_robustness_delta`, learning-curve nếu trích được từ log (đã ghi hạn chế).

## 5. Ablation (1 trang)
- ASF: A1 fixed vs A2 adaptive vs A2-conv control vs A2fix-ent (bảng 040/040b) — entropy reg là nhân tố sống còn của gate.
- CMP: C1→C3fix — margin 0.3 hại, prototype collapse bền vững qua 3 fix.
- Gate analysis hình `fig_gate_mean`; prototype usage bảng.

## 6. Robustness (0.5 trang)
- Ma trận 6 corruption × 7 model (`fig_robustness_delta`): ASF ≥ B1 5/6; EffB0 sụp blur σ2 (−34%); ConvNeXt-T robust nhất.
- Ghi chú: synthetic ≠ deployment.

## 7. Lỗi điển hình & phân tích (0.5 trang)
- Confusion theo cặp ngữ nghĩa: church↔palace, dense↔medium residential, wetland↔lake, circular↔rectangular farmland (từ confusion matrix ASF).
- Ví dụ đúng/sai từ `test_predictions.csv` (chọn 6–8 ảnh minh họa).

## 8. Hạn chế (0.5 trang)
Liệt kê từ `notes/limitations.md` — đủ 4 nhóm, không giấu.

## 9. Phân công (0.25 trang)
- TV1: code, pipeline, huấn luyện, thực nghiệm (src/models.py, 9 kernel Kaggle, protocol).
- TV2: review tài liệu, báo cáo, slides (đang xử lý bộ này).

## Hình cần có (checklist)
- [x] `confusion_*.png` ×7 (từ 070b)
- [x] `largest_group.png`, `near_dup_*.png` (từ 010)
- [ ] `fig_asf_arch`, `fig_cmp_arch` (vẽ tay/draw.io — TV2)
- [ ] `fig_class_dist`, sample images (sinh từ data/raw — script sẵn sàng)
- [ ] `fig_val_f1_final`, `fig_test_f1_latency`, `fig_robustness_delta`, `fig_gate_mean` (scripts/make_figures.py)
- [!] learning curves per-epoch — artifact chưa lưu (limitations §3.5)
