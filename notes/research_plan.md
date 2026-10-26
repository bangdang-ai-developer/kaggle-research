# Research Plan — Phân loại cảnh ảnh viễn thám (NWPU-RESISC45)

**Tên làm việc:** Phân loại cảnh ảnh viễn thám bằng lựa chọn đặc trưng đa tỉ lệ và học phân biệt các lớp dễ nhầm lẫn
**Ngày lập:** 2026-10-26 · **Trạng thái:** ACTIVE · **Nhóm:** 2 thành viên (TV1: triển khai, TV2: review tài liệu)

## 1. Câu hỏi nghiên cứu

| RQ | Nội dung | Cách trả lời |
|---|---|---|
| RQ1 | Lựa chọn thông tin đa tỉ lệ và dải tần theo từng ảnh có cải thiện phân loại so với cùng backbone không? | So ASF-ResNet18 (A2) với ResNet18 (A0) và fusion cố định (A1) trên cùng split |
| RQ2 | Nhiều prototype/lớp + hướng dẫn từ cặp dễ nhầm có cải thiện ranh giới phân lớp không? | So CMP-ResNet18 (C3) với cosine 1 prototype (C1) và multi-prototype không graph (C2) |
| RQ3 | Lợi ích có tương xứng với chi phí huấn luyện, số tham số, suy luận? | Bảng ACC/F1 vs training time, params, latency cho mọi model |

## 2. Dataset

- **NWPU-RESISC45** (Cheng et al., arXiv:1703.00121): 45 lớp × 700 ảnh = 31.500 ảnh RGB 256×256.
- Nguồn Kaggle: `punfake/resisc45` (đã xác minh cấu trúc `NWPU-RESISC45/<class>/<class>_NNN.jpg`, size ~427MB trùng bản gốc). Chi tiết: `notes/dataset_card.md`.
- Split cố định 75/10/15 stratified: 525/70/105 ảnh mỗi lớp. Dup/near-dup giữ cùng split. Chi tiết protocol: `notes/dataset_card.md`.

## 3. Mô hình

### 3.1 Baselines (5)

| ID | Model | Weights |
|---|---|---|
| B0 | SmallCNN (3 block conv, train từ đầu) | — |
| B1 | ResNet18 | ImageNet (torchvision IMAGENET1K_V1) |
| B2 | MobileNetV3-Small | ImageNet (IMAGENET1K_V1) |
| B3 | EfficientNet-B0 | ImageNet (IMAGENET1K_V1) |
| B4 | ConvNeXt-Tiny | ImageNet (IMAGENET1K_V1) |

### 3.2 Proposed 1 — ASF-ResNet18 (Adaptive Scale–Frequency)

Giả thuyết: mức hữu ích của bố cục tổng thể vs chi tiết thay đổi theo ảnh; gate thích ứng theo ảnh trên các stage/dải tần tốt hơn fusion cố định.

Cơ chế: ResNet18 pretrained, feature maps stage2+stage3/4 → Haar DWT 2D trên feature map → LL/LH/HL/HH → chiếu 1×1 về cùng chiều → gate nhẹ (GAP+FC, softmax) sinh trọng số theo ảnh → hợp nhất residual với feature gốc → head CE.

Ràng buộc phiên bản đầu: không thay conv pretrained; nhánh mới khởi tạo zero (hoặc gần zero) để bắt đầu ≈ ResNet18 gốc; chỉ ablation A0–A3 (A0 gốc, A1 fusion cố định, A2 gate thích ứng, A3 + gate consistency nếu còn ngân sách). Gate consistency (2 biến đổi nhẹ giữ nhãn, L1 giữa 2 gate) là giả thuyết cần kiểm chứng, không ép khi blur mạnh.

### 3.3 Proposed 2 — CMP-ResNet18 (Confusion-guided Multi-Prototype)

Giả thuyết: mỗi lớp có nhiều dạng hình thức; K prototype/lớp + margin/trọng số loss theo cặp dễ nhầm (graph chỉ từ train) cải thiện ranh giới phân lớp.

Cơ chế: ResNet18 pretrained → embedding chuẩn hóa L2 → K prototype chuẩn hóa/lớp → score = log-sum-exp cosine (chuẩn hóa theo K, temperature cố định τ) → CE + confusion-aware term (chọn MỘT cơ chế: margin âm lớn hơn cho lớp dễ nhầm HOẶC trọng số loss cao hơn cho cặp khó — quyết định sau screening).

Confusion graph: probe model (train trên phần con của train) → OOF/confusion trên phần còn lại của train → graph cố định (đối xứng, có smoothing + cap) → train lại model cuối trên toàn bộ train. Không dùng val làm dữ liệu graph, không dùng test.

Ablation: C0 (classifier gốc) → C1 (cosine K=1) → C2 (K=2/4, không graph) → C3 (C2+graph) → C4 (+diversity reg nếu cần).

## 4. Protocol thực nghiệm

- **Tiêu chí chọn config/checkpoint:** validation macro-F1. Báo cáo đủ ACC, Precision/Recall/F1 macro, per-class, confusion matrix, training time, params (total/trainable), best epoch, latency.
- **Seeds:** final = 3 seed (0,1,2) cho 5 baseline + 2 proposed; ablation screening = 1 seed. Báo mean±std + paired diff theo seed. Nếu cắt giảm do quota → ghi rõ seed nào thiếu.
- **Test locked:** chỉ chạy sau khi lưu `notes/final_plan.json` (configs, seeds, split hash, checkpoint selection). Không tuning sau khi nhìn test.
- **Augmentation chung:** RandomResizedCrop(256, scale 0.7–1.0), HFlip, VFlip, Rot90; normalize theo mean/std ImageNet cho mọi model pretrained; B0 dùng cùng resize + normalize của chính nó (ghi rõ).
- **Optimizer chung:** AdamW, lr backbone 1e-4, head 1e-3, cosine schedule, weight decay 1e-4, batch 128, AMP. Tuning ngân sách ≤ 2 cấu hình lr cho mỗi proposed (ghi trước: {1e-3/1e-4 mặc định, 3e-4/3e-5 dự phòng}).
- **Epochs:** screening 8 (hoặc subset 50% train nếu cần), final 15 với early-stop patience 5 trên val macro-F1.

## 5. Robustness (bonus) — chạy 1 lần trên test ở giai đoạn E

Blur (Gaussian σ ∈ {1, 2}), JPEG (q ∈ {30, 60}), brightness (× {0.6, 1.4}) — mức chọn trên validation, áp cùng phép biến đổi cho mọi model, lưu tham số biến đổi, báo clean vs corrupted + delta.

## 6. Giai đoạn (chi tiết ngân sách: notes/compute_budget.md)

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| A | Local checks, data audit logic | IN PROGRESS |
| B | Kaggle pilot (data audit + đo thời gian + forward/backward proposed) | PENDING |
| C | Baseline + ablation screening | PENDING |
| D | Final training 3 seed | PENDING |
| E | Final eval, robustness, demo, handoff | PENDING |

## 7. Rủi ro đã nhận diện

1. **Prior art trùng:** FSPA-Net (KBS 2026) làm frequency+structural fusion cho RS scene classification → ASF phải khác: gate theo ảnh trên wavelet subbands của feature map (không phải dual-stream structural prior); chi tiết `notes/related_work.md`. GDPNet làm multi-prototype nhưng ở few-shot episode learning, khác supervised full-data.
2. **Quota GPU 30h/tuần:** chưa kiểm tra số dư thực tế (Kaggle không có API công khai cho GPU quota) → pilot phải đo s/epoch, tổng kế hoạch ≤ 30h, giữ 15–20% dự phòng.
3. **Wavelet trên feature map kích thước lẻ:** xử lý bằng DWT trên spatial size chẵn (pad nếu cần) — kiểm tra trong pilot.
4. **Multi-prototype collapse:** theo dõi usage stats + khoảng cách inter-prototype; có C4 nếu cần.
5. **Nếu proposed không thắng baseline:** ghi nhận negative result, phân tích bằng chứng (gate/prototype stats), đề xuất bước tiếp theo — không che giấu.
