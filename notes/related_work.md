# Related Work — đã xác minh nguồn (2026-10-26)

Mọi citation dưới đây đã tra cứu nguồn gốc qua web (arXiv/IEEE/publisher). Mục "Nguy cơ trùng" ghi rõ điểm cần phân biệt.

## 1. NWPU-RESISC45 (dataset gốc)

- **Nguồn:** Cheng, Han, Lu, "Remote Sensing Image Scene Classification: Benchmark and State of the Art",Proceedings of the IEEE, 2017; preprint [arXiv:1703.00121](https://arxiv.org/abs/1703.00121) (~4.200+ citations).
- **Nội dung:** 31.500 ảnh, 45 lớp, 700 ảnh/lớp, RGB 256×256, ba mức GSD (0.2–30 m/px). Đã xác minh bản tải về đúng 45×700 = 31.500 ảnh.
- **Kế thừa:** toàn bộ dataset + protocol chia 75/10/15 stratified.
- **Chưa chứng minh:** khả năng tổng quát hóa địa lý (dataset không có metadata vị trí đủ dùng — không tuyên bố).

## 2. WTConv — Wavelet Convolutions for Large Receptive Fields

- **Nguồn:** Finder, Shaham, Treister, ECCV 2024, [arXiv:2407.05848](https://arxiv.org/abs/2407.05848) (~1.000+ citations). Repo chính thức: [BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv).
- **Ý tưởng cốt lõi:** thay depthwise conv bằng pyramid Haar wavelet: DWT → conv nhỏ trên từng subband (cả cascade tầng thấp) → inverse DWT. Receptive field lớn với tham số ít, cho response đa tần số.
- **Kế thừa:** dùng Haar DWT/IDWT (pywt hoặc tự viết, chỉ dùng tầng 1) như phép tách dải tần trên **feature map**.
- **Khác biệt dự kiến:** WTConv là *thay thế convolution* đặt trong backbone; ASF của chúng ta **không thay conv pretrained** — wavelet chỉ dùng để tách subband của feature map đã có, rồi gate theo ảnh. Không dùng cascade đa tầng.
- **Nguy cơ trùng:** thấp về cơ chế (không thay thế conv), nhưng phải trích dẫn WTConv là nguồn ý tưởng tách tần số bằng wavelet trong CNN.

## 3. Supervised Contrastive Learning (SupCon)

- **Nguồn:** Khosla et al., NeurIPS 2020, [arXiv:2004.11362](https://arxiv.org/abs/2004.11362). Code: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast).
- **Ý tưởng:** kéo embedding cùng lớp lại, đẩy khác lớp ra, tận label trong batch.
- **Vai trò với CMP:** tham chiếu nền tảng cho học embedding phân biệt; phiên bản đầu CMP **không** dùng contrastive loss (tránh chồng cơ chế); chỉ cân nhắc sau nếu ablation cho thấy prototype collapse.
- **Nguy cơ trùng:** không — chỉ là nền lý thuyết.

## 4. FSPA-Net (prior art gần nhất với ASF — cần phân biệt rõ)

- **Nguồn:** Wan et al., "FSPA-Net: Frequency–structural prior guided dual-stream dynamic fusion for remote sensing scene classification", Knowledge-Based Systems, 2026. [ScienceDirect S0950705126010695](https://www.sciencedirect.com/science/article/pii/S0950705126010695) (~2 citations tính 10/2026).
- **Ý tưởng:** hai stream (frequency prior + structural prior), module fusion động thích ứng theo ngữ cảnh tần số.
- **Điểm giống:** dùng prior tần số + fusion động cho RS scene classification.
- **Điểm khác dự kiến:** FSPA-Net là dual-stream với structural prior riêng; ASF là **một backbone pretrained duy nhất**, wavelet áp trên feature map nội tại ở 2 stage, gate softmax nhẹ theo ảnh, hợp nhất residual — giữ nguyên biểu diễn gốc, thêm phần bổ sung. Không có stream structural riêng.
- **Điều chưa chứng minh:** (a) chưa đọc full-text FSPA-Net (paywall) — chỉ đọc abstract/sản phẩm; nếu full text cho thấy cơ chế gate trùng, phải điều chỉnh cơ chế ASF và ghi lý do; (b) chưa chạy so sánh với FSPA-Net (ngoài phạm vi môn học, không có code công khai).

## 5. GDPNet (prior art gần nhất với CMP)

- **Nguồn:** Liu et al., "GDPNet: Gated Dynamic Prototype Network for Few-Shot Remote Sensing Scene Classification", IEEE (IEEE Xplore 11208651), 2025 (~4 citations).
- **Ý tưởng:** prototype động + gating cho **few-shot episode learning** trên RS scene.
- **Điểm khác dự kiến:** GDPNet giải few-shot (5-way 1/5-shot, metric-based episodic training); CMP của chúng ta giải **supervised full-data** (525 ảnh/lớp train), prototype cố định học cùng end-to-end, và đóng góp chính là **confusion graph từ train-only** điều chỉnh margin/loss. Bài toán, protocol, metric khác.
- **Nguyên họ liên quan khác:** Dynamic Prototype Convolution Network (Liu et al., 2022, arXiv) — DPCN đổi prototype theo batch trong few-shot; SPNet (Siamese-Prototype); S3OPT (2026, semi-supervised prototype tree). Tất cả đều ít nhiều ở setting few-shot/semi-supervised.

## 6. Nhận định tổng hợp

- Khoảng trống chúng ta nhắm: **cơ chế nhẹ gắn vào backbone pretrained phổ thông**, đánh giá có kiểm soát chi phí (RQ3) trên protocol supervised chuẩn — hai proposed mỗi cái một cơ chế chính, có ablation trực tiếp cùng backbone.
- Không tuyên bố "đầu tiên"/"SOTA". Mọi phát biểu trong báo cáo ở dạng "trong phạm vi thí nghiệm của chúng ta trên NWPU-RESISC45…".
- **Việc còn nợ:** đọc kỹ hơn phần method của FSPA-Net nếu tìm được bản hợp lệ (preprint/request) trước khi chốt thiết kế ASF; nếu không được, ghi vào limitations rằng phân biệt dựa trên abstract + mô tả publisher.

## Yêu cầu citation khi viết báo cáo

1. Cheng et al. 2017 (dataset) — Proceedings of the IEEE 105(10), 2017.
2. Finder et al. ECCV 2024 (WTConv).
3. Khosla et al. NeurIPS 2020 (SupCon, nền lý thuyết).
4. Wan et al. KBS 2026 (FSPA-Net — prior art cần phân biệt).
5. Liu et al. IEEE 2025 (GDPNet — prior art cần phân biệt).
6. torchvision/PyTorch (weights + implementation), theo version dùng thật trong run.
