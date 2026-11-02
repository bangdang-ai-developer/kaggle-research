# Limitations — hạn chế và những gì KHÔNG được tuyên bố

**Ngày:** 2026-10-28 · Đối chiếu `notes/results_summary.md` (số chính thức).

## 1. Về kết quả ASF

1. **Selection effect trên test (phát hiện v2):** bản công bố đầu "ASF thắng B1 +0.19% test" đo bằng seed tốt nhất theo val. Khi đánh giá đủ 3 seed trên test (final_plan_v2), ASF mean 0.9520 vs B1 0.9537 (paired 1/3 seed dương) → **lợi ích val (+0.39%, 3/3 seed) không khái quát sang test. Tuyên bố đúng: ASF ≈ B1 trên test; thắng rõ trên validation.**
2. **Độ lớn lợi ích nhỏ:** val +0.39% nằm gần biên nhiễu seed (std 0.002–0.004). Paired 3/3 + p≈0.05 hỗ trợ trên val, chỉ 3 seed.
3. **Chỉ trên 1 dataset:** mọi kết quả là NWPU-RESISC45. Không có EuroSAT/dataset thứ hai.
4. **So với baseline lớn:** ASF không vượt B3 (0.9607) và B4 (0.9677) trên test 3-seed — thắng rõ B2, không phải SOTA tổng thể.
5. **Gate sau entropy reg gần đều** (max_mean 0.14–0.19): chưa có bằng chứng gate chọn mạnh theo nội dung ảnh; cơ chế lợi ích chưa chứng minh nhân quả.
6. **Latency +52%** so B1 (3.82 vs 2.52 ms median, 3-seed mean) — chi phí cần tính nếu ứng dụng hạn chế.

## 2. Về kết quả CMP (chưa thắng)

1. **Negative result chính thức:** 7 cấu hình qua 2 vòng — v1: C1 0.9422 / C2 0.9441 / C3 0.9404 / C2fix 0.9398 / C3fix 0.9413 (screen 8ep); v2: CMP-P 0.9394 / CMP-PG 0.9416. Tốt nhất vẫn là C2 v1; test 3-seed 0.9509 vs B1 0.9537 (paired 0/3 seed dương).
2. **Prototype collapse bền vững** (inter-cos = 1.0 mọi seed v1). V2 hard-max đã phá collapse cơ học (inter-cos → 0.73) **nhưng F1 không tăng** (0.9394/0.9416) — 2 prototype tách nhau mà không cùng được dùng (minority ≈ 0) → vấn đề không nằm ở head prototype.
3. **Cơ chế collapse đầy đủ chưa được tái lập** trong synthetic (LSE/hard-max với embedding cố định đều tách nhau) — nghi parallel-drift qua LSE murette + embedding chuyển động; ghi "chưa đủ bằng chứng".
4. **Confusion margin v1 sai chiều đã xác nhận** (diagnostics D1: giảm CE mẫu khó, giảm gradient lớp khó) — v2 pair loss sửa chiều nhưng cũng không mang lợi ích F1.
5. **Trần soft-max tham số hóa CMP** (v3 kappa check, `notes/v3b_kappa_check.json`): logits = κ·cos bị chặn [−κ, κ] với κ học được 3.823 → trần soft-max student 13.32%/lớp (T=4); teacher soft targets median 0.5985 → **99.43% teacher targets vượt trần student** — KD loss không thể hội tụ. Hướng tiếp ưu tiên: tham số hóa scale kiểu ArcFace (κ = exp(θ), scale 30–64) trước khi thêm loss/kiến trúc mới.
6. Hướng tiếp khác (đề xuất, chưa chạy): prototype điều hòa (orthogonal init + hinge trực tiếp), hoặc chấp nhận cosine/FC head đơn giản cho task này.

## 2b. Về vòng v2 (diagnostics + 4 candidate)

- Diagnostics xác nhận: margin v1 sai chiều; collapse thật (không phải bug đo); signed-GAP triệt tiêu có căn cứ; residual branch KHÔNG chết (0.37) — bác bỏ nhận định v1 "gate collapse ≡ ResNet18".
- ASF-E (energy gate) vượt control fixed (+0.39%) nhưng không vượt GAP-gate v1; ASF-EA không giúp trong 8 epoch.
- Chi tiết: `notes/v2_diagnostics.json`, `notes/v2_plan.md`, reports 080/090.

## Changelog tài liệu (v2 — các sửa quan trọng so với bản công bố đầu)

1. §2 tách 2a/2b: bổ sung test 3-seed (21 checkpoints) + cảnh báo selection effect bảng 2a.
2. §3/§6: ASF "thắng B1 trên test" → "thắng trên val, ≈ B1 trên test (−0.16%, 1/3 seed)"; CMP giữ "chưa thắng".
3. Bỏ claim "gate collapse ≡ ResNet18" (residual/backbone đo được = 0.37 — diagnostics D4).
4. "Motion blur" → "Gaussian blur"; bỏ kết luận nhân quả "stride lớn" về điểm yếu EffB0 (chưa có thí nghiệm chứng minh).
5. "LL ≡ full feature map" giữ nguyên — khớp code (LL không phải DWT-average).
6. Bỏ citation counts trong related_work.md; bổ sung Sub-center ArcFace, SoftTriple, CosFace, FcaNet.
7. Tổng GPU hours phân biệt measured (runtime log) và estimated.
8. (v4 audit) Cột per-seed test trong bản nghiên cứu chi tiết viết tay sai (không khớp mean±std/paired diffs từ artifacts) — đã sửa lấy trực tiếp từ `test_3seed_report.json`; mean±std và paired diffs giữ nguyên (đúng).
9. (v4 audit) layer4 ResNet18 với input 256 là 8×8 (không phải 16×16 như chú thích cũ) — chỉ sai chú thích, logic code đúng; đã sửa docs + sơ đồ.
10. (v4 audit) Thống kê signed-GAP đo TRƯỚC projection trên subband thô — chưa đo tín hiệu sau proj/ReLU tại head; diễn cảm đã soften thành "gợi ý khảo sát energy descriptor".
11. (v4 audit) KD chạy đủ 15 epoch (không early-stop), train time thật ~0.39h/run vs CE early-stop ~0.20h/run — chênh lệch thời gian huấn luyện được ghi rõ trong diễn giải KD.

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
