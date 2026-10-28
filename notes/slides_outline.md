# Slides Outline — ~15 phút thuyết trình (~18 slides)

**Nguyên tắc:** mỗi slide 1 thông điệp; số từ results_summary.md; có phụ lục negative results (điểm cộng với giảng viên).

1. **Title** — đề tài + tên 2 thành viên + phạm vi (NWPU-RESISC45).
2. **Bài toán & động cơ** — 45 cảnh viễn thám; vì sao chọn backbone phổ thông + module nhẹ (chi phí).
3. **Dataset** — 31.500 ảnh 45×700; audit (0 lỗi, dup handling, split chống leakage) — 1 hình class distribution.
4. **Tổng quan 2 proposed** — ASF (scale–frequency gate) + CMP (multi-prototype + confusion) trên cùng backbone B1.
5. **ASF: kiến trúc** — sơ đồ 8 thành phần Haar × 2 stage → gate → residual; entropy reg (bài học từ collapse).
6. **ASF: gate collapse & fix** — negative rồi fix: minh bạch quá trình (slide điểm cộng).
7. **CMP: kiến trúc** — embedding → K prototype → LSE cosine; confusion graph từ train-only (8 cặp đúng ngữ nghĩa).
8. **Protocol thực nghiệm** — split 75/10/15 chống leakage; 15 epoch early-stop; 3 seeds; test khóa 1 lần.
9. **Kết quả chính — bảng TEST** — 7 model; highlight ASF 0.9550 > B1 0.9531; B4 0.9678 trần.
10. **ASF thắng đối chứng** — paired 3/3 seed +0.24/+0.38/+0.54%; TEST +0.19%.
11. **RQ3 chi phí** — F1 vs latency scatter; ASF +3.6% params +45% latency; EffB0/B4 đắt hơn.
12. **CMP: negative result** — hòa B1; prototype collapse inter-cos=1.0 mọi seed/fix — trình bày như bài học thiết kế.
13. **Ablation ASF** — A1/A2/A2conv/A2fix-ent: entropy reg là nhân tố quyết định gate sống.
14. **Robustness** — ASF ≥ B1 5/6 corruption; EffB0 sụp blur σ2 −34%; B4 robust nhất.
15. **Lỗi điển hình** — 2–3 ảnh sai kèm cặp dễ nhầm ngữ nghĩa.
16. **Hạn chế** — 1 dataset; lợi ích +0.2% gần biên; CMP chưa đạt; synthetic corruption.
17. **Kết luận & hướng tiếp** — ASF giữ; CMP: hinge margin + per-prototype gating; EuroSAT second dataset.
18. **Phụ lục** — bảng đầy đủ + reproduction (repo + kernel Kaggle).

## Phân công trình bày (đề xuất)
- TV1: slides 3–8 (pipeline + kiến trúc) + 13 (ablation).
- TV2: slides 1–2, 9–12 (kết quả + negative), 14–17.
