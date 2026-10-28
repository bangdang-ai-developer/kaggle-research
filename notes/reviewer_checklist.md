# Reviewer Checklist — cho TV2 kiểm tra trước khi nộp

Mỗi mục cần trả lời ✅/❌/⚠️ + ghi chú. Nguồn số: `notes/results_summary.md`; artifacts: `outputs/`; nhật ký: `notes/experiment_registry.csv`, `notes/progress.md`.

## A. Số liệu

- [ ] Mọi số trong báo cáo/slides khớp `results_summary.md` (val §1, test §2, robustness §4, ablation §5)
- [ ] Không lẫn số SCREENING (8 epoch, seed 0) vào bảng FINAL (15 epoch, 3 seed)
- [ ] ASF: "thắng B1" có kèm bằng chứng (TEST +0.19% + paired 3/3 seed +0.39% mean)
- [ ] CMP: ghi là "hòa B1 / chưa thắng" — KHÔNG viết "cải thiện"
- [ ] Không có tuyên bố "SOTA"/"đầu tiên" — chỉ "trong phạm vi thí nghiệm trên NWPU-RESISC45"
- [ ] B4/B3 cao hơn ASF được nêu rõ (ASF không phải mô hình tốt nhất tổng thể)
- [ ] ±0.2% lợi ích ASF có kèm cảnh báo biên nhiễu (std seed 0.002–0.004)

## B. Phương pháp

- [ ] Split 75/10/15 group-aware, chống leakage (4 nhóm cross-class co-split) được mô tả
- [ ] Confusion graph CMP ghi rõ train-only (90/10 fit/graph), không dùng val/test
- [ ] Checkpoint selection = val macro-F1; test chạy 1 lần sau final_plan.json
- [ ] Entropy regularizer λ=0.05 + out_proj init std=1e-3 được nêu trong công thức ASF
- [ ] Nguồn pretrained weights: torchvision IMAGENET1K_V1 (không có checkpoint fine-tune sẵn trên RESISC45)

## C. Negative results (phải hiện diện — tiêu chí "không che giấu")

- [ ] Gate collapse v1 (mean=[0,0,0,0,1,0,0,0], std=0) và fix entropy
- [ ] Prototype collapse inter-cos=1.0 bền vững qua 5 cấu hình CMP
- [ ] Confusion margin 0.3 gây hại (−0.44%), 0.1 trung tính
- [ ] C1 cosine head đơn thuần thua FC gốc (−0.26%)

## D. Hình

- [ ] Confusion matrix ASF + B1 (files: outputs/resisc45-final-evaluation-v2-offline/confusion_*.png)
- [ ] Robustness delta chart
- [ ] Sơ đồ kiến trúc ASF/CMP (cần vẽ — TV2)
- [ ] Learning curves — nếu dùng, phải ghi chú nguồn (kernel log) hoặc bỏ (artifact không lưu per-epoch)
- [ ] Không dùng ảnh validation/test trong "gallery minh họa train"

## E. Tuyên bố ngôn ngữ

- [ ] "Robust hơn" chỉ kèm số cụ thể, không tuyên bốdeployment
- [ ] Softmax scores không gọi là "confidence calibrated"
- [ ] Gate/prototype visualization không được dùng làm bằng chứng nhân quả
- [ ] Mỗi citation kiểm tra nguồn (related_work.md)

## F. Bàn giao

- [ ] reproduction.md chạy được từ đầu (đã test thứ tự kernel)
- [ ] experiment_registry.csv khớp lịch sử (9 kernel thành công + các fail có nguyên nhân)
- [ ] Checkpoint + artifacts nằm trong outputs/ (không bắt buộc commit — .gitignore)
- [ ] Demo app: CHƯA làm — quyết định có làm trước hạn nộp không (spec mục 14, được cộng điểm)
