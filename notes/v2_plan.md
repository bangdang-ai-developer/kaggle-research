# V2 Plan — vòng nghiên cứu mở rộng sau v1

**Ngày bắt đầu:** 2026-10-28 · **Ngân sách:** tối đa ~6h GPU · **Registry:** `notes/experiment_registry.csv` (cột stage = v2-*)
**Nguyên tắc:** test đã quan sát sau v1 → vòng này là *mở rộng*; không chọn cấu hình bằng test; mọi phát triển dùng train/val; khóa `final_plan_v2.json` trước đánh giá cuối; báo cáo giữ tách bạch v1/v2.

## 1. Chẩn đoán đã chạy (scripts/v2_diagnostics.py → notes/v2_diagnostics.json)

| ID | Nghi vấn | Kết quả | Hệ quả |
|---|---|---|---|
| D1 | Margin v1 chiều nào? | **XÁC NHẬN sai chiều**: GIẢM CE mẫu khó (−0.11), GIẢM gradient lớp khó (0.401→0.331) → là loss down-weighting, không phải CosFace margin | CMP-v2 dùng pair loss chiều rõ: `softplus(m + s_c − s_y)` |
| D2/D2b | Prototype collapse cơ chế? | **XÁC NHẬN THẬT** ở checkpoint (45/45 lớp pair-cos=1.0, euclid 0.00035; cross-class −0.023 → metric đúng). **Chưa tái lập** đầy đủ trong synthetic (LSE/hard-max κ=10 + E cố định → tách nhau −0.85…−0.92) → nghi parallel-drift qua κ murette + embedding động; ghi "chưa đủ bằng chứng" cho cơ chế | Đổi sang **hard-max** (chỉ winner nhận gradient — không parallel drift) + k-means init |
| D3 | K=2 thực chất? | K=2 ≡ K=1 (minority 0.11, 35/45 lớp 1 "dead") | Competitive mechanism bắt buộc |
| D4a | "Gate collapse ≡ B1"? | **BÁC BỎ cho final A2fix-ent**: residual/backbone = 0.37; max\|Δlogit vs B1\| = 8.50 | Sửa tài liệu v1; gate/residual = diagnostic chuẩn v2 |
| D4b | Signed-GAP triệt tiêu? | **CĂN CỨ XÁC NHẬN**: \|signed-GAP\| LL4=1.025 vs LH4=0.054 (19×); RMS chỉ 4.3× | ASF-E energy descriptor có căn cứ |

## 2. Candidates (tối đa 4) + controls

| Kernel | Candidate | Mô tả | Control |
|---|---|---|---|
| 080-cmp-v2 | **CMP-P** | K=2, **hard-max** logits (sub-center ArcFace style), k-means init, embedding L2-norm, κ học được clamp như v1 | C1 v1 (050: 0.9422 — cùng protocol 8ep/seed0, reuse hợp lệ) + B1 0.9504 (final) |
| 080-cmp-v2 | **CMP-PG** | CMP-P + pair loss `Σ w(y,c)·softplus(m + s_c − s_y)/Z`, s=κ·max_k cos, m=0.1, λ=0.5 warmup 3ep, Z=Σw, không self-edge | — |
| 090-asf-v2 | **ASF-E** | Gate input = energy descriptor [RMS, mean\|·\|] 16-dim → BN → Linear→8; giữ entropy 0.05; fusion branch giữ nguyên | A1fix-small (fixed fusion, cùng init) + ASF-v1 0.9543 + B1 0.9504 |
| 090-asf-v2 | **ASF-EA** | ASF-E + entropy anneal: λ(epoch)=0.05·max(0,1−epoch/8) | ASF-E |

Screening: 8 epoch, seed 0, staged local, seed 1 xác nhận nếu tín hiệu nhỏ. Tiêu chí vượt: val F1 > mốc tương ứng (CMP: >0.9441 C2-v1 screen; ASF: >0.9472 A2fix-ent screen) và diagnostics sạch (CMP: inter-cos < 0.95; ASF: max_mean < 0.9).

## 3. Final v2 (chỉ nếu screening có tín hiệu; mặc định 15 epoch recipe v1)

- Tối đa 1 CMP-v2 + 1 ASF-v2 × seeds {0,1,2} → `final_plan_v2.json` khóa trước.
- Nếu baseline cần chạy lại do sửa shared code → chạy đối xứng. Không tăng epoch riêng proposed.
- Test: đánh giá 1 lần, **báo cáo rõ test đã quan sát v1** — kết quả v2 là nghiên cứu mở rộng, không chọn bằng test.

## 4. Logging v2 (bắt buộc từ kernel 080+)

History per-epoch JSONL trong report: epoch, train_loss, **aux losses** (entropy/diversity/pair), lr, val acc/F1, sec. Mỗi run lưu `history` đầy đủ vào report (v1 chỉ lưu summary — đã ghi hạn chế).

## 5. Hướng继续 nếu candidate âm

Giữ v1 làm model nộp chính; v2 trình bày như nghiên cứu bổ sung + diagnostics. Không thử vô hạn: hết 4 candidate + 1 vòng fix nhẹ là dừng, hoàn thiện báo cáo.
