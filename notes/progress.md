# Progress Log

## 2026-10-26 (chiều) — GIAI ĐOẠN B HOÀN THÀNH: pilot PASS trên T4

**Kết quả pilot (bằng chứng: `outputs/resisc45-pilot-v2-local-stage/pilot_report.json`):**
- 8/8 checks PASS: B1 epoch hợp lý (42–44s), ASF init ≡ ResNet18 (Δ=0.0), gate hoạt động, ASF residual học được (grad đúng lý thuyết: gate grad 0 ở step0 → >0 từ step1, Δw=158), CMP margin targeting chính xác (Δ=−0.24=−m·w, Δ nhãn đúng =0), prototype/embed/backbone đều có grad, latency đo được 7 model, chi phí val/ckpt đo được (4s / 42.8MB).
- Chất lượng sanity: ResNet18 val **acc 91.1% / macro-F1 91.2% sau 2 epoch** — hợp lý với RESISC45.
- Params: RN18 11.20M / ASF 11.60M (+3.6%) / CMP 11.49M (+2.5%).
- Prototype usage: 0 unused, inter-proto cos 0.29 (không collapse ở smoke).
- Compute budget chốt: **~8.7h mục tiêu + dự phòng → trần ~11h/30h tuần** — dư dả cho 3 seed mọi model.

**Bài học vận hành (đã ghi vào compute_budget.md):** stage local bắt buộc; device-check khi smoke CPU; 409 khi push đè; report phải pull/verify chứ không tin status.

**Bước tiếp — Giai đoạn C:** chờ duyệt push 030-baselines (screening B0–B4 8 epoch 1 seed) → 040/050 ablation screening → quyết định config final.

## 2026-10-26 — Giai đoạn A hoàn thành (local checks + docs + data audit logic)

**Đã xác minh:**
- Workspace: git clean @ 685db6f; scripts + template + smoke-test metadata chuẩn; settings.json **chưa allow push**.
- Kaggle credentials OK (doctor: token auth, CLI 2.2.5).
- Dataset `punfake/resisc45`: tải local 31.500 ảnh = 45 lớp × 700 ✓, cấu trúc chuẩn.
- Related work nguồn gốc đã xác minh: [RESISC45 arXiv:1703.00121](https://arxiv.org/abs/1703.00121), [WTConv arXiv:2407.05848 ECCV 2024](https://arxiv.org/abs/2407.05848), [SupCon arXiv:2004.11362 NeurIPS 2020](https://arxiv.org/abs/2004.11362), [FSPA-Net KBS 2026](https://www.sciencedirect.com/science/article/pii/S0950705126010695), [GDPNet IEEE 2025](https://ieeexplore.ieee.org/document/11208651).

**Đã xây:**
- Tài liệu: `notes/research_plan.md`, `related_work.md`, `dataset_card.md`, `compute_budget.md` (ước tính chờ pilot), `experiment_registry.csv`.
- `scripts/nbmake.py`: sinh notebook từ nguồn `nb.py`, hỗ trợ embed `src/models.py` (một nguồn logic).
- `src/models.py`: 5 baseline + ASF-ResNet18 + CMP-ResNet18 + train/eval/latency + confusion-graph builder.
- `notebooks/010-data-audit`: audit + split manifest 75/10/15 group-aware seed 42. Local SMOKE PASS (300 ảnh: 0 decode lỗi, 0 dup, split chuẩn, leakage PASS). Đã sửa bug OOM tiềm ẩn: pairwise Hamming bằng popcount LUT (unpackbits ~8GB ở full).
- `notebooks/020-pilot`: đo timing + sanity 2 proposed. Local CPU SMOKE PASS 5/5 checks (ASF init Δ=0 chính xác, gate var>0, CMP margin −0.24 đúng chiều, latency mọi model, s/epoch hợp lý).

**Bug đã sửa trong quá trình smoke:** (1) unpackbits OOM; (2) `make_loaders` nhận nhầm train-only → val rỗng; (3) `ConvNeXt_Tiny_Weights` tên enum; (4) DataLoader worker crash macOS → NUM_WORKERS=0 darwin / 4 linux; (5) init-check so nhầm resnet18 đã train (BN stats) → so fresh.

**Bước tiếp:** chờ 010 complete trên Kaggle → pull artifacts full audit → kiểm tra (decode lỗi, exact/near dup, nhóm, split, checksum) → review PNG near-dup → push 020 (đã được xác nhận) → chốt compute_budget từ số đo T4.

## 2026-10-26 — Push 010: 3 lỗi hạ tầng Kaggle đã xác định và sửa

User xác nhận phạm vi: push 010 + 020, theo dõi/pull/kiểm tra; giữ nguyên permissions; không push kernel huấn luyện.

**Lỗi lần 1** `No kernel name found`: papermill Kaggle cần `kernelspec` trong notebook metadata — nbmake.py không emit. **Fix:** nbmake.py thêm kernelspec python3 (áp dụng mọi notebook sau).
**Lỗi lần 2** `FAIL-FAST dataset root`: dataset không thấy ở path cũ. Diagnostics (lần 3) cho thấy **Kaggle 2026 mount dataset ở `/kaggle/input/datasets/<owner>/<slug>/`** thay vì `/kaggle/input/<slug>/`. **Fix:** resolve bằng `glob.glob("/kaggle/input/**/NWPU-RESISC45", recursive=True)` — không phụ thuộc layout mount. Thêm diagnostics in listing vào assert khi fail.
**Lỗi metadata:** Kaggle lấy slug kernel từ **title** (`resisc45-data-audit-split-manifest`), không theo `id` trong metadata — đã đổi id khớp slug thật (cả kernel_sources của 020).

**Bài học ghi lại:** kernel `complete` ≠ thành công — mọi lần ERROR phải pull log, xác định nguyên nhân trước khi push lại (đã giữ 3 log trong `outputs/resisc45-data-audit-split-manifest/`).

**020 nâng cấp theo yêu cầu review trước push:** (1) ASF gradient mechanics — bằng chứng trên T4: step0 out_proj grad > 0, gate grad = 0 (đúng lý thuyết zero-init), step2 gate grad > 0, trọng số đổi; (2) CMP margin targeting chính xác Δ = −m·w trên negative, Δ = 0 trên nhãn đúng (allclose 1e-6); (3) prototype/embedding/backbone đều nhận gradient; (4) đo thêm chi phí evaluate + checkpoint size + peak memory từng model. Local smoke PASS 8/8.
