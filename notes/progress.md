# Progress Log

## 2026-11-16..17 (v4 Hướng A) — Sửa `κ = exp(θ)` cho CMP (`CMP-exp`) & Screening `ASF-Lite`

**Phát hiện gốc:** Trong `CMPResNet18`, `logit_scale` được khởi tạo `math.log(10.0)` nhưng lại thiếu `.exp()` khi tính `kappa`, khiến `κ ≤ 4.605` (tạo trần softmax `13.32%` dưới KD như `v3b_kappa_check.json` đã chỉ ra). Đồng thời thiết kế `ASFLiteResNet18` (`layer2+layer3` Haar DWT + RMS, bỏ 8 nhánh conv lặp).

**Screening seed 0 (kernel `120-h1a-screen` — `outputs/resisc45-h1a-screen-cmp-exp-asf-lite/h1a_screen_report.json`):**
- `cmp_exp_ce`: val F1 **0.9542** (+0.59% vs B1 s0 `0.9483`), `κ = 14.26`, `inter-cos = 0.553`
- `cmp_exp_kd`: val F1 **0.9561** (+0.11% vs RN18-KD s0 `0.9550`), `KD loss: 15.21 → 0.508` (hết kẹt `18.46`), `κ = 19.38`
- `asf_lite_ce` (`0.9504`) / `asf_lite_kd` (`0.9555`): không vượt `ASF v1 KD` (`0.9565`) và latency cao hơn (`4.07–4.28 ms`) → dừng ở screening theo `notes/final_plan_v4_cmp_exp.json`.

**Final 3-seed Val + TEST (`121-cmp-exp-final-eval` — `outputs/resisc45-cmp-exp-3-seeds-and-test-eval/`):**
- `CMP-exp-CE`: Val **0.9544 ± 0.0010** (3/3 thắng B1); **TEST 0.9549 acc / 0.9548 ± 0.0021 F1** — paired vs `CMP v1` **+0.39% (3/3)**, paired vs `B1` **+0.11% (2/3)**, same-machine latency **3.69 ms** vs ResNet18 `4.11 ms`.
- `CMP-exp-KD`: Val **0.9557 ± 0.0005**; **TEST 0.9573 acc / 0.9574 ± 0.0005 F1** — paired vs `CMP-exp-CE` **+0.26% (3/3)**, `std = 0.0005` thấp nhất nhóm KD (vẫn dưới `ResNet18-KD` `0.9601`).

---

## 2026-10-29 (v3) — KD round HOÀN TẤT: distillation cải thiện thật, nhất quán 3/3 seed trên test

**Pipeline:** diagnostics K1–K5 PASS → pilot 100b (teacher verify hash `9627ca43`, val acc 0.9692; KD 100–104s/epoch) → screening 110b (RN18-KD +0.67%, ASF-KD +0.58% vs CE s0; CMP-KD negative → dừng) → final 111 (seeds 1,2) → test eval 112c (6 checkpoints).

**Kết quả TEST 3-seed (`outputs/resisc45-eval-kd-3-seeds-v2-fixed-sources/test_kd_3seed_report.json`):**
- ResNet18-KD **0.9602±0.0020** (paired vs CE +0.65%, 3/3 seed)
- ASF-KD **0.9591±0.0014** (paired vs CE +0.70%, 3/3 seed)
- ASF-KD ≈ RN18-KD trên test (−0.10%); trên val ASF-KD +0.24% — lợi ích kiến trúc thu hẹp dưới KD, đóng góp chính là distillation
- Models nhỏ tốt nhất: ResNet18-KD gần B3 EffB0 (0.9608) với latency 1/3; B4 vẫn trần

**Số liệu đầy đủ:** `notes/results_summary.md` §V3. **Demo:** thêm B1-KD + ASF-KD (checkpoint chọn theo val — ASF-KD s1 0.9587).

**Bài học vận hành v3:** kernel có pretrained build → internet BẮT BUỘC true; checkpoint naming phải thống nhất (110b không suffix seed); smoke local phải chạy đủ tới cell kiểm tra model (không dừng sớm ở assert).

---

## 2026-10-28 (v2) — Vòng nghiên cứu mở rộng HOÀN TẤT

**Diagnostics** (`scripts/v2_diagnostics.py` → `notes/v2_diagnostics.json`):
- XÁC NHẬN: margin v1 sai chiều (giảm CE/gradient trên mẫu khó — không phải CosFace); prototype collapse thật 45/45 lớp (không phải bug đo — cross-class cos −0.02 khác hẳn); signed-GAP triệt tiêu 19× (căn cứ energy descriptor); residual branch KHÔNG chết (0.37) — bác bỏ claim v1 "≡B1".
- KHÔNG tái lập được cơ chế collapse đầy đủ trong synthetic — ghi minh bạch.

**Screening v2:** CMP-P 0.9394 / CMP-PG 0.9416 (hard-max phá collapse inter-cos 1.0→0.73/0.74 — xác nhận chẩn đoán — nhưng F1 không tăng; minority≈0); ASF-E 0.9464 (vượt control fixed +0.39%, không vượt GAP-gate v1 0.9472) / ASF-EA 0.9454. → KHÔNG candidate nào vượt v1 → **giữ v1 làm model chính** (`notes/final_plan_v2.json` khóa).

**Expanded test eval (071b):** 21/21 checkpoints × test — phát hiện **selection effect**: ASF test 3-seed 0.9520±0.0026 vs B1 0.9537±0.0019 (paired 1/3 seed dương) → lợi ích val không khái quát sang test; bảng best-val-seed cũ (ASF 0.9550 > B1 0.9531) giữ như lịch sử với cảnh báo selection. Cả 2 proposed vẫn thắng B2 rõ.

**Demo:** `demo/app.py` (Streamlit) — B1/ASF/CMP, top-3, model-only vs e2e latency, 7 corruption options, cache models, offline.

**GPU v2:** ~2h (diagnostics local 0 + 080 0.5 + 090 0.6 + 071b 0.5 + 071v1 fail ~0.2 trước cancel) — tổng project ~12h/30h tuần.

**Còn lại (TV2):** 2 sơ đồ kiến trúc; viết report/slides theo outline; soát reviewer_checklist.

---

## 2026-10-28 — DỰ ÁN HOÀN TẤT: toàn pipeline + tài liệu bàn giao

**Kết quả chính thức (TEST, 4.725 ảnh, 1 lần duy nhất):** ASF **0.9549 acc / 0.9550 F1 — THẮNG backbone gốc B1 (0.9530/0.9531, +0.19%)** và B2; nhất quán val paired 3/3 seed (+0.39% mean, p≈0.05). CMP 0.9515/0.9515 — hòa B1, **chưa thắng** (negative result: prototype collapse inter-cos=1.0 bền vững qua 5 cấu hình). B4 ConvNeXt-T 0.9678 trần (đắt 3.8×).

**GPU thực dùng:** ~10h/30h tuần (060a 5.0 + 060b 1.4 + screening ~2.5 + pilot ~1.2 + 070 ~0.7) — dưới trần 11h đã chốt. CPU: audit ×8 phiên ≈ 3h (miễn phí).

**Bộ tài liệu bàn giao:** results_summary.md (số chính thức), report_outline.md, slides_outline.md, reviewer_checklist.md, limitations.md, reproduction.md, compute_budget.md (actual), figures ×6 (val bar, F1-latency, robustness delta, gate distribution, class dist, samples).

**Việc còn mở cho TV2:** vẽ 2 sơ đồ kiến trúc (ASF/CMP); viết report + slides theo outline; quyết định demo app (mục 14 spec — chưa làm); soát reviewer_checklist.md.

---

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
