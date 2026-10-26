# Progress Log

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

**Bước tiếp:** push 010 lên Kaggle (chờ xác nhận user vì push chưa allow), poll → pull artifacts audit FULL (31.500 ảnh) → push 020 (cần output 010) → chốt compute_budget từ số đo T4 thật.

**Chưa có / cần:** số dư GPU quota thực tế (không có API — user có thể xem web UI Kaggle → Settings); xác nhận push.
