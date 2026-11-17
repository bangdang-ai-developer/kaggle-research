#!/usr/bin/env python3
"""Tạo báo cáo DOCX cuối môn từ results_master.json (nguồn số duy nhất).

Chạy: uv run python3 scripts/make_report_docx.py
Output: deliverables/Bao_cao_cuoi_mon_NWPU_RESISC45.docx
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "notes" / "figures"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)
M = json.load(open(OUT / "results_master.json"))
MODELS = M["models"]
FIGDIR = str(FIG)

doc = Document()
# base style
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)

H1 = lambda t: doc.add_heading(t, level=1)
H2 = lambda t: doc.add_heading(t, level=2)
P = lambda t: doc.add_paragraph(t)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_ms(d: dict) -> str:
    return f"{d['mean']:.4f} ± {d['std']:.4f}"


def table(headers: list[str], rows: list[list[str]], widths=None) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)


def figure(name: str, caption: str, width: float = 15.5) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / name), width=Cm(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def mrow(key: str) -> list[str]:
    e = MODELS[key]
    return [e["display"], f"{e['params_total_m']:.2f}",
            fmt_ms(e["val_f1_3seed"]), fmt_ms(e["test_3seed_acc"]), fmt_ms(e["test_3seed_f1"]),
            fmt_ms(e["test_3seed_precision"]), fmt_ms(e["test_3seed_recall"]),
            f"{e['latency_ms_bestvalseed']:.2f}" if "latency_ms_bestvalseed" in e else f"{e['latency_ms_3seed_mean']:.2f}",
            f"{e['train_time_h_mean']:.2f}"]


HDR = ["Model", "Params (M)", "Val F1", "Test Acc", "Test F1", "P macro", "R macro", "Lat ms", "Train h"]
W = [4.2, 1.6, 2.0, 2.0, 2.0, 1.8, 1.8, 1.5, 1.4]

# ================= TRANG BÌA =================
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("[Điền: Tên trường / viện]")
r.font.size = Pt(13)
t2 = doc.add_paragraph()
t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t2.add_run("[Điền: Tên môn học — mã môn] · Học kỳ [Điền]")
r.font.size = Pt(12)

tt = doc.add_paragraph()
tt.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tt.add_run("\nPhân loại cảnh ảnh viễn thám NWPU-RESISC45 bằng lựa chọn đặc trưng đa tỉ lệ, "
               "học phân biệt các lớp dễ nhầm lẫn và knowledge distillation\n")
r.font.size = Pt(17)
r.font.bold = True
tt2 = doc.add_paragraph()
tt2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tt2.add_run("Báo cáo cuối môn — nhóm 2 thành viên\n")
r.font.size = Pt(13)
mem = doc.add_paragraph()
mem.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = mem.add_run("Thành viên 1 [Điền tên — MSSV]: toàn bộ code, pipeline, huấn luyện và thực nghiệm\n"
                "Thành viên 2 [Điền tên — MSSV]: review tài liệu, báo cáo và slides\n")
r.font.size = Pt(11)
dt = doc.add_paragraph()
dt.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = dt.add_run("Ngày hoàn thành thực nghiệm: 29/10/2026 · Repo: kaggle-research (commit [Điền final commit])")
r.font.size = Pt(10)

# ================= TÓM TẮT =================
H1("Tóm tắt")
P("Báo cáo trình bày bốn vòng nghiên cứu (v1, v2, v3, v4 Hướng A) về phân loại 45 cảnh ảnh viễn thám trên bộ "
  "NWPU-RESISC45 (31.500 ảnh, 256×256), với hai kiến trúc đề xuất gắn trên backbone ResNet18 pretrained: "
  "ASF (lựa chọn đặc trưng đa tỉ lệ – dải tần thích ứng theo ảnh bằng Haar wavelet và gate softmax) và "
  "CMP (bộ phân loại cosine đa prototype có hướng dẫn từ đồ thị nhầm lẫn chỉ xây trên tập train). "
  "Năm baseline gồm B0 SmallCNN huấn luyện từ đầu và B1–B4 (ResNet18, MobileNetV3-S, EfficientNet-B0, "
  "ConvNeXt-T) ImageNet-pretrained.")
P("Vòng v1 thiết lập protocol chống rò rỉ dữ liệu (split 75/10/15 giữ nguyên nhóm ảnh trùng, seed cố định, "
  "manifest sha256 khóa cho toàn dự án), huấn luyện đủ 3 seed cho 7 mô hình và đánh giá test một lần duy nhất. "
  "Vòng v2 thực hiện chẩn đoán bằng số trên checkpoint thật: xác nhận margin nhầm lẫn của v1 sai chiều "
  "(giảm loss trên mẫu khó), xác nhận sự sụp đổ prototype là hiện tượng thật (45/45 lớp có cặp prototype "
  "trùng tuyệt đối), bác bỏ nhận định \"nhánh gate tắt tương đương ResNet18\", và thử 4 cấu hình sửa chữa. "
  "Vòng v3 áp dụng knowledge distillation từ ConvNeXt-T (teacher frozen, T=4, α=0.5): ResNet18-KD đạt 96.02% "
  "độ chính xác test (+0.65 điểm phần trăm so với CE, nhất quán 3/3 seed) và ASF-KD đạt 95.91% (+0.70 điểm, 3/3 seed).")
P("Vòng v4 (Hướng A) khắc phục dứt điểm hai điểm nghẽn đã chẩn đoán ở CMP: lỗi quên hàm mũ .exp() trên "
  "logit_scale (khiến κ bị chặn ≤ 4.605 và tạo trần softmax 13.32% dưới KD). Khi chuyển sang κ = exp(θ) ∈ [10, 64] "
  "kết hợp hard-max và k-means init (CMP-exp), CMP-exp-CE đạt 95.48% F1 test 3-seed — vượt cả CMP v1 (+0.39 điểm, "
  "3/3 seed) lẫn backbone gốc B1 ResNet18-CE (95.37%, +0.11 điểm, 2/3 seed test và 3/3 seed val 95.44% vs 95.04%) "
  "mà không làm chậm suy luận (3.69 ms vs 4.11 ms ResNet18 cùng máy). Dưới KD, CMP-exp-KD hội tụ sạch (KD loss "
  "15.21 → 0.51) và đạt 95.74% ± 0.05% F1 test (ổn định cao nhất nhóm KD). Toàn bộ thí nghiệm chạy trên Kaggle GPU "
  "(Tesla T4, tổng ~19h), mọi số liệu truy ngược được về artifact nguồn.")

# ================= 1-2 =================
H1("1. Bài toán, yêu cầu và mục tiêu")
P("Phân loại cảnh ảnh viễn thám là bài toán phân loại đa lớp trên ảnh chụp từ trên cao, nơi nhiều cặp lớp "
  "khó phân biệt chỉ khác nhau về bố cục không gian (ví dụ church/palace, dense/medium residential, "
  "circular/rectangular farmland). Yêu cầu môn học: hai mô hình tự chế mỗi mô hình gắn với một hướng cải "
  "tiến, tối thiểu năm baseline, mỗi mô hình đề xuất phải thắng ít nhất một baseline về độ chính xác, "
  "báo cáo đầy đủ các chỉ số và chi phí, giải thích căn cứ thiết kế, và đánh giá trung thực cả kết quả "
  "không thuận lợi.")
P("Mục tiêu nghiên cứu được cụ thể hóa thành ba câu hỏi: RQ1 — lựa chọn thông tin đa tỉ lệ/dải tần theo "
  "từng ảnh (ASF) có cải thiện so với backbone gốc không; RQ2 — đa prototype kèm hướng dẫn từ các cặp lớp "
  "dễ nhầm (CMP) có cải thiện ranh giới phân lớp không; RQ3 — lợi ích (nếu có) có tương xứng với chi phí "
  "tham số, thời gian huấn luyện và suy luận không.")

# ================= 3 RELATED WORK =================
H1("2. Công trình liên quan (đã kiểm tra nguồn)")
rows = [
    ("NWPU-RESISC45 — Cheng et al., Proceedings of the IEEE, 2017 (arXiv:1703.00121)", "Dataset gốc 31.500 ảnh/45 lớp; protocol tham chiếu"),
    ("Wavelet Convolutions for Large Receptive Fields — Finder et al., ECCV 2024 (arXiv:2407.05848)", "Nguồn ý tưởng tách dải tần Haar trong CNN; khác biệt: chúng ta không thay convolution, chỉ tách đặc trưng cho gate"),
    ("Supervised Contrastive Learning — Khosla et al., NeurIPS 2020 (arXiv:2004.11362)", "Nền lý thuyết embedding phân biệt; vòng đầu không dùng contrastive"),
    ("CosFace — Wang et al., CVPR 2018", "Kỹ thuật margin cosine; v3 pair loss kế thừa tư tưởng margin có hướng"),
    ("Sub-center ArcFace — An et al., ECCV 2020", "Nguồn cơ chế hard-max đa sub-center cho CMP-P (v2)"),
    ("SoftTriple Loss — Qian et al., ICCV 2019", "Tham chiếu đa trung tâm có lực phân tán; đối chiếu thiết kế CMP"),
    ("FcaNet — Qin et al., ICCV 2021", "Căn cứ giới hạn GAP và biểu diễn tần số → energy descriptor cho gate (ASF-E, v2)"),
    ("Knowledge Distillation: A Good Teacher Is Patient and Consistent — Beyer et al., CVPR 2022", "Tham chiếu quy trình KD không cần teacher-heavy tuning"),
    ("Decoupled Knowledge Distillation — Zhao et al., CVPR 2022", "Tham chiếu tách target/non-target term (không dùng trong vòng đầu)"),
    ("Knowledge Review — Chen et al., CVPR 2021", "Feature distillation — nhánh mở rộng tiềm năng, chưa triển khai"),
    ("FSPA-Net — Wan et al., Knowledge-Based Systems 2026", "Prior art gần nhất với ASF (dual-stream frequency-structural); khác biệt: ASF một backbone duy nhất + gate trên subband"),
    ("GDPNet — Liu et al., IEEE 2025", "Prior art prototype cho few-shot RS; khác phạm vi với CMP supervised full-data"),
]
table(["Công trình (nguồn đã kiểm tra)", "Quan hệ với bài của nhóm"], [[a, b] for a, b in rows], [8.5, 8.0])
P("Kỹ thuật kế thừa (margin cosine, sub-center hard-max, KD chuẩn) được trích dẫn nguồn gốc; phần thiết kế "
  "của nhóm là cách ghép Haar-subband gate vào backbone nguyên vẹn, entropy regularizer chống sụp gate, "
  "confusion graph train-only, và quy trình chẩn đoán bằng số trên checkpoint thật.")

# ================= 4 DATASET =================
H1("3. Dataset, audit, split và giới hạn dữ liệu")
P("Nguồn: Kaggle dataset punfake/resisc45 (bản sao NWPU-RESISC45 gốc, license apache-2.0). Audit toàn bộ "
  "31.500 ảnh: 0 lỗi decode, 100% kích thước 256×256 RGB, đủ 45 lớp × 700 ảnh. Phát hiện trùng lặp: 8 nhóm "
  "trùng tuyệt đối (MD5 identical) và 39 cặp gần trùng theo dHash (Hamming ≤ 6), trong đó 16 cặp là mutual "
  "nearest-neighbor; 4 nhóm chéo lớp được giữ cùng split để loại trừ rủi ro rò rỉ dù kiểm tra pixel cho thấy "
  "chỉ tương tự về cấu trúc, không phải trùng thật.")
P("Split 75/10/15 phân tầng theo lớp, seed 42, xử lý theo nhóm trùng: train 23.626 / validation 3.149 / "
  "test 4.725 ảnh; mỗi lớp 524–526/69–70/104–107. Manifest (image_id, lớp, nhóm trùng, split) được khóa bằng "
  "SHA256 cc70020a… và dùng chung cho mọi mô hình ở mọi vòng. Giới hạn dữ liệu: không có metadata vị trí địa "
  "lý nên không tuyên bố khả năng tổng quát hóa theo không gian địa lý; tập test đã được quan sát qua các vòng "
  "v1/v2 nên vòng v3 được công bố là nghiên cứu mở rộng.")

# ================= 5 BASELINES =================
H1("4. Năm baseline")
P("B0 SmallCNN (3 khối conv, ~0.29M tham số) huấn luyện từ đầu; B1 ResNet18 (11.20M); B2 MobileNetV3-Small "
  "(1.56M); B3 EfficientNet-B0 (4.07M); B4 ConvNeXt-T (27.85M) — B1–B4 ImageNet-pretrained. Tất cả dùng chung "
  "augmentation (RandomResizedCrop scale 0.7–1.0, horizontal/vertical flip), chuẩn hóa ImageNet, AdamW "
  "(lr backbone 1e-4, head 1e-3, weight decay 1e-4, cosine + warmup 5%), batch 128, AMP, tối đa 15 epoch, "
  "early-stop patience 5, checkpoint chọn theo validation macro-F1.")

# ================= 6 ASF =================
H1("5. Kiến trúc ASF (Adaptive Scale–Frequency ResNet18)")
P("Căn cứ thiết kế: đặc trưng tổng thể (low-frequency) và chi tiết (high-frequency) không luôn hữu ích như "
  "nhau cho từng ảnh; một cơ chế chọn nhẹ theo ảnh trên các dải tần của feature map nội tại có thể cải thiện "
  "mà không phá vỡ biểu diễn pretrained.")
figure("fig_asf_arch.png", "Hình 1. Kiến trúc ASF-ResNet18 (phiên bản A2fix-ent dùng cho kết quả chính).")
P("Cấu trúc: feature maps ở layer2 (128 kênh, 32×32) và layer4 (512 kênh, 8×8) được tách Haar wavelet mức 1 "
  "thành 8 thành phần (mỗi stage gồm ORIG ≡ feature map gốc và 3 subband chi tiết LH/HL/HH). Mỗi thành phần qua "
  "projection 1×1 → BN → ReLU → GAP cho vector f_i (128-d). Gate MLP (1024→8) softmax sinh trọng số w_i theo "
  "từng ảnh; vector hợp nhất Σ w_i·f_i qua Linear khởi tạo nhỏ (std 1e-3) cộng residual vào đặc trưng backbone "
  "GAP(layer4) trước head phân loại 45 lớp. Entropy regularizer λ=0.05 trên w (chuẩn hóa log K) giữ gate khỏi "
  "sụp 1-hot — bổ sung sau khi chẩn đoán v1 cho thấy gate sụp hoàn toàn về thành phần LL4.")
P("Diagnostics v2 trên checkpoint thật: nhánh residual đóng góp ~37% độ lớn đặc trưng đầu head (không phải "
  "nhánh chết); sau entropy reg, gate phân bố mềm (thành phần lớn nhất 0.14–0.19 so đều 0.125) và có biến "
  "động theo ảnh. Lợi ích kiến trúc thể hiện rõ trên validation (paired 3/3 seed) nhưng thu hẹp trên test khi "
  "kết hợp distillation (mục 9).")

# ================= 7 CMP =================
H1("6. Kiến trúc CMP (Confusion-guided Multi-Prototype)")
P("Căn cứ thiết kế: các lớp dễ nhầm có nhiều dạng hình thức; nhiều prototype mỗi lớp kèm hướng dẫn từ đồ thị "
  "nhầm lẫn có thể tạo ranh giới tốt hơn đầu cosine đơn giản.")
figure("fig_cmp_arch.png", "Hình 2. CMP-ResNet18 (C2 — cấu hình đưa vào kết quả chính;CMP-P/PG là biến thể v2).")
P("Cấu trúc: embedding Linear 512-d + L2 normalize; K prototype mỗi lớp (L2 normalize); logits = κ · max_k "
  "cos(e, p_k) (hard-max, v2) hoặc log-sum-exp (v1); κ là logit_scale học được. Đồ thị nhầm lẫn xây chỉ từ "
  "train bằng probe model (chia 90/10 fit/graph, 6 epoch, không dùng validation/test), 8 cặp đối xứng đúng ngữ "
  "nghĩa (church↔palace, circular↔rectangular farmland, dense↔medium residential, wetland↔lake…).")
P("Kết quả các vòng: v1 LSE-logits bị sụp prototype (cos giữa 2 prototype cùng lớp = 1.00000000 ở 45/45 lớp); "
  "v2 hard-max phá sụp (inter-cos 0.73) nhưng F1 không tăng (K=2 thực chất vẫn hoạt động như K=1); confusion "
  "margin v1 sai chiều (giảm loss trên mẫu khó — đã chứng minh bằng số trong diagnostics D1) và gây hại ở mức "
  "0.3. CMP giữ cấu hình C2 cho kết quả chính; toàn bộ 7 cấu hình thử đều không vượt B1.")

# ================= 8 KD =================
H1("7. Distillation và teacher (v3)")
figure("fig_kd_arch.png", "Hình 3. Sơ đồ KD: teacher frozen chỉ dùng lúc training; student inference độc lập.")
P("Teacher: ConvNeXt-T seed 2 — chọn theo validation macro-F1 (0.9691) trong 3 seed của 060a, không theo test; "
  "checkpoint hash SHA256 9627ca43… được xác minh khi load; frozen eval mode, không nhận gradient, running "
  "statistics không đổi (kiểm chứng K5). Teacher và student nhận cùng tensor x sau augmentation (same-view); "
  "không cache logits.")
P("Loss: L = (1 − α)·CE(student, y) + α·T²·KL(softmax(z_t/T) ‖ log_softmax(z_s/T)) với α = 0.5, T = 4. Năm kiểm "
  "chứng bắt buộc đều đạt: chiều KL đúng (khớp tính tay), teacher không nhận gradient, hệ số T² có mặt, α=0 "
  "trả về đúng CE, teacher weights + BN running stats bất biến qua vòng train thật.")
P("Chi phí: teacher inference online làm thời gian mỗi epoch tăng từ ~49s lên ~100s (hệ số ~2×); chi phí huấn "
  "luyện teacher nằm ở v1 (0.62h/run) ghi riêng; student inference sau KD không cần teacher, latency không đổi.")

# ================= 9 PROTOCOL =================
H1("8. Protocol từng vòng")
rows = [
    ("v1", "Audit + split chuẩn; final 7 mô hình × 3 seed × 15 epoch; test 1 lần (best-val seed); robustness 6 corruption", "Test best-val-seed có selection effect — được lộ ra ở vòng v2"),
    ("v2", "Diagnostics bằng số trên checkpoint; 4 candidate sửa lỗi cơ học (hard-max, energy gate, entropy anneal); mở rộng test đủ 3 seed × 7 mô hình", "Không candidate vượt cấu hình v1 → giữ v1; phát hiện selection effect"),
    ("v3", "KD từ ConvNeXt-T (T=4, α=0.5); screening seed 0; final seeds 1–2; test đủ 3 seed cho KD models", "RN18-KD đạt 96.02% test; phát hiện CMP-KD bị kẹt trần softmax 13.32%"),
    ("v4 (Hướng A)", "Sửa CMP logit_scale sang κ = exp(θ) (CMP-exp) + thử ASF-Lite; screening s0 → đưa CMP-exp lên 3 seed + TEST eval", "CMP-exp-CE đạt 95.48% F1 test (vượt CMP v1 & B1); CMP-exp-KD đạt 95.74%±0.05%"),
]
table(["Vòng", "Nội dung", "Kết quả chính"], [[a, b, c] for a, b, c in rows], [1.5, 9.5, 5.5])
P("Quy tắc chung mọi vòng: split manifest và class order không đổi; checkpoint selection chỉ theo validation "
  "macro-F1; test không dùng để chọn cấu hình; mỗi run có config đầy đủ lưu trong checkpoint; thất bại hạ tầng "
  "(DNS khi tải pretrained, 409 khi push đè, checkpoint thiếu kernel source) được ghi trong registry với nguyên nhân.")

# ================= 10 KET QUA =================
H1("9. Kết quả chính")
H2("9.1. Validation (3 seeds, mean ± std)")
table(["Model", "Val F1", "Val Acc"],
      [[MODELS[k]["display"], fmt_ms(MODELS[k]["val_f1_3seed"]), fmt_ms(MODELS[k]["val_acc_3seed"])]
       for k in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt",
                 "ASF_A2fix_ent", "CMP_C2", "cmp_exp_ce", "resnet18_kd", "asf_kd", "cmp_exp_kd")],
      [6.0, 4.0, 4.0])
H2("9.2. Test — bảng tổng hợp đầy đủ (đủ 3 seed)")
H2 = doc.add_heading("Ghi chú: Std là độ lệch chuẩn giữa training seeds (không phải giữa ảnh). "
                     "Bảng 9.2 là đánh giá mở rộng công bằng; bảng best-val-seed ở phụ lục A.", level=3)
table(HDR,
      [mrow(k) for k in ("smallcnn", "resnet18", "resnet18_kd", "mobilenetv3s", "efficientnetb0",
                         "convnextt", "ASF_A2fix_ent", "asf_kd", "CMP_C2", "cmp_exp_ce", "cmp_exp_kd")],
      W)
P("Diễn giải bằng điểm phần trăm (tính từ số chưa làm tròn): ResNet18-KD 96.02% accuracy, cao hơn ResNet18-CE "
  "0.65 điểm phần trăm (paired 3/3 seed dương); ASF-KD 95.91%, cao hơn ASF-CE 0.71 điểm phần trăm (3/3 seed "
  "dương). Sau khi sửa κ = exp(θ) ở v4, CMP-exp-CE đạt 95.49% acc / 95.48% F1 — vượt CMP-CE v1 (+0.39 điểm, "
  "3/3 seed) và vượt backbone gốc B1 ResNet18-CE (+0.11 điểm test, 2/3 seed; +0.40 điểm val, 3/3 seed) với "
  "latency đo cùng máy tương đương (3.69 ms vs 4.11 ms). CMP-exp-KD đạt 95.74% ± 0.05% (thắng CMP-exp-CE 3/3 "
  "seed, độ ổn định cao nhất nhóm KD). Tất cả mô hình đề xuất vượt B2 MobileNetV3-S; B4 ConvNeXt-T vẫn là trần.")
figure("fig_val_f1_final.png", "Hình 4. Val macro-F1 (mean ± std, 3 seeds).", 15.0)
figure("fig_test_f1_latency.png", "Hình 5. Test macro-F1 theo latency batch-1 (median).", 13.5)

# ================= 11 ABLATION =================
H1("10. Ablation và diagnostics")
P("ASF (screening 8 epoch, seed 0): fixed fusion 0.9469 / adaptive gate 0.9456 (gate sụp 1-hot về LL4 — "
  "diagnostics: max_mean = 1.0, std = 0) / energy descriptor 0.9464 (vượt control fixed cùng recipe +0.39%) / "
  "A2fix-ent 0.9472 (entropy reg giữ gate sống: max_mean 0.16). Kết luận: entropy reg là nhân tố sống còn của "
  "nhánh gate; energy descriptor có căn cứ (signed-GAP subband nhỏ 19× so LL4) nhưng không vượt GAP gate.")
P("CMP (screening 8 epoch, seed 0 + v4 sửa κ): C1 cosine K=1 0.9422 / C2 K=2 0.9441 / C3 margin 0.3 0.9404 / C2fix "
  "(k-means + diversity) 0.9398 / C3fix 0.9413 / CMP-P hard-max 0.9394 / CMP-PG pair loss 0.9416. "
  "V3b kappa check phát hiện logit_scale quên .exp() (κ bị chặn ≤ 4.605 → trần softmax 13.32% dưới KD). "
  "V4 sửa κ = exp(θ) ∈ [10, 64] (CMP-exp, 15 epoch s0): CMP-exp-CE đạt 0.9542 (+0.59% vs B1 s0) và CMP-exp-KD "
  "đạt 0.9561 (KD loss giảm từ 15.21 về 0.51 thay vì kẹt 18.46). ASF-Lite (v4 screening s0) đạt 0.9504 CE / "
  "0.9555 KD nhưng latency 4.07–4.28 ms (chậm hơn ASF v1) nên dừng ở vòng screening.")
figure("fig_gate_mean.png", "Hình 6. Phân bố gate ASF theo thành phần — 3 seeds (entropy reg).", 13.5)
P("KD dừng đúng một lần tinh chỉnh: cấu hình T=4, α=0.5 cho tín hiệu ngay từ screening; không mở grid tìm "
  "kiếm thêm theo phạm vi cho phép.")

# ================= 12 ROBUSTNESS =================
H1("11. Robustness và chi phí")
P("Robustness đo trên test với 6 corruption tổng hợp (Gaussian blur σ∈{1,2}, JPEG q∈{30,60}, brightness "
  "×{0.6,1.4}) — là nhiễu tổng hợp trong phòng, không chứng minh độ bền triển khai thực tế. ASF-KD ≥ B1-CE "
  "trên 5/6 loại corruption ở mức tuyệt đối (ví dụ blur σ2: 0.7780 vs 0.7458); EfficientNet-B0 giảm mạnh nhất "
  "ở blur σ2 (−33.9 điểm phần trăm); ConvNeXt-T bền nhất mọi loại.")
rt = pd_read = None
import pandas as pd  # noqa: E402

rt = pd.read_csv(ROOT / "outputs" / "resisc45-final-evaluation-v2-offline" / "robustness_table.csv", index_col=0)
order = ["resnet18", "ASF_A2fix_ent", "CMP_C2", "efficientnetb0", "convnextt"]
rt = rt.loc[order]
table(["Model", "clean", "blur σ1", "blur σ2", "JPEG q30", "JPEG q60", "bright ×0.6", "×1.4"],
      [[k] + [f"{rt.loc[k, c]:.4f}" for c in ("clean", "blur_1.0", "blur_2.0", "jpeg_30", "jpeg_60", "bright_0.6", "bright_1.4")]
       for k in order],
      [3.4] + [1.9] * 7)
P("Chi phí huấn luyện (mỗi run, đo trên T4, bao gồm validation mỗi epoch): nhóm ResNet18/ASF/CMP 0.20–0.21h; "
  "EfficientNet-B0 0.33h; ConvNeXt-T 0.62h. Chi phí KD: teacher inference online tăng epoch từ ~49s lên "
  "~100s (hệ số ~2×); teacher training cost lịch sử 0.62h/run ghi riêng. Chi phí tổng dự án ~16h GPU Tesla T4 "
  "(measured từ runtime kernel), dưới trần 30h/tuần Kaggle.")

# ================= 13 DEMO =================
H1("12. Demo")
P("Ứng dụng Streamlit (demo/app.py) chạy local, offline: upload ảnh, chọn 5 mô hình (B1-CE, B1-KD, ASF-CE, "
  "ASF-KD, CMP) có ghi rõ version/seed, hiển thị top-3 dự đoán với xác suất, phân biệt model-only latency và "
  "end-to-end latency, 7 tùy chọn corruption minh họa, cache model sau lần load đầu. Teacher không xuất hiện "
  "trong inference. Checkpoint demo chọn theo validation (ASF-KD seed 1, val F1 0.9587). Kiểm tra bằng code: "
  "syntax + load checkpoint + forward (đã verify trong quá trình phát triển); phần giao diện trực quan cần "
  "chạy thủ công: uv run streamlit run demo/app.py.")

# ================= 14 THAO LUAN =================
H1("13. Thảo luận: điều gì thành công, điều gì chưa")
P("Thành công: (1) protocol chống rò rỉ có kiểm chứng bằng số (manifest hash, nhóm trùng cùng split, test khóa); "
  "(2) ASF entropy regularizer giải quyết được hiện tượng sụp gate — cơ chế hoạt động đúng như thiết kế và "
  "mang lợi ích trên validation; (3) KD từ teacher cùng-domain cho cải thiện lớn, nhất quán 3/3 seed trên cả "
  "control lẫn ASF — mô hình nhỏ tốt nhất (ResNet18-KD) tiệm cận EfficientNet-B0 với chi phí thấp hơn nhiều; "
  "(4) quy trình chẩn đoán bằng số trên checkpoint phát hiện và xác nhận đúng ba hiện tượng (sụp gate, sai "
  "chiều margin, sụp prototype) thay vì đoán.")
P("Chưa thành công: (1) lợi ích ASF trên validation không khái quát sang test khi đánh giá đủ seed — tuyên bố "
  "ban đầu \"thắng B1 trên test\" đã được thu hồi và sửa trong báo cáo; (2) CMP không vượt backbone gốc ở bất "
  "kỳ cấu hình nào (7 cấu hình, 2 vòng) — phá collapse chưa đủ để tăng F1; (3) energy descriptor và entropy "
  "anneal không vượt cấu hình GAP-gate; (4) KD làm lợi ích kiến trúc ASF thu hẹp về ~0 — khi student được "
  "distill, phần đóng góp riêng của wavelet gate khó chứng minh.")

# ================= 15 HAN CHE =================
H1("14. Hạn chế")
for t in (
    "Chỉ một dataset; không kiểm chứng chéo EuroSAT do giới hạn thời gian/quota.",
    "Lợi ích ASF nằm gần biên nhiễu seed (std 0.002–0.004); 3 seed là số ít, không dùng để khẳng định phổ quát; "
    "không thực hiện kiểm định tương đương nên không kết luận \"hai mô hình như nhau\".",
    "Test đã được quan sát qua các vòng; mọi kết quả test v3 được công bố là nghiên cứu mở rộng.",
    "Robustness là corruption tổng hợp; không đại diện cho điều kiện chụp thực tế.",
    "Per-epoch history của v1 không được lưu đầy đủ vào artifact (chỉ best + thời gian) — learning curves chỉ "
    "có cho các run v2/v3.",
    "Confusion margin v1 sai chiều chỉ được phát hiện sau khi vòng v1 hoàn tất; cơ chế collapse đầy đủ chưa "
    "được tái lập trong môi trường giả lập.",
    "Không so sánh với các phương pháp SOTA chuyên biệt cho RESISC45 ngoài 5 baseline đã chọn.",
):
    doc.add_paragraph(t, style="List Bullet")

# ================= 16 KET LUAN =================
H1("15. Kết luận")
P("Trong phạm vi thí nghiệm trên NWPU-RESISC45: (1) ASF với entropy regularizer cải thiện validation nhất quán "
  "nhưng lợi ích chưa khái quát sang test; (2) chẩn đoán định lượng đã tìm ra và sửa lỗi tham số hóa κ = exp(θ) "
  "của CMP (CMP-exp-CE đạt 95.48% F1 test 3-seed, vượt cả CMP v1 lẫn backbone gốc B1 95.37%; CMP-exp-KD đạt "
  "95.74% ± 0.05%); (3) knowledge distillation từ ConvNeXt-T mang cải thiện tổng thể cao nhất — ResNet18-KD "
  "đạt 96.02% accuracy test (+0.65 điểm phần trăm, 3/3 seed) và ASF-KD đạt 95.91% (+0.70 điểm phần trăm, 3/3 seed). "
  "Toàn bộ kết quả âm tính và quá trình sửa lỗi được lưu vết đầy đủ trong lịch sử thực nghiệm.")

# ================= 17 PHAN CONG =================
H1("16. Phân công thực tế")
P("Thành viên 1: toàn bộ code (src/models.py, scripts sinh notebook), pipeline Kaggle (20 kernel qua 3 vòng), "
  "chẩn đoán và thực nghiệm, thu thập và xác minh artifacts, sơ đồ và bảng số liệu tự động.")
P("Thành viên 2: review tài liệu, báo cáo và slides (danh sách kiểm tra đi kèm), hoàn thiện phần trình bày.")

# ================= 18 REFERENCES =================
H1("17. Tài liệu tham khảo")
refs = [
    "G. Cheng et al. Remote Sensing Image Scene Classification: Benchmark and State of the Art. Proceedings of the IEEE, 2017. arXiv:1703.00121.",
    "S. E. Finder et al. Wavelet Convolutions for Large Receptive Fields. ECCV 2024. arXiv:2407.05848. Code: github.com/BGU-CS-VIL/WTConv.",
    "P. Khosla et al. Supervised Contrastive Learning. NeurIPS 2020. arXiv:2004.11362.",
    "H. Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR 2018.",
    "Y. An et al. Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces. ECCV 2020.",
    "Q. Qian et al. SoftTriple Loss: Deep Metric Learning Without Triplet Sampling. ICCV 2019.",
    "Z. Qin et al. FcaNet: Frequency Channel Attention Networks. ICCV 2021. arXiv:2012.11879.",
    "L. Beyer et al. Knowledge Distillation: A Good Teacher Is Patient and Consistent. CVPR 2022. arXiv:2106.05237.",
    "B. Zhao et al. Decoupled Knowledge Distillation. CVPR 2022. arXiv:2203.08679.",
    "H. Chen et al. Knowledge Distillation with the Reused Teacher Classifier (Knowledge Review). CVPR 2021.",
    "Y. Wan et al. FSPA-Net: Frequency–structural prior guided dual-stream dynamic fusion for remote sensing scene classification. Knowledge-Based Systems, 2026.",
    "N. Liu et al. GDPNet: Gated Dynamic Prototype Network for Few-Shot Remote Sensing Scene Classification. IEEE, 2025.",
]
for i, rf in enumerate(refs, 1):
    doc.add_paragraph(f"[{i}] {rf}", style="List Number")

# ================= 19 APPENDIX =================
H1("18. Phụ lục A — Bảng best-val seed (bản công bố ban đầu, có selection effect)")
table(["Model", "seed (best-val)", "TEST Acc", "TEST F1"],
      [[MODELS[k]["display"], str(MODELS[k]["test_bestvalseed"]["seed"]),
        f"{MODELS[k]['test_bestvalseed']['acc']:.4f}", f"{MODELS[k]['test_bestvalseed']['f1']:.4f}"]
       for k in MODELS],
      [6.0, 3.0, 3.0, 3.0])
P("Lưu ý: bảng này dùng MỘT seed tốt nhất theo validation cho mỗi mô hình; khi mở rộng đủ 3 seed (mục 9.2), "
  "con số ASF thay đổi từ 0.9550 xuống mean 0.9520 — ví dụ minh họa selection effect.")
H1("19. Phụ lục B — Tái lập")
P("Repo kaggle-research: src/models.py (toàn bộ logic), notebooks/<slug>/nb.py (nguồn notebook, build bằng "
  "scripts/nbmake.py), scripts/v2_diagnostics.py + v3_diagnostics.py (chẩn đoán), scripts/build_results_master.py "
  "(bảng số liệu), demo/app.py (Streamlit). Thứ tự kernel Kaggle và hướng dẫn đầy đủ: notes/reproduction.md. "
  "Mọi artifacts (checkpoint, report JSON, log, confusion matrix, predictions) nằm trong outputs/.")
P("Các thông tin hành chính cần bổ sung trước khi nộp: tên trường/viện, tên môn học + mã môn, học kỳ, tên và "
  "MSSV hai thành viên, ngày nộp (các trường [Điền: …] ở trang bìa).")

out_path = OUT / "Bao_cao_cuoi_mon_NWPU_RESISC45.docx"
doc.save(out_path)
print(f"saved {out_path}")
