#!/usr/bin/env python3
"""Tạo báo cáo DOCX cuối môn từ results_master.json (nguồn số duy nhất).

Chạy: uv run python3 scripts/make_report_docx.py
Output: deliverables/Bao_cao_cuoi_mon_NWPU_RESISC45.docx
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "notes" / "figures"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)
with (OUT / "results_master.json").open(encoding="utf-8") as _f:
    M = json.load(_f)
MODELS = M["models"]

doc = Document()
for sec in doc.sections:
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
# base style
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)


def H1(t: str):
    return doc.add_heading(t, level=1)


def H2(t: str):
    return doc.add_heading(t, level=2)


def H3(t: str):
    return doc.add_heading(t, level=3)


def P(t: str):
    return doc.add_paragraph(t)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_ms(d: dict) -> str:
    return f"{d['mean'] * 100:.2f}±{d['std'] * 100:.2f}"


def _set_tr_prop(row, tag: str) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement(tag)
    tr_pr.append(el)


def table(headers: list[str], rows: list[list[str]], widths=None, font_pt: float = 8.5) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _set_tr_prop(t.rows[0], "w:tblHeader")
    _set_tr_prop(t.rows[0], "w:cantSplit")
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(font_pt)
    for row in rows:
        tr = t.add_row()
        _set_tr_prop(tr, "w:cantSplit")
        cells = tr.cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_pt)
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


def mrow_main(key: str) -> list[str]:
    e = MODELS[key]
    km = e.get("kmeans_init_h_mean", 0.0)
    tr_str = f"{e['train_time_h_mean']:.3f}" if km == 0 else f"{e['train_time_h_mean']:.3f} (+{km:.3f})"
    return [
        e["display"],
        f"{e['params_total_m']:.2f}",
        fmt_ms(e["val_f1_3seed"]),
        fmt_ms(e["test_3seed_acc"]),
        fmt_ms(e["test_3seed_f1"]),
        f"{e['latency_ms_3seed_mean']:.2f}",
        tr_str,
    ]


def mrow_full(key: str) -> list[str]:
    e = MODELS[key]
    return [
        e["display"],
        f"{e['params_total_m']:.2f}",
        fmt_ms(e["val_f1_3seed"]),
        fmt_ms(e["test_3seed_acc"]),
        fmt_ms(e["test_3seed_f1"]),
        fmt_ms(e["test_3seed_precision"]),
        fmt_ms(e["test_3seed_recall"]),
        f"{e['latency_ms_3seed_mean']:.2f}",
        f"{e['train_time_h_mean']:.3f}",
        f"{e.get('kmeans_init_h_mean', 0.0):.3f}",
    ]


HDR_MAIN = ["Model", "Params (M)", "Val F1 (%)", "Test Acc (%)", "Test F1 (%)", "Lat (ms)", "Train h (+km)"]
W_MAIN = [3.6, 1.8, 2.5, 2.5, 2.5, 1.7, 2.4]
HDR_FULL = ["Model", "Params", "Val F1 (%)", "Test Acc (%)", "Test F1 (%)", "P macro (%)", "R macro (%)", "Lat", "Train h", "km h"]
W_FULL = [2.9, 1.1, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.8]

# ================= TRANG BÌA =================
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("[Điền: Tên trường / viện]")
r.font.size = Pt(13)
t2 = doc.add_paragraph()
t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t2.add_run("[Điền: Tên môn học — mã môn] · Học kỳ [Điền]\nGiảng viên hướng dẫn: [Điền: Giảng viên hướng dẫn]")
r.font.size = Pt(12)

tt = doc.add_paragraph()
tt.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tt.add_run("\nPhân loại cảnh ảnh viễn thám NWPU-RESISC45 bằng lựa chọn đặc trưng đa tỉ lệ, "
               "bộ phân loại đa prototype theo đồ thị nhầm lẫn và knowledge distillation\n")
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
r = dt.add_run("Cập nhật hoàn thiện (v1–v4): 17/11/2026 · Repo: kaggle-research (commit [Điền final commit])")
r.font.size = Pt(10)

# ================= TÓM TẮT =================
H1("Tóm tắt")
P("Báo cáo trình bày bốn vòng nghiên cứu (v1–v4) trên NWPU-RESISC45 (31.500 ảnh, 45 lớp, 256×256) với 5 baseline "
  "(B0 SmallCNN train từ đầu; B1 ResNet18, B2 MobileNetV3-S, B3 EfficientNet-B0, B4 ConvNeXt-T ImageNet-pretrained) "
  "và hai kiến trúc tự chế gắn trên ResNet18: ASF-CE (Haar wavelet + gate softmax kèm entropy regularizer) và "
  "CMP-exp-CE (đa prototype cosine với khởi tạo k-means, hard-max và κ = exp(θ) ∈ [10, 64] theo cấu hình V4).")
P("Vòng v1–v2 thiết lập split chống rò rỉ 75/10/15 khóa bằng SHA256, đánh giá đủ 3 seed trên tập test (4.725 ảnh) "
  "và chẩn đoán định lượng trên checkpoint thật. Theo yêu cầu môn học, hai phiên bản chính huấn luyện bằng CE đạt "
  "kết quả test macro-F1 trung bình 3 seed như sau: ASF-CE đạt 95.20% (vượt B2 MobileNetV3-S 94.69% khoảng +0.51 điểm "
  "phần trăm, chưa vượt B1 ResNet18-CE 95.37%); CMP-exp-CE (cấu hình V4) đạt 95.48% (vượt B2 khoảng +0.79 điểm và "
  "hơn B1 trung bình +0.11 điểm, dương ở 2/3 seed) với latency cùng máy 121 là 3.69 ms vs 4.11 ms ResNet18.")
P("Trong phần mở rộng Knowledge Distillation (ConvNeXt-T s2 teacher frozen, T=4, α=0.5, cả CE và KD đều chạy đủ "
  "15 epoch/seed): đối chứng ResNet18-KD đạt 96.01% macro-F1 (96.02% accuracy), ASF-KD đạt 95.91% macro-F1 và "
  "CMP-exp-KD đạt 95.74% ± 0.05% macro-F1 (+0.26 điểm so với CMP-exp-CE; trên validation seed 0, cấu hình V4 khắc phục "
  "trần softmax 13.32% của bản CMP-KD cũ và nâng val F1 seed 0 từ 94.48% lên 95.61%). Tổng chi phí toàn dự án qua "
  "v1–v4 là ~18.5h GPU Tesla T4, mọi số liệu đều truy ngược về artifact nguồn.")

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
P("Nguồn: Kaggle dataset punfake/resisc45 (bản sao NWPU-RESISC45 gốc, license apache-2.0). Audit 010 trên toàn bộ "
  "31.500 ảnh: 0 lỗi decode, 100% kích thước 256×256 RGB, đủ 45 lớp × 700 ảnh. Phát hiện trùng lặp: 8 nhóm "
  "trùng tuyệt đối (16 tệp MD5 giống hệt nhau) và 20 cặp ứng viên gần trùng theo dHash (Hamming ≤ 6: 13 cùng "
  "lớp, 7 khác lớp), trong đó 16 cặp là mutual nearest-neighbor (12 cùng lớp, 4 khác lớp); các cặp khác lớp "
  "được giữ cùng split để loại trừ rò rỉ dù kiểm tra pixel cho thấy chỉ tương tự cấu trúc.")
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
P("Căn cứ thiết kế: các lớp dễ nhầm có thể có nhiều mode nội tại; nhiều prototype mỗi lớp kèm chuẩn hóa cosine "
  "và tham số scale đủ lớn có thể tạo ranh giới phân tách sắc hơn đầu linear/cosine đơn tâm.")
figure("fig_cmp_arch.png", "Hình 2. Sơ đồ CMP-exp (cấu hình V4 dùng trong bảng kết quả): k-means init, cosine, hard-max và κ = exp(θ) ∈ [10, 64].")
P("Cấu trúc CMP-exp (V4): embedding Linear(512→512) + L2 normalize; K=2 prototype mỗi lớp khởi tạo bằng k-means "
  "trên vector đặc trưng tập train trước vòng lặp train (thời gian k-means đo riêng ~42–45s/run); phân lớp dùng "
  "cosine similarity, hard-max max_k cos(e, p_{c,k}) và scale mũ κ = exp(θ) khởi tạo tại 10.0 và chặn trên tại "
  "64.0 (clamp trong forward). Nhánh đồ thị nhầm lẫn train-only (8 cặp đối xứng như church↔palace, dense↔medium "
  "residential), margin cosine và pair loss thuộc nhóm ablation (C3, C3fix, CMP-PG).")
P("Tiến trình qua các vòng: v1 (CMP_C2 với scale linear và LSE-logits) bị sụp prototype ở 45/45 lớp; v2 (hard-max, "
  "k-means) phá collapse nhưng K=2 ở scale cũ không vượt B1; v3 phát hiện quên .exp() trên logit_scale ở bản "
  "CMP-KD cũ (κ = 3.823, trần softmax 13.32% tại T=4). Tổ hợp cấu hình V4 (κ = exp(θ) ∈ [10, 64] + hard-max + "
  "k-means init) giải quyết đồng thời cả hai vấn đề và nâng CMP-exp-CE lên 95.48% F1 test.")

# ================= 8 KD =================
H1("7. Distillation và teacher (phần mở rộng v3–v4)")
figure("fig_kd_arch.png", "Hình 3. Sơ đồ KD: teacher frozen chỉ dùng lúc training; student inference độc lập.")
P("Teacher: ConvNeXt-T seed 2 — chọn theo validation macro-F1 (0.9691) trong 3 seed của 060a, không theo test; "
  "checkpoint hash SHA256 9627ca43… được xác minh khi load; frozen eval mode, không nhận gradient, running "
  "statistics không đổi (kiểm chứng K5). Teacher và student nhận cùng tensor x sau augmentation (same-view); "
  "không cache logits.")
P("Loss: L = (1 − α)·CE(student, y) + α·T²·KL(softmax(z_t/T) ‖ log_softmax(z_s/T)) với α = 0.5, T = 4. Năm kiểm "
  "chứng bắt buộc đều đạt: chiều KL đúng (khớp tính tay), teacher không nhận gradient, hệ số T² có mặt, α=0 "
  "trả về đúng CE, teacher weights + BN running stats bất biến qua vòng train thật.")
P("Chi phí và kiểm chứng epoch: đối chiếu executed_epochs xác nhận cả CE lẫn KD ở mọi seed (0, 1, 2) đều chạy "
  "đủ 15 epoch. Thời gian mỗi epoch tăng từ ~47–50s (CE) lên ~100s (KD, hệ số ~2×) hoàn toàn do forward pass "
  "online của teacher ConvNeXt-T, không phải do CE dừng sớm. Chi phí train teacher ở v1 (0.616h/run) được ghi "
  "riêng; student inference độc lập không cần teacher.")

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
H1("9. Kết quả chính (3 seeds, mean ± std %)")
H2("9.1. Validation (3 seeds)")
table(["Model", "Val F1 (%)", "Val Acc (%)"],
      [[MODELS[k]["display"], fmt_ms(MODELS[k]["val_f1_3seed"]), fmt_ms(MODELS[k]["val_acc_3seed"])]
       for k in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt",
                 "ASF_A2fix_ent", "CMP_C2", "cmp_exp_ce", "resnet18_kd", "asf_kd", "cmp_exp_kd")],
      [6.0, 4.5, 4.5])
H2("9.2. Test chính — Baselines B0–B4 & hai kiến trúc tự chế (CE)")
P("Ghi chú: bảng chính trình bày B0–B4 và hai mô hình đề xuất ở cấu hình CE (ASF-CE, CMP-exp-CE v4, kèm CMP-CE v1 "
  "để đối chiếu). Std là độ lệch giữa 3 training seed; thời gian k-means của CMP-exp (+0.012h ≈ 44.7s CE / 42.0s KD) "
  "nằm ngoài train_model và ghi riêng trong ngoặc. Latency là trung bình 3 seed của các median batch-1 T4 (ms); "
  "bảng tổng hợp gộp nhiều lượt đo (071b, 112c, 121). Bảng đầy đủ cả Precision/Recall và best-val-seed ở Phụ lục A.")
table(HDR_MAIN,
      [mrow_main(k) for k in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt",
                              "ASF_A2fix_ent", "CMP_C2", "cmp_exp_ce")],
      W_MAIN)
H2("9.3. Phần mở rộng — Knowledge Distillation (teacher ConvNeXt-T s2)")
P("ResNet18-KD đóng vai trò đối chứng KD trên backbone chuẩn; ASF-KD và CMP-exp-KD là phiên bản huấn luyện KD "
  "của hai kiến trúc tự chế:")
table(HDR_MAIN,
      [mrow_main(k) for k in ("resnet18_kd", "asf_kd", "cmp_exp_kd")],
      W_MAIN)
P("Đối chiếu yêu cầu môn học trên test macro-F1 (paired differences tính theo từng seed rồi lấy trung bình): "
  "(1) ASF-CE đạt 95.20% (accuracy 95.20%), vượt B2 MobileNetV3-S (94.69%) trung bình +0.52 điểm paired (+0.51 điểm "
  "theo hiệu số trung bình, thắng 3/3 seed), chưa vượt B1 ResNet18-CE (95.37%, paired −0.16 điểm, 1/3 seed); "
  "(2) CMP-exp-CE (V4) đạt 95.48% (accuracy 95.49%), vượt B2 trung bình +0.79 điểm paired (3/3 seed), vượt CMP v1 "
  "(95.09%) +0.39 điểm (3/3 seed), và vượt B1 ResNet18-CE trung bình +0.11 điểm paired (dương ở 2/3 seed: +0.42, "
  "−0.34, +0.26; trên validation dương 3/3 seed 95.44% vs 95.04%) với latency cùng lượt đo 121 là 3.69 ms vs "
  "4.11 ms ResNet18.")
P("Kết quả phần mở rộng KD: ResNet18-KD (đối chứng) đạt 96.01% macro-F1 / 96.02% accuracy (+0.65 điểm paired vs "
  "ResNet18-CE, 3/3 seed); ASF-KD đạt 95.91% macro-F1 (+0.70 điểm paired vs ASF-CE, 3/3 seed); CMP-exp-KD đạt "
  "95.74% ± 0.05% macro-F1 (accuracy 95.73%, +0.26 điểm paired vs CMP-exp-CE, 3/3 seed).")
figure("fig_val_f1_final.png", "Hình 4. Val macro-F1 (mean ± std, 3 seeds).", 15.0)
figure("fig_test_f1_latency.png", "Hình 5. Test macro-F1 theo latency batch-1 (toàn cảnh và zoom nhóm chính).", 14.5)

# ================= 11 ABLATION =================
H1("10. Ablation và diagnostics")
P("ASF (screening 8 epoch, seed 0): fixed fusion 0.9469 / adaptive gate 0.9456 (gate sụp 1-hot về LL4 — "
  "diagnostics: max_mean = 1.0, std = 0) / energy descriptor 0.9464 (vượt control fixed cùng recipe +0.39%) / "
  "A2fix-ent 0.9472 (entropy reg giữ gate sống: max_mean 0.16). Kết luận: entropy reg là nhân tố sống còn của "
  "nhánh gate; energy descriptor có căn cứ (signed-GAP subband nhỏ 19× so LL4) nhưng không vượt GAP gate.")
P("CMP (screening 8 epoch, seed 0 + cấu hình V4): C1 cosine K=1 0.9422 / C2 K=2 0.9441 / C3 margin 0.3 0.9404 / "
  "C2fix (k-means + diversity) 0.9398 / C3fix 0.9413 / CMP-P hard-max 0.9394 / CMP-PG pair loss 0.9416 (nhánh "
  "confusion graph, margin và pair loss thuộc phần ablation này). V3b kappa check cho thấy bản CMP-KD cũ bị kẹt ở "
  "κ = 3.823 tạo cận xác suất mềm 13.32% tại T = 4. Chuyển sang cấu hình V4 (κ = exp(θ) khởi tạo 10.0, clamp 64.0 "
  "+ hard-max + k-means init): ở seed 0 (15 epoch), val macro-F1 tăng từ 94.48% (CMP-KD cũ) lên 95.61% (CMP-exp-KD) "
  "và KD loss giảm sạch 15.21 → 0.51 thay vì kẹt 18.46; trên test 3-seed đạt 95.74% (KD) và 95.48% (CE). Mức tăng "
  "so với v1 được quy cho tổ hợp cấu hình V4 do đồng thời đổi scale, hard-max và khởi tạo k-means.")
P("ASF-Lite (V4 screening seed 0) đạt val F1 0.9504 (CE) và 0.9555 (KD) nhưng latency 4.07–4.28 ms chậm hơn "
  "ASF v1 (3.82 ms) nên dừng ở vòng screening.")
figure("fig_gate_mean.png", "Hình 6. Phân bố gate ASF theo thành phần — 3 seeds (entropy reg).", 13.5)
P("KD dừng đúng một lần tinh chỉnh: cấu hình T=4, α=0.5 cho tín hiệu ngay từ screening; không mở grid tìm "
  "kiếm thêm theo phạm vi cho phép.")

# ================= 12 ROBUSTNESS =================
H1("11. Robustness và chi phí")
P("Robustness (nguồn 070b) đo bằng accuracy trên một checkpoint được chọn theo validation cho mỗi mô hình (ResNet18 s2, "
  "ASF-CE s2, CMP_C2 s1, EfficientNet-B0 s0, ConvNeXt-T s2) qua 6 corruption tổng hợp — là nhiễu mô phỏng, không "
  "đại diện cho mọi điều kiện chụp thực tế. ASF-CE đạt accuracy tuyệt đối cao hơn B1 ResNet18-CE trên 6/6 loại "
  "corruption (ví dụ blur σ2: 0.7780 vs 0.7458) và có mức suy giảm (Δ so với clean) nhỏ hơn ở 5/6 loại (ngoại lệ "
  "JPEG q30 giảm −1.90 điểm vs −1.84 điểm ở B1). EfficientNet-B0 giảm mạnh nhất ở blur σ2 (−33.9 điểm); ConvNeXt-T bền nhất.")

rt = pd.read_csv(ROOT / "outputs" / "resisc45-final-evaluation-v2-offline" / "robustness_table.csv", index_col=0)
order = ["resnet18", "ASF_A2fix_ent", "CMP_C2", "efficientnetb0", "convnextt"]
rt_names = {"resnet18": "B1 ResNet18-CE (s2)", "ASF_A2fix_ent": "ASF-CE (s2)", "CMP_C2": "CMP-CE v1 (s1)",
            "efficientnetb0": "B3 EfficientNet-B0 (s0)", "convnextt": "B4 ConvNeXt-T (s2)"}
rt = rt.loc[order]
table(["Model (1 best-val ckpt)", "clean", "blur σ1", "blur σ2", "JPEG q30", "JPEG q60", "bright ×0.6", "×1.4"],
      [[rt_names[k]] + [f"{rt.loc[k, c]:.4f}" for c in ("clean", "blur_1.0", "blur_2.0", "jpeg_30", "jpeg_60", "bright_0.6", "bright_1.4")]
       for k in order],
      [3.8] + [1.85] * 7)
P("Chi phí huấn luyện (mỗi run trên T4, gồm validation mỗi epoch, tất cả 15 epoch): nhóm CE ResNet18/ASF/CMP ~0.20–0.21h; "
  "riêng CMP-exp có bước k-means init đo tách biệt (+0.012h/run: ~44.7s CE / ~42.0s KD); EfficientNet-B0 0.331h; "
  "ConvNeXt-T 0.616h. KD chạy đủ 15 epoch mất ~0.38–0.40h/run (~2× do teacher forward online). Tổng chi phí toàn "
  "dự án (v1–v4) ~18.5h GPU Tesla T4 (measured từ kernel runtime), dưới trần 30h/tuần Kaggle.")

# ================= 13 DEMO =================
H1("12. Demo")
P("Ứng dụng Streamlit (demo/app.py) chạy local, offline: upload ảnh, chọn đủ 7 mô hình (B1 ResNet18-CE, B1 ResNet18-KD, "
  "ASF-CE, ASF-KD, CMP-CE v1, CMP-exp-CE v4, CMP-exp-KD v4) kèm seed/sha256, top-3 dự đoán và 7 tùy chọn corruption. "
  "Hai đồng hồ đo riêng phạm vi: Model-only đo forward pass trên tensor (1,3,256,256) sau khi load + 3 warm-up; "
  "App-side end-to-end đo từ lúc ứng dụng nhận đối tượng PIL đến preprocess → forward → postprocess (không gồm "
  "thời gian truyền file qua trình duyệt). Teacher không xuất hiện khi inference. scripts/demo_smoke.py đã kiểm tra "
  "thành công cả 7/7 checkpoint thật (deliverables/demo_smoke.json); chạy UI: uv run --with streamlit streamlit run demo/app.py.")

# ================= 14 THAO LUAN =================
H1("13. Thảo luận: điều gì thành công, điều gì chưa")
P("Thành công: (1) protocol chống rò rỉ có kiểm chứng bằng số (manifest hash, 20 cặp gần trùng / 16 mutual-NN "
  "giữ cùng split, test khóa); (2) ASF entropy regularizer giữ gate phân bố mềm và cải thiện +0.52 điểm vs B2 "
  "(3/3 seed) đồng thời tăng độ bền corruption (6/6 acc, 5/6 Δ so với B1); (3) chẩn đoán định lượng tìm ra "
  "nguyên nhân thất bại của CMP v1–v3 và tổ hợp cấu hình V4 giúp CMP-exp-CE vượt B2 (+0.79 điểm) lẫn B1 (+0.11 điểm "
  "trung bình, 2/3 seed) ở latency tương đương; (4) phần mở rộng KD cho mức tăng cao nhất (ResNet18-KD 96.01% F1).")
P("Chưa thành công: (1) ASF-CE chưa vượt được B1 ResNet18-CE trên test (95.20% vs 95.37%); (2) CMP v1–v2 và nhánh "
  "confusion margin/pair loss không vượt backbone gốc; (3) energy descriptor và entropy anneal không vượt GAP-gate; "
  "(4) trong phần mở rộng KD, lợi ích riêng của nhánh gate ở ASF-KD thu hẹp so với đối chứng ResNet18-KD.")

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
P("Trong phạm vi thí nghiệm trên NWPU-RESISC45 (test macro-F1 trung bình 3 seed): hai mô hình tự chế ở cấu hình "
  "chính đều vượt baseline B2 MobileNetV3-S (94.69%): ASF-CE đạt 95.20% (+0.52 điểm paired, 3/3 seed; chưa vượt B1 "
  "ResNet18-CE 95.37%), và CMP-exp-CE (cấu hình V4) đạt 95.48% (+0.79 điểm paired vs B2; +0.11 điểm trung bình "
  "vs B1, dương 2/3 seed; latency cùng lượt 121 là 3.69 ms vs 4.11 ms ResNet18). Trong phần mở rộng KD (v3–v4): "
  "đối chứng ResNet18-KD đạt 96.01% macro-F1 / 96.02% accuracy, ASF-KD đạt 95.91% và CMP-exp-KD đạt 95.74% ± 0.05%. "
  "Mọi số liệu và cả các nhánh ablation âm tính được lưu vết đầy đủ từ artifact nguồn.")

# ================= 17 PHAN CONG =================
H1("16. Phân công thực tế")
P("Thành viên 1: toàn bộ code (src/models.py, scripts sinh notebook), pipeline Kaggle (v1–v4), "
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
H1("18. Phụ lục A — Bảng đầy đủ 11 mô hình & best-val seed")
H2("A.1. Toàn bộ 11 mô hình với Precision/Recall, Latency và k-means init")
table(HDR_FULL,
      [mrow_full(k) for k in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt",
                              "ASF_A2fix_ent", "CMP_C2", "cmp_exp_ce", "resnet18_kd", "asf_kd", "cmp_exp_kd")],
      W_FULL,
      font_pt=8.0)
H2("A.2. Bảng best-val seed (bản đánh giá 1 seed 070b, có selection effect)")
table(["Model", "seed (best-val)", "TEST Acc (%)", "TEST F1 (%)"],
      [[MODELS[k]["display"], str(MODELS[k]["test_bestvalseed"]["seed"]),
        pct(MODELS[k]['test_bestvalseed']['acc']), pct(MODELS[k]['test_bestvalseed']['f1'])]
       for k in MODELS],
      [6.0, 3.0, 3.0, 3.0])
P("Lưu ý: bảng A.2 dùng MỘT seed tốt nhất theo validation cho mỗi mô hình; khi mở rộng đủ 3 seed (mục 9.2), "
  "ASF-CE thay đổi từ 95.50% xuống mean 95.20% — ví dụ minh họa selection effect.")
H1("19. Phụ lục B — Tái lập & thông tin cần điền")
P("Repo kaggle-research: src/models.py (toàn bộ logic), notebooks/<slug>/nb.py (nguồn notebook, build bằng "
  "scripts/nbmake.py), scripts/v2_diagnostics.py + v3_diagnostics.py + v4_audit.py (chẩn đoán), "
  "scripts/build_results_master.py (bảng số liệu), scripts/validate_submission.py, demo/app.py (Streamlit). "
  "Hướng dẫn đầy đủ: deliverables/README_REPRODUCE.md và README.md.")
P("Các thông tin hành chính cần bổ sung trước khi nộp: [Điền: Tên trường / viện], [Điền: Tên môn học — mã môn], "
  "[Điền: Học kỳ], [Điền: Giảng viên hướng dẫn], [Điền tên — MSSV] cho Thành viên 1 và Thành viên 2, "
  "và [Điền final commit] ở trang bìa.")

out_path = OUT / "Bao_cao_cuoi_mon_NWPU_RESISC45.docx"
doc.save(out_path)
print(f"saved {out_path}")
