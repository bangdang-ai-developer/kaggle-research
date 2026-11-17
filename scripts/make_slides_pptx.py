#!/usr/bin/env python3
"""Tạo slides PPTX cuối môn (19 slides kèm speaker notes) từ results_master.json.

Chạy: uv run python3 scripts/make_slides_pptx.py
Output: deliverables/Slides_cuoi_mon_NWPU_RESISC45.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "notes" / "figures"
OUT = ROOT / "deliverables"
with (OUT / "results_master.json").open(encoding="utf-8") as _f:
    M = json.load(_f)
MO = M["models"]

prs = Presentation()
prs.slide_width = Cm(33.87)  # 16:9
prs.slide_height = Cm(19.05)
BLANK = prs.slide_layouts[6]
DARK = RGBColor(0x1F, 0x38, 0x64)
ACCENT = RGBColor(0xC0, 0x50, 0x30)


def slide(title=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Cm(1.2), Cm(0.6), Cm(31), Cm(1.8))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(30)
        p.font.bold = True
        p.font.color.rgb = DARK
    return s


def bullets(s, items, top=2.6, size=18, left=1.6, width=30, height=14.8):
    tb = s.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (txt, lvl, bold) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "   – ") + txt
        p.level = lvl
        p.font.size = Pt(size - lvl * 2)
        p.font.bold = bold
    return tb


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def picture(s, path, left=1.6, top=2.6, width=None, height=None):
    kw = {}
    if width:
        kw["width"] = Cm(width)
    if height:
        kw["height"] = Cm(height)
    s.shapes.add_picture(str(FIG / path), Cm(left), Cm(top), **kw)


def result_table(s, rows, headers, left=1.4, top=2.6, width=31.0, row_h=0.95, fs=13, col_widths=None):
    n_r, n_c = len(rows) + 1, len(headers)
    shape = s.shapes.add_table(n_r, n_c, Cm(left), Cm(top), Cm(width), Cm(row_h * n_r))
    tbl = shape.table
    if col_widths:
        for j, cw in enumerate(col_widths):
            tbl.columns[j].width = Cm(cw)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].font.size = Pt(fs)
        c.text_frame.paragraphs[0].font.bold = True
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(v)
            c.text_frame.paragraphs[0].font.size = Pt(fs)
    return tbl


def F1(k: str) -> str:
    return f"{MO[k]['test_3seed_f1']['mean'] * 100:.2f}"


def ACC(k: str) -> str:
    return f"{MO[k]['test_3seed_acc']['mean'] * 100:.2f}"


# 1 Title
s = slide()
tb = s.shapes.add_textbox(Cm(2.2), Cm(5.5), Cm(29.5), Cm(8.5))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Phân loại cảnh ảnh viễn thám NWPU-RESISC45"
p.font.size = Pt(40)
p.font.bold = True
p.font.color.rgb = DARK
p2 = tf.add_paragraph()
p2.text = "ASF (Đa tỉ lệ – dải tần) · CMP-exp (Đa prototype) · Knowledge Distillation (Mở rộng)"
p2.font.size = Pt(21)
p2.font.color.rgb = ACCENT
p3 = tf.add_paragraph()
p3.text = ("Nhóm 2 thành viên — [Điền tên + MSSV]\n"
           "[Điền tên môn học — mã môn — học kỳ] · GVHD: [Điền: Giảng viên hướng dẫn]\n"
           "Cập nhật hoàn thiện (v1–v4): 17/11/2026")
p3.font.size = Pt(16)
notes(s, "Giới thiệu nhóm và 4 vòng nghiên cứu (v1–v4). Mô hình chính nộp theo yêu cầu môn học là ASF-CE và CMP-exp-CE (đều vượt B2 MobileNetV3-S); KD là phần mở rộng.")

# 2 Bài toán + dataset
s = slide("Bài toán và dataset")
bullets(s, [
    ("NWPU-RESISC45: 31.500 ảnh viễn thám, 45 lớp × 700 ảnh, 256×256 RGB", 0, False),
    ("Nhiều cặp lớp khó chỉ khác bố cục: church↔palace, farmland, residential, wetland↔lake", 0, False),
    ("Audit 010: 0 lỗi decode, 8 nhóm trùng tuyệt đối (16 tệp MD5), 20 cặp ứng viên gần trùng (13 cùng, 7 khác lớp) / 16 mutual-NN (12 cùng, 4 khác)", 0, False),
    ("Split 75/10/15 (23.626 / 3.149 / 4.725 ảnh) giữ nhóm trùng cùng split — manifest SHA256 khóa toàn dự án", 0, True),
    ("Hạn chế: không có metadata địa lý → không tuyên bố tổng quát hóa địa lý", 0, False),
], top=2.4, size=16)
picture(s, "fig_class_dist.png", left=1.4, top=10.8, width=22)
notes(s, "Nguồn audit 010: 20 cặp ứng viên dHash (Hamming <= 6), trong đó 16 mutual-NN. Giữ nguyên nhóm trùng cùng split, khóa SHA256 cc70020a...")

# 3 Căn cứ 2 proposed
s = slide("Hai kiến trúc đề xuất — căn cứ thiết kế")
bullets(s, [
    ("ASF-CE: đặc trưng tổng thể và chi tiết không luôn hữu ích như nhau cho từng ảnh → gate softmax thích ứng theo ảnh trên dải tần Haar", 0, False),
    ("Giữ backbone ResNet18 pretrained nguyên vẹn; nhánh wavelet cộng residual + entropy regularizer", 1, False),
    ("CMP-exp-CE (cấu hình V4): đa prototype cosine K=2 khởi tạo k-means trên train, hard-max và scale mũ κ = exp(θ) ∈ [10, 64]", 0, False),
    ("Nhánh đồ thị nhầm lẫn train-only, margin và pair loss thuộc phần ablation (không nằm trong bản CMP-exp V4 tốt nhất)", 1, False),
], top=2.8, size=18)
notes(s, "Cả hai mô hình tự chế đều gắn trên backbone ResNet18 (B1) để đối chiếu công bằng về kiến trúc và độ trễ.")

# 4 ASF arch
s = slide("ASF-ResNet18 (ASF-CE) — kiến trúc")
picture(s, "fig_asf_arch.png", left=2.2, top=2.5, width=29)
notes(s, "8 thành phần = (ORIG + 3 subband Haar LH/HL/HH) × 2 stage (layer2 32×32, layer4 8×8); gate softmax theo ảnh; entropy reg λ=0.05 chống sụp.")

# 5 CMP arch
s = slide("CMP-exp (V4) — kiến trúc đa prototype")
picture(s, "fig_cmp_arch.png", left=2.2, top=2.5, width=29)
notes(s, "Sơ đồ CMP-exp cấu hình V4: Linear 512→512 + L2 norm, K=2 prototype/lớp init bằng k-means trên train, cosine similarity, hard-max và κ = exp(θ) khởi tạo 10.0 clamp 64.0. Confusion graph/margin/pair loss nằm ở ablation.")

# 6 Protocol
s = slide("Protocol thực nghiệm")
bullets(s, [
    ("Split manifest cố định SHA256 cc70020a… — mọi vòng (v1–v4), mọi mô hình dùng chung", 0, True),
    ("Tối đa 15 epoch (cả CE và KD ở cả 3 seed đều chạy đủ 15 epoch), early-stop patience 5, chọn checkpoint theo val macro-F1", 0, False),
    ("3 seeds {0,1,2}; AdamW 1e-4/1e-3; batch 128; AMP; cùng augmentation", 0, False),
    ("Test đánh giá có kiểm soát: v1 một lần (best-val seed); v2 mở rộng đủ 3 seed; v3–v4 là nghiên cứu mở rộng", 0, False),
    ("Latency đo trên Tesla T4 (cuda), batch=1, 256×256, median (n_warmup=10, n_iter=50); tổng hợp nhiều phiên đo (071b, 112c, 121)", 0, False),
], top=2.6, size=17)
notes(s, "Xác nhận executed_epochs = 15 cho cả CE và KD ở mọi seed. Checkpoint rule chỉ theo validation macro-F1.")

# 7 Kết quả TEST chính (CE: B0–B4 + ASF-CE + CMP-exp-CE + CMP-CE v1)
s = slide("Kết quả TEST chính (CE) — đủ 3 seeds (mean ± std %)")
rows = []
for k in ("smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt",
          "ASF_A2fix_ent", "CMP_C2", "cmp_exp_ce"):
    e = MO[k]
    km = e.get("kmeans_init_h_mean", 0.0)
    tr_s = f"{e['train_time_h_mean']:.3f}" if km == 0 else f"{e['train_time_h_mean']:.3f} (+{km:.3f})"
    rows.append([
        e["display"],
        f"{e['test_3seed_acc']['mean']*100:.2f} ± {e['test_3seed_acc']['std']*100:.2f}",
        f"{e['test_3seed_f1']['mean']*100:.2f} ± {e['test_3seed_f1']['std']*100:.2f}",
        f"{e['params_total_m']:.2f}",
        f"{e['latency_ms_3seed_mean']:.2f}",
        tr_s,
    ])
result_table(
    s, rows,
    ["Model", "Test Acc (%)", "Test Macro-F1 (%)", "Params (M)", "Lat (ms)", "Train h (+km)"],
    top=2.5, row_h=0.95, fs=12, col_widths=[6.2, 5.2, 5.4, 3.6, 3.8, 6.8],
)
bullets(s, [
    ("Tách riêng bước k-means init của CMP-exp (+0.012h ≈ 44.7s/run); latency 121 cùng máy: CMP-exp-CE 3.69 ms vs ResNet18 4.11 ms", 0, False),
], top=13.5, size=13, height=3.5)
notes(s, "Bảng chính tập trung 5 baseline (B0–B4) và hai kiến trúc tự chế ở cấu hình CE. Bản mở rộng KD ở các slide sau.")

# 8 Đối chiếu yêu cầu môn học (TRƯỚC phần chẩn đoán!)
s = slide("Đối chiếu yêu cầu môn học: ASF-CE & CMP-exp-CE vs B2 & B1")
bullets(s, [
    ("Yêu cầu: ≥ 2 mô hình tự chế thắng ít nhất 1 baseline trên cùng test split (4.725 ảnh, 3 seeds, Test Macro-F1):", 0, True),
    ("ASF-CE: 95.20% test macro-F1 (acc 95.20%) — vượt B2 MobileNetV3-S (94.69%) +0.51 điểm bảng (+0.52 điểm paired, thắng 3/3 seed)", 0, True),
    ("Chưa vượt B1 ResNet18-CE (95.37%, paired −0.16 điểm, thắng 1/3 seed)", 1, False),
    ("CMP-exp-CE (V4): 95.48% test macro-F1 (acc 95.49%) — vượt B2 +0.79 điểm paired (3/3 seed) và vượt CMP v1 (95.09%) +0.39 điểm (3/3 seed)", 0, True),
    ("Hơn B1 ResNet18-CE trung bình +0.11 điểm paired, dương ở 2/3 seed (+0.42, −0.34, +0.26); trên val dương 3/3 seed (95.44% vs 95.04%)", 1, False),
    ("Hiệu quả suy luận (phiên 121 cùng máy): CMP-exp-CE 3.69 ms ≤ ResNet18 4.11 ms", 0, False),
], top=2.5, size=17)
notes(s, "Đây là slide chốt tiêu chí môn học trước khi đi vào quy trình chẩn đoán và phần mở rộng KD.")

# 9 Thay đổi kết luận sau 3-seed
s = slide("Selection effect: điều thay đổi khi đánh giá đủ 3 seeds")
bullets(s, [
    ("Bản công bố 070b (1 best-val seed): ASF-CE (s2) 95.50% test macro-F1 > B1 (s2) 95.31%", 0, False),
    ("Khi mở rộng đủ 3 seeds (071b): ASF-CE 95.20±0.26% vs B1 95.37±0.19% test macro-F1 (−0.16 điểm paired, 1/3 seed)", 0, True),
    ("Nguyên nhân: selection effect — seed tốt nhất trên validation chưa chắc tốt nhất trên test", 0, False),
    ("Điều chỉnh trung thực: ASF-CE cải thiện validation (95.43% vs 95.04%) và thắng B2 3/3 seed, nhưng chưa vượt B1 trên test", 0, False),
], top=2.8, size=18)
notes(s, "Minh bạch hóa việc rút tuyên bố ASF thắng B1 từ best-val seed và chuyển sang đánh giá công bằng 3 seed.")

# 10 V2 diagnostics
s = slide("V2 — Diagnostics bằng số trên checkpoint thật")
bullets(s, [
    ("XÁC NHẬN: margin v1 sai chiều — GIẢM loss và gradient trên mẫu khó (không phải CosFace margin)", 0, True),
    ("XÁC NHẬN: prototype collapse thật — 45/45 lớp pair-cos = 1.00000000 ở v1", 0, True),
    ("BÁC BỎ: \"gate tắt ≡ ResNet18\" — nhánh residual/backbone đo được = 0.37", 0, True),
    ("CĂN CỨ MỚI: signed-GAP detail nhỏ 19× so LL4 (RMS chỉ 4.3×) → thử energy descriptor", 0, False),
], top=2.8, size=18)
notes(s, "Mọi kết luận đều truy về scripts/v2_diagnostics.py và notes/v2_diagnostics.json.")

# 11 V2 screening negative
s = slide("Ablation V2 — confusion margin, pair loss & energy gate")
rows = [
    ["C1 cosine K=1 (v1)", "94.22"], ["C2 K=2 (v1)", "94.41"], ["C3 confusion margin 0.3 (ablation)", "94.04"],
    ["CMP-P hard-max (v2)", "93.94"], ["CMP-PG pair loss (ablation)", "94.16"],
    ["ASF A1fix (fixed control)", "94.25"], ["ASF-E energy gate (v2)", "94.64"], ["ASF A2fix-ent GAP gate (v1)", "94.72"],
]
result_table(s, rows, ["Cấu hình ablation (8 epoch, seed 0)", "Val Macro-F1 (%)"], top=2.4, row_h=0.82, fs=12, col_widths=[21.0, 10.0])
bullets(s, [
    ("Hard-max phá collapse (inter-cos 1.0 → 0.73), nhưng với scale cũ F1 chưa tăng; nhánh confusion margin & pair loss không cải thiện F1", 0, False),
    ("Energy gate vượt fixed control (+0.39 điểm val) nhưng không vượt GAP-gate A2fix-ent → giữ ASF-CE A2fix-ent", 0, False),
], top=11.8, size=15, height=5.5)
notes(s, "Ghi rõ các biến thể confusion graph/margin/pair loss thuộc phần ablation; screening 8 epoch trên seed 0.")

# 12 V3 KD kiến trúc
s = slide("Phần mở rộng (V3–V4) — Knowledge Distillation từ ConvNeXt-T")
picture(s, "fig_kd_arch.png", left=2.6, top=2.6, width=28)
notes(s, "Teacher ConvNeXt-T seed 2 (val F1 0.9691, sha 9627ca43) đóng băng chỉ chạy lúc training; T=4, α=0.5, same-view.")

# 13 V4 Hướng A + Kết quả KD mở rộng
s = slide("Phần mở rộng KD & cấu hình V4 (CMP-exp)")
bullets(s, [
    ("V3b phát hiện lỗi cũ của CMP-KD: quên .exp() trên logit_scale → κ = 3.823, trần softmax 13.32% tại T=4 (kẹt KD loss 18.46)", 0, True),
    ("Cấu hình V4 (κ = exp(θ) ∈ [10, 64] + hard-max + k-means init) giải quyết cả lỗi gradient lẫn collapse:", 0, True),
    ("Validation seed 0: tăng từ 94.48% (CMP-KD cũ) lên 95.61% (CMP-exp-KD), KD loss hội tụ 15.21 → 0.51", 1, False),
    ("Test Macro-F1 3 seeds (mở rộng KD): CMP-exp-KD đạt 95.74% ± 0.05% (+0.26 điểm vs CMP-exp-CE; 3/3 seed)", 1, False),
    ("ASF-KD đạt 95.91% test macro-F1 (+0.70 điểm paired vs ASF-CE); đối chứng ResNet18-KD đạt 96.01% F1 / 96.02% acc", 0, True),
    ("Cả CE và KD chạy đủ 15 epoch/seed; thời gian ~2× (~0.38–0.40h vs ~0.20h) do forward online của teacher", 0, False),
], top=2.5, size=16)
notes(s, "Phân biệt rạch ròi 94.48% -> 95.61% là validation seed 0, còn 95.74% là test 3 seeds. Quy mức tăng cho tổ hợp cấu hình V4.")

# 14 Bức tranh tổng
s = slide("Bức tranh tổng — kiến trúc vs distillation")
bullets(s, [
    ("Trên cấu hình chính CE: CMP-exp-CE (95.48% F1) > B1 (95.37%) > ASF-CE (95.20%) > B2 (94.69%)", 0, True),
    ("Trên phần mở rộng KD: ResNet18-KD đối chứng (96.01% F1) ≈ ASF-KD (95.91%) > CMP-exp-KD (95.74%±0.05%)", 0, True),
    ("KD giúp nâng nhóm backbone ResNet18 lên tiệm cận B3 EfficientNet-B0 (96.07% F1) ở latency thấp hơn (~2.5–3.8 vs 7.40 ms)", 0, False),
    ("Khi dùng KD, phần đóng góp riêng của gate ASF thu hẹp; B4 ConvNeXt-T (96.77% F1) vẫn là trần mô hình", 0, False),
], top=2.8, size=18)
notes(s, "Trình bày cân bằng vai trò của hai mô hình tự chế ở CE và đối chứng ResNet18-KD ở phần mở rộng.")

# 15 Robustness + chi phí
s = slide("Robustness (nguồn 070b) và chi phí thực nghiệm")
picture(s, "fig_robustness_delta.png", left=1.2, top=2.7, width=19.8)
bullets(s, [
    ("Đo trên 1 checkpoint best-val / mô hình (6 corruption tổng hợp)", 0, True),
    ("ASF-CE (seed 2) có accuracy cao hơn B1 (seed 2) trên 6/6 corruption (blur σ2: 77.80% vs 74.58%)", 0, True),
    ("Mức giảm Δ vs clean nhỏ hơn ở 5/6 corruption (ngoại lệ JPEG q30: −1.90 vs −1.84 điểm)", 1, False),
    ("EffB0 giảm mạnh ở blur σ2 (−33.9 điểm); ConvNeXt-T bền nhất", 0, False),
    ("Chi phí: ~18.5h GPU T4 (v1–v4); CMP-exp tách riêng k-means (+0.012h); CE & KD đều 15 epoch", 0, False),
], left=21.6, top=2.9, size=14, width=11.4)
notes(s, "Đã sửa nhãn ASF-KD -> ASF-CE đúng nguồn 070b robustness_table.csv (ASF_A2fix_ent seed 2). Phân biệt 6/6 accuracy cao hơn và 5/6 độ giảm nhỏ hơn.")

# 16 Demo
s = slide("Demo (Streamlit, offline local)")
bullets(s, [
    ("Đủ 7 mô hình: B1-CE, B1-KD, ASF-CE, ASF-KD, CMP-CE v1, CMP-exp-CE v4, CMP-exp-KD v4 (kèm seed & sha256)", 0, False),
    ("Đo 2 đồng hồ riêng: Model-only (forward sau load + 3 warm-up) vs App e2e (preprocess→forward→postprocess, ngoài truyền file)", 0, True),
    ("Top-3 dự đoán + xác suất; 7 corruption minh họa; teacher không xuất hiện khi suy luận", 0, False),
    ("Smoke test thật 7/7 checkpoint qua scripts/demo_smoke.py PASS; chạy UI: uv run --with streamlit streamlit run demo/app.py", 0, False),
], top=2.8, size=17)
notes(s, "Đã xác minh cả 7/7 checkpoint thật trong deliverables/demo_smoke.json.")

# 17 Hạn chế
s = slide("Hạn chế")
bullets(s, [
    ("Một dataset (NWPU-RESISC45); chưa kiểm chứng chéo EuroSAT", 0, False),
    ("3 seed là cỡ mẫu nhỏ (chưa kiểm định tương đương); chênh lệch giữa kiến trúc gần biên dao động seed", 0, False),
    ("Tập test đã được quan sát qua các vòng v1–v2 → mọi kết quả v3–v4 là nghiên cứu mở rộng", 0, False),
    ("Robustness dùng corruption tổng hợp trên 1 best-val checkpoint, chưa đại diện dữ liệu thực tế", 0, False),
    ("Không so sánh các kiến trúc SOTA chuyên biệt viễn thám ngoài 5 baseline B0–B4", 0, False),
], top=2.8, size=18)

# 18 Kết luận
s = slide("Kết luận")
bullets(s, [
    ("Hai mô hình tự chế (CE) đạt yêu cầu môn học — vượt B2 MobileNetV3-S (94.69% test macro-F1):", 0, True),
    ("ASF-CE: 95.20% test macro-F1 (+0.52 điểm paired, 3/3 seed; chưa vượt B1 95.37%), bền hơn B1 trên 6/6 corruption acc", 1, False),
    ("CMP-exp-CE (V4): 95.48% test macro-F1 (+0.79 điểm vs B2; +0.11 điểm vs B1, dương 2/3 seed; 3.69 vs 4.11 ms cùng máy)", 1, False),
    ("Mở rộng KD: đối chứng ResNet18-KD đạt 96.01% test F1 / 96.02% acc; ASF-KD 95.91% F1; CMP-exp-KD 95.74% F1", 0, True),
    ("Quy trình minh bạch: bảo toàn cả kết quả âm tính, đối chiếu đủ artifact nguồn từ 010–121", 0, False),
], top=2.7, size=17)
notes(s, "Chốt kết luận đồng nhất hoàn toàn với README, báo cáo DOCX và results_master.json.")

# 19 Phụ lục — pointer
s = slide("Phụ lục & tài liệu bàn giao")
bullets(s, [
    ("Số liệu chuẩn hóa: deliverables/results_master.json & .csv (khớp 071b, 112c, 120, 121)", 0, False),
    ("Báo cáo đầy đủ: deliverables/Bao_cao_cuoi_mon_NWPU_RESISC45.docx", 0, False),
    ("Hướng dẫn cài đặt & tái lập: README.md & deliverables/README_REPRODUCE.md", 0, False),
    ("Smoke test 7 checkpoint & validator: scripts/demo_smoke.py, scripts/validate_submission.py", 0, False),
], top=2.8, size=18)

out = OUT / "Slides_cuoi_mon_NWPU_RESISC45.pptx"
prs.save(out)
print(f"saved {out} ({len(prs.slides)} slides)")
