#!/usr/bin/env python3
"""Tạo slides PPTX cuối môn (~18 slides chính + phụ lục) từ results_master.json.

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
M = json.load(open(OUT / "results_master.json"))
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


def bullets(s, items, top=2.6, size=18, left=1.6, width=30):
    tb = s.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(15))
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
    if width: kw["width"] = Cm(width)
    if height: kw["height"] = Cm(height)
    s.shapes.add_picture(str(FIG / path), Cm(left), Cm(top), **kw)


def result_table(s, rows, headers, left=1.6, top=3.0, width=30.5, row_h=0.9, fs=13):
    n_r, n_c = len(rows) + 1, len(headers)
    shape = s.shapes.add_table(n_r, n_c, Cm(left), Cm(top), Cm(width), Cm(row_h * n_r))
    tbl = shape.table
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


F1 = lambda k: f"{MO[k]['test_3seed_f1']['mean'] * 100:.2f}"
ACC = lambda k: f"{MO[k]['test_3seed_acc']['mean'] * 100:.2f}"

# 1 Title
s = slide()
tb = s.shapes.add_textbox(Cm(2.5), Cm(6), Cm(29), Cm(7))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Phân loại cảnh ảnh viễn thám NWPU-RESISC45"
p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = DARK
p2 = tf.add_paragraph()
p2.text = "Đa tỉ lệ – dải tần thích ứng · Đa prototype · Knowledge Distillation"
p2.font.size = Pt(22); p2.font.color.rgb = ACCENT
p3 = tf.add_paragraph()
p3.text = "Nhóm 2 thành viên — [Điền tên + MSSV]\n[Điền tên môn học — mã môn — học kỳ]"
p3.font.size = Pt(16)
notes(s, "Giới thiệu nhóm và phạm vi: 3 vòng nghiên cứu trên cùng dataset, mô hình nộp chính là các bản tốt nhất theo validation.")

# 2 Bài toán + dataset
s = slide("Bài toán và dataset")
bullets(s, [
    ("NWPU-RESISC45: 31.500 ảnh viễn thám, 45 lớp × 700 ảnh, 256×256", 0, False),
    ("Nhiều cặp lớp khó chỉ khác bố cục: church↔palace, farmland, residential, wetland↔lake", 0, False),
    ("Audit đầy đủ: 0 lỗi decode, 8 nhóm trùng tuyệt đối + 16 cặp gần trùng (mutual-NN) → cùng split", 0, False),
    ("Split 75/10/15 chống leakage — manifest SHA256 khóa toàn dự án", 0, True),
    ("Hạn chế: không có metadata địa lý → không tuyên bố tổng quát hóa địa lý", 0, False),
], top=2.4, size=17)
picture(s, "fig_class_dist.png", left=1.4, top=10.6, width=22)
notes(s, "Nhấn mạnh split cố định seed 42 và việc giữ nhóm trùng cùng split để chống rò rỉ.")

# 3 Căn cứ 2 proposed
s = slide("Hai kiến trúc đề xuất — căn cứ thiết kế")
bullets(s, [
    ("ASF: đặc trưng tổng thể và chi tiết không luôn hữu ích như nhau cho từng ảnh → gate thích ứng theo ảnh trên dải tần Haar của feature map", 0, False),
    ("Giữ backbone pretrained nguyên vẹn; nhánh wavelet là phần bổ sung residual", 1, False),
    ("CMP: các lớp dễ nhầm cần ranh giới riêng → K prototype/lớp + đồ thị nhầm lẫn xây chỉ từ train", 0, False),
    ("Cùng backbone ResNet18 → so sánh công bằng với B1; teacher ConvNeXt-T cho v3", 0, False),
], top=2.8, size=19)

# 4 ASF arch
s = slide("ASF-ResNet18 — kiến trúc")
picture(s, "fig_asf_arch.png", left=2.2, top=2.5, width=29)
notes(s, "8 thành phần = (LL + 3 subband Haar) × 2 stage; gate softmax theo ảnh; entropy reg 0.05 chống sụp — bổ sung sau diagnostics.")

# 5 CMP arch
s = slide("CMP-ResNet18 — kiến trúc")
picture(s, "fig_cmp_arch.png", left=2.2, top=2.8, width=29)
notes(s, "Graph 8 cặp đúng ngữ nghĩa, chỉ từ train probe 90/10; margin v1 sai chiều đã bị diagnostics bắt.")

# 6 Protocol
s = slide("Protocol thực nghiệm")
bullets(s, [
    ("Split manifest cố định SHA256 — mọi vòng, mọi mô hình dùng chung", 0, True),
    ("15 epoch tối đa, early-stop patience 5, checkpoint chọn theo validation macro-F1", 0, False),
    ("3 seeds {0,1,2}; AdamW 1e-4/1e-3; batch 128; AMP; cùng augmentation", 0, False),
    ("Test đánh giá có kiểm soát: v1 một lần (best-val seed); v2 mở rộng đủ 3 seed; v3 công khai là nghiên cứu mở rộng", 0, False),
    ("Mỗi run lưu config trong checkpoint; thất bại hạ tầng ghi registry kèm nguyên nhân", 0, False),
], top=2.6, size=18)
notes(s, "Checkpoint rule chỉ theo validation — điểm quan trọng với giảng viên về tính hợp lệ.")

# 7 Kết quả đầy đủ (bảng chính)
s = slide("Kết quả TEST — đủ 3 seeds (mean ± std)")
rows = []
for k in ("smallcnn", "resnet18", "resnet18_kd", "mobilenetv3s", "efficientnetb0", "convnextt",
          "ASF_A2fix_ent", "asf_kd", "CMP_C2", "cmp_exp_ce", "cmp_exp_kd"):
    e = MO[k]
    rows.append([e["display"], f"{e['test_3seed_acc']['mean']*100:.2f}±{e['test_3seed_acc']['std']*100:.2f}",
                 f"{e['test_3seed_f1']['mean']*100:.2f}±{e['test_3seed_f1']['std']*100:.2f}",
                 f"{e['params_total_m']:.2f}", f"{e['latency_ms_3seed_mean']:.2f}" if "latency_ms_3seed_mean" in e else f"{e['latency_ms_bestvalseed']:.2f}"])
result_table(s, rows, ["Model", "Test Acc %", "Test F1 %", "Params M", "Lat ms"], top=2.4, row_h=1.05, fs=12)
notes(s, "Bảng này là đánh giá mở rộng công bằng đủ 3 seed. CMP-exp-CE vượt B1 (95.48 vs 95.37); ResNet18-KD cao nhất nhóm nhỏ (96.02).")

# 8 Thay đổi kết luận sau 3-seed
s = slide("Điều thay đổi khi đánh giá đủ 3 seeds")
bullets(s, [
    ("Bản công bố đầu (best-val seed): ASF 95.50% > B1 95.31% — \"thắng +0.19%\"", 0, False),
    ("Đủ 3 seed trên test: ASF 95.20±0.26 vs B1 95.37±0.19 → ASF ≈ B1 (−0.16 điểm, 1/3 seed)", 0, True),
    ("Nguyên nhân: selection effect — seed tốt theo val chưa chắc tốt trên test", 0, False),
    ("Kết luận điều chỉnh: ASF cải thiện validation nhất quán (3/3 seed, +0.39%) nhưng không khái quát sang test", 0, False),
    ("Bài học: bảng best-val-seed phải kèm cảnh báo; đánh giá đủ seed là chuẩn công bằng", 0, False),
], top=2.8, size=19)
notes(s, "Đây là điểm minh bạch quan trọng — nói thẳng về việc thu hồi một tuyên bố ban đầu.")

# 9 V2 diagnostics
s = slide("V2 — Diagnostics bằng số trên checkpoint thật")
bullets(s, [
    ("XÁC NHẬN: margin v1 sai chiều — GIẢM loss và gradient trên mẫu khó (không phải CosFace margin)", 0, True),
    ("XÁC NHẬN: prototype collapse thật — 45/45 lớp pair-cos = 1.00000000 (không phải bug đo; cross-class −0.02)", 0, True),
    ("BÁC BỎ: \"gate tắt ≡ ResNet18\" — residual/backbone đo được = 0.37", 0, True),
    ("CĂN CỨ MỚI: signed-GAP detail nhỏ 19× so LL4 (RMS chỉ 4.3×) → energy descriptor có lý", 0, False),
], top=2.8, size=19)
notes(s, "Mọi kết luận đều truy về scripts/v2_diagnostics.py + notes/v2_diagnostics.json.")

# 10 V2 screening negative
s = slide("V2 screening — sửa cơ chế xong, F1 vẫn không tăng")
rows = [["C1 cosine K=1 (v1)", "94.22"], ["C2 K=2 (v1)", "94.41"], ["C3 margin 0.3 (v1)", "94.04"],
        ["CMP-P hard-max (v2)", "93.94"], ["CMP-PG pair loss (v2)", "94.16"], ["B1 final (tham chiếu)", "95.37"]]
result_table(s, rows, ["Cấu hình CMP", "Val F1 %"], top=3.2, row_h=1.0, fs=14)
bullets(s, [
    ("Hard-max phá collapse (inter-cos 1.0 → 0.73) — chẩn đoán đúng, cơ chế hoạt động", 0, False),
    ("Nhưng F1 không tăng → vấn đề không nằm ở head prototype; CMP dừng đúng trần", 0, False),
], top=11.5, size=17)

# 11 ASF v2
s = slide("ASF v2 — energy gate và entropy anneal")
rows = [["A1fix_small (control cùng recipe)", "94.25"], ["ASF-E energy gate", "94.64"],
        ["ASF-EA energy + anneal", "94.54"], ["A2fix-ent GAP gate (v1)", "94.72"], ["B1 final (tham chiếu)", "95.37"]]
result_table(s, rows, ["Cấu hình ASF", "Val F1 %"], top=3.2, row_h=1.0, fs=14)
bullets(s, [
    ("Energy gate VƯỢT control fixed cùng recipe (+0.39%) — gate adaptive có giá trị", 0, False),
    ("Nhưng không vượt GAP-gate v1; anneal không giúp trong 8 epoch → ASF giữ v1", 0, False),
], top=11.8, size=17)

# 12 V3 KD
s = slide("V3 — Knowledge Distillation từ ConvNeXt-T")
picture(s, "fig_kd_arch.png", left=2.6, top=2.6, width=28)
notes(s, "Teacher frozen seed 2 val F1 0.9691, hash xác minh; T=4 α=0.5; same-view; chỉ student học.")

# 13 KD kết quả
s = slide("KD — cải thiện nhất quán 3/3 seeds trên TEST")
bullets(s, [
    ("ResNet18-KD 96.02% vs ResNet18-CE 95.37% → +0.65 điểm, 3/3 seed dương", 0, True),
    ("ASF-KD 95.91% vs ASF-CE 95.20% → +0.70 điểm, 3/3 seed dương", 0, True),
    ("Chi phí chỉ ở training: teacher online ~2× thời gian/epoch; inference KHÔNG cần teacher", 0, False),
    ("Latency không đổi: RN18-KD 2.53ms ≈ B1 2.52ms; ASF-KD 3.76ms", 0, False),
    ("CMP-KD negative: KD loss không hội tụ trên hard-max head → CMP không đưa KD", 0, False),
], top=2.8, size=19)
notes(s, "Đây là đóng góp chính của v3 — cải thiện lớn nhất, nhất quán, chi phí inference bằng không.")

# 13b V4 Hướng A — CMP-exp phá trần softmax
s = slide("V4 (Hướng A) — Sửa κ = exp(θ) cho CMP")
bullets(s, [
    ("Phát hiện lỗi: logit_scale khởi tạo ln(10) nhưng quên .exp() → κ ≤ 4.605, trần softmax 13.32% ở T=4", 0, True),
    ("Sửa κ = exp(θ) ∈ [10, 64] + hard-max + k-means (CMP-exp):", 0, True),
    ("CMP-exp-CE: 95.44% val (3/3 thắng B1) · 95.48% F1 test (vượt CMP v1 +0.39% 3/3 seed & B1 95.37%)", 1, False),
    ("CMP-exp-KD: KD loss 15.21 → 0.51 (hết kẹt 18.46) → 95.74% ± 0.05% test (thắng CMP-exp-CE 3/3 seed)", 1, False),
    ("Latency đo cùng máy: CMP-exp 3.69–3.71 ms ≈ ResNet18 4.11 ms (không tăng chi phí suy luận)", 0, False),
], top=2.6, size=18)
notes(s, "V4 khép kín vòng lặp chẩn đoán → sửa lỗi → kiểm chứng 3-seed. ASF-Lite dừng ở screening vì chậm hơn ASF v1.")

# 14 Bức tranh tổng
s = slide("Bức tranh tổng — kiến trúc vs distillation")
bullets(s, [
    ("Test 3-seed: ResNet18-KD 96.02 ≈ ASF-KD 95.91 — lợi ích kiến trúc ASF thu hẹp dưới KD", 0, True),
    ("Val: ASF-KD 95.76 > RN18-KD 95.52 (+0.24) — vẫn có tín hiệu kiến trúc ở val", 0, False),
    ("Cả hai vượt B1-CE 95.37 và B2 94.69; gần B3 EffB0 96.07 với chi phí thấp hơn; B4 96.78 trần", 0, False),
    ("Đóng góp chính của v3: distillation, không phải kiến trúc — trình bày đúng giới hạn", 0, True),
], top=2.8, size=19)

# 15 Robustness + chi phí
s = slide("Robustness và chi phí")
picture(s, "fig_robustness_delta.png", left=1.2, top=2.8, width=20)
bullets(s, [
    ("ASF-KD ≥ B1 5/6 corruption (blur σ2: −17.7 vs −20.7)", 0, False),
    ("EffB0 sụp blur σ2 (−34%); ConvNeXt-T bền nhất", 0, False),
    ("KD: teacher chỉ lúc train; student latency không đổi", 0, False),
], left=22.5, top=3.5, size=15, width=10.5)

# 16 Demo
s = slide("Demo (Streamlit, local)")
bullets(s, [
    ("7 model: B1-CE, B1-KD, ASF-CE, ASF-KD, CMP-v1, CMP-exp-CE, CMP-exp-KD — chọn theo validation", 0, False),
    ("Top-3 dự đoán + xác suất; model-only vs end-to-end latency phân biệt rõ", 0, False),
    ("7 corruption minh họa; teacher không xuất hiện trong inference; chạy offline", 0, False),
    ("Chạy: uv run streamlit run demo/app.py", 0, False),
], top=2.8, size=19)

# 17 Hạn chế
s = slide("Hạn chế")
bullets(s, [
    ("Một dataset (NWPU-RESISC45); chưa kiểm chứng EuroSAT", 0, False),
    ("Lợi ích ASF nằm gần biên nhiễu seed; 3 seed là mẫu nhỏ; không có kiểm định tương đương", 0, False),
    ("Test đã quan sát qua các vòng → v3 công khai là nghiên cứu mở rộng", 0, False),
    ("Robustness là corruption tổng hợp; không đại diện deployment", 0, False),
    ("Per-epoch history v1 không lưu đầy đủ (chỉ v2/v3 có)", 0, False),
], top=2.8, size=19)

# 18 Kết luận
s = slide("Kết luận")
bullets(s, [
    ("ASF + entropy reg: cơ chế gate hoạt động đúng thiết kế, cải thiện validation 3/3 seed", 0, False),
    ("CMP-exp (v4, sửa κ=exp(θ)): vượt B1 ở CE (95.48% vs 95.37% test; 95.44% vs 95.04% val) & mở khóa KD (95.74%)", 0, True),
    ("KD từ teacher cùng-domain: cải thiện lớn nhất — ResNet18-KD 96.02% test @ 2.53ms", 0, True),
    ("Toàn bộ negative results được giữ kèm phân tích cơ học; mọi số truy ngược được artifact", 0, False),
], top=2.8, size=19)

# 19 Phụ lục — pointer
s = slide("Phụ lục")
bullets(s, [
    ("Bảng đầy đủ + per-class metrics: deliverables/results_master.csv", 0, False),
    ("Reproduction: notes/reproduction.md + README trong deliverables", 0, False),
    ("Registry mọi run (kể cả fail): notes/experiment_registry.csv", 0, False),
    ("Confusion matrices ×7: outputs/resisc45-final-evaluation-v2-offline/", 0, False),
], top=2.8, size=18)

out = OUT / "Slides_cuoi_mon_NWPU_RESISC45.pptx"
prs.save(out)
print(f"saved {out} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
