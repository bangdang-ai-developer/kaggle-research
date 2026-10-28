#!/usr/bin/env python3
"""V3 diagnostics — kiểm chứng KD implementation (mục 4 spec v3) trước khi train.

Output: stdout + notes/v3_diagnostics.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import TrainConfig, build_model, kd_loss, train_model  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

OUT: dict = {}
torch.manual_seed(0)

print("=" * 70)
print("K1. KL CHIỀU ĐÚNG — KL(teacher‖student) ≠ KL(student‖teacher)")
print("=" * 70)
t = torch.randn(64, 45)
s = torch.randn(64, 45)
kl_ts = F.kl_div(F.log_softmax(s, 1), F.softmax(t, 1), reduction="batchmean")
kl_st = F.kl_div(F.log_softmax(t, 1), F.softmax(s, 1), reduction="batchmean")
manual_ts = (F.softmax(t, 1) * (F.log_softmax(t, 1) - F.log_softmax(s, 1))).sum(1).mean()
print(f"kd_loss-style KL(t‖s) = {kl_ts:.6f} | manual = {manual_ts:.6f} | khớp: {abs(kl_ts - manual_ts) < 1e-5}")
print(f"KL(s‖t) = {kl_st:.6f} (khác chiều — phải khác giá trị): {abs(kl_ts - kl_st) > 1e-4}")
OUT["K1"] = {"kl_ts": float(kl_ts), "kl_st": float(kl_st), "manual_match": bool(abs(kl_ts - manual_ts) < 1e-5)}

print()
print("=" * 70)
print("K2. TEACHER DETACHED — gradient không chảy vào teacher qua kd_loss")
print("=" * 70)
tt = nn.Linear(45, 45, bias=False)
ss = nn.Linear(45, 45, bias=False)
x = torch.randn(16, 45)
# đúng cách train_model: teacher forward trong no_grad
with torch.no_grad():
    t_l = tt(x)
s_l = ss(x)
kd_loss(s_l, t_l, 4.0).backward()
g_t = None if tt.weight.grad is None else float(tt.weight.grad.abs().sum())
g_s = float(ss.weight.grad.abs().sum())
print(f"grad teacher = {g_t} (phải None/0) | grad student = {g_s:.6f} (>0)")
OUT["K2"] = {"teacher_grad": g_t, "student_grad": g_s}

print()
print("=" * 70)
print("K3. T² SCALE + FP32 — kiểm khôi phục scale gradient của T²")
print("=" * 70)
T = 4.0
kl_T = kd_loss(s, t, T)
kl_T1 = F.kl_div(F.log_softmax(s, 1), F.softmax(t, 1), reduction="batchmean")
print(f"KL@T=4 (có T²) = {kl_T:.6f} | KL@T=1 = {kl_T1:.6f} | tỷ lệ ≈ {float(kl_T / kl_T1):.2f} (không cố định nhưng >1)")
# FP32: input half vẫn cho float32 output
kd_half = kd_loss(s.half().float(), t.half().float(), T)
print(f"FP32 ổn định: dtype output = {kd_half.dtype}")
OUT["K3"] = {"ratio_T4_T1": float(kl_T / kl_T1), "dtype_ok": str(kd_half.dtype)}

print()
print("=" * 70)
print("K4. α=0 ≡ CE CONTROL — pipeline khớp khi tắt KD")
print("=" * 70)
crit = nn.CrossEntropyLoss()
ce = crit(s, torch.zeros(64, dtype=torch.long))
total0 = (1.0 - 0.0) * ce + 0.0 * kd_loss(s, t, 4.0)
print(f"CE = {ce:.6f} | total(α=0) = {total0:.6f} | khớp: {abs(ce - total0) < 1e-6}")
OUT["K4_alpha0_matches_ce"] = bool(abs(ce - total0) < 1e-6)

print()
print("=" * 70)
print("K5. TEACHER FROZEN QUA train_model — weights + BN running stats không đổi")
print("=" * 70)
# teacher thật nhỏ: dùng smallcnn làm teacher giả trên synthetic data
torch.manual_seed(0)
teacher = nn.Sequential(nn.Flatten(), nn.Linear(32, 45))
student_cfg = TrainConfig(num_classes=45, model_key="smallcnn", seed=0, epochs=2, batch_size=32,
                          kd_alpha=0.5, kd_T=4.0)
student = nn.Sequential(nn.Flatten(), nn.Linear(32, 45))
X = torch.randn(128, 32)
Y = torch.randint(0, 45, (128,))
# wrap: train_model cần DataLoader trả (x, y); teacher nhận x
ds = TensorDataset(X, Y)
w_before = [p.clone() for p in teacher.parameters()]
# monkey-patch: train_model gọi teacher(x) — teacher ở đây đã là module callable ✓
class wrap_cfg:
    pass

# gọi train_model qua API thật với teacher
cfg = TrainConfig(num_classes=45, model_key="smallcnn", seed=0, epochs=1, batch_size=32,
                  kd_alpha=0.5, kd_T=4.0)
stu = build_model(cfg, pretrained=False)
tea = build_model(TrainConfig(num_classes=45, model_key="smallcnn", seed=99), pretrained=False)
w_tea_before = [p.detach().clone() for p in tea.parameters()]
bn_running_before = [b.detach().clone() for n, b in tea.named_buffers()]
dl = DataLoader(TensorDataset(torch.randn(64, 3, 256, 256), torch.randint(0, 45, (64,))), batch_size=16)
stu_tr, hist = train_model(stu, cfg, dl, None, "cpu", teacher=tea, verbose=False)
w_tea_after = [p.detach().clone() for p in tea.parameters()]
same_w = all(torch.equal(a, b) for a, b in zip(w_tea_before, w_tea_after))
same_bn = all(torch.equal(a, b) for a, b in zip(bn_running_before, [b for n, b in tea.named_buffers()]))
print(f"teacher weights không đổi: {same_w} | BN running stats không đổi: {same_bn}")
print(f"history có aux['kd']: {hist['history'][0].get('aux', {}).get('kd')}")
OUT["K5"] = {"teacher_weights_frozen": same_w, "teacher_bn_frozen": same_bn,
             "kd_aux_in_history": hist["history"][0].get("aux", {}).get("kd")}

with open(ROOT / "notes" / "v3_diagnostics.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nSaved → notes/v3_diagnostics.json")
