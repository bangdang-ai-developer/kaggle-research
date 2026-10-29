#!/usr/bin/env python3
"""V4 audit — 2 kiểm tra bắt buộc trước khi đóng bảng số liệu:
A. Fairness CE vs KD: so configs thật từ checkpoint files.
B. Seed-0 merge: config của ckpt seed 0 (110b) vs seed 1/2 (111) — tương đương để gộp?
C. Git archaeology: refactor _gate_input ảnh hưởng run ASF-KD seed 0 không.

Output: notes/v4_audit.json + stdout.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT_B = ROOT / "outputs" / "resisc45-final-baselines-b0-b4-3-seeds"
OUT_P = ROOT / "outputs" / "resisc45-final-proposed-asf-cmp-3-seeds"
OUT_KD0 = ROOT / "outputs" / "resisc45-screen-kd-resnet18-asf-cmp-v2"
OUT_KD12 = ROOT / "outputs" / "resisc45-kd-final-seeds-1-2"

FIELDS_DROP = {"extra"}  # trường tự do


def load_cfg(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg["epoch_saved"] = ck.get("epoch")
    return cfg


def diff(a: dict, b: dict, ignore: set) -> list[str]:
    keys = (set(a) | set(b)) - ignore
    return [k for k in sorted(keys) if a.get(k) != b.get(k)]


OUT: dict = {}

# ---------------- A. Fairness CE vs KD (ResNet18 + ASF, seed 0/1/2) ----------------
print("=" * 70)
print("A. FAIRNESS CE vs KD — so config từng cặp cùng model/seed")
print("=" * 70)
pairs = []
for model, ce_dir, kd_dir, ce_name, kd_name in (
    ("resnet18", OUT_B, OUT_KD0 if False else None, "ckpt_resnet18_s0.pt", "ckpt_resnet18_kd.pt"),
):
    pass  # placeholder — xử lý bên dưới theo seed

diffs_summary = {}
for seed in (0, 1, 2):
    ce_path = OUT_B / f"ckpt_resnet18_s{seed}.pt"
    kd_path = (OUT_KD0 / "ckpt_resnet18_kd.pt") if seed == 0 else (OUT_KD12 / f"ckpt_resnet18_kd_s{seed}.pt")
    ca, ck_ = load_cfg(ce_path), load_cfg(kd_path)
    d = diff(ca, ck_, FIELDS_DROP)
    diffs_summary[f"resnet18_s{seed}"] = d
    print(f"resnet18 seed {seed}: khác biệt config: {d if d else 'KHÔNG (trừ kd_alpha/kd_T)'}")
for seed in (0, 1, 2):
    ce_path = OUT_P / f"ckpt_ASF_A2fix_ent_s{seed}.pt"
    kd_path = (OUT_KD0 / "ckpt_asf_kd.pt") if seed == 0 else (OUT_KD12 / f"ckpt_asf_kd_s{seed}.pt")
    ca, ck_ = load_cfg(ce_path), load_cfg(kd_path)
    d = diff(ca, ck_, FIELDS_DROP)
    diffs_summary[f"asf_s{seed}"] = d
    print(f"asf seed {seed}: khác biệt config: {d if d else 'KHÔNG (trừ kd_alpha/kd_T)'}")

# các trường chắc chắn khác (kd_alpha/kd_T) + những gì GIỐNG nhau là cơ sở fairness
same_key_check = {}
ce0 = load_cfg(OUT_B / "ckpt_resnet18_s0.pt")
kd0 = load_cfg(OUT_KD0 / "ckpt_resnet18_kd.pt")
for k in ("epochs", "batch_size", "lr_backbone", "lr_head", "weight_decay", "amp",
          "warmup_frac", "early_stop_patience", "input_size", "label_smoothing",
          "num_workers", "seed"):
    same_key_check[k] = (ce0.get(k), kd0.get(k), ce0.get(k) == kd0.get(k))
print("\nTrường protocol then chốt (CE, KD, bằng nhau?):")
for k, (a, b, eq) in same_key_check.items():
    print(f"  {k}: {a} vs {b} → {'GIỐNG' if eq else 'KHÁC'}")
OUT["A_fairness"] = {"config_diffs": diffs_summary, "key_fields": {k: {"ce": v[0], "kd": v[1], "equal": v[2]} for k, v in same_key_check.items()}}

# ---------------- B. Seed-0 KD config vs seeds 1/2 KD ----------------
print()
print("=" * 70)
print("B. SEED-0 KD MERGE — config s0 (110b) vs s1/s2 (111)")
print("=" * 70)
merge_ok = {}
for kd_name_s0, model in (("ckpt_resnet18_kd.pt", "resnet18_kd"), ("ckpt_asf_kd.pt", "asf_kd")):
    c0 = load_cfg(OUT_KD0 / kd_name_s0)
    for seed in (1, 2):
        cn = load_cfg(OUT_KD12 / f"ckpt_{model}_s{seed}.pt")
        d = diff(c0, cn, FIELDS_DROP | {"seed"})
        merge_ok[f"{model}_s0_vs_s{seed}"] = d
        print(f"{model}: s0 vs s{seed} → khác: {d if d else 'KHÔNG (trừ seed)'}")
OUT["B_merge"] = merge_ok

# teacher hash: từ logs kernel (đã in khi chạy) — xác minh lại bằng records đã pull
log_110b = (ROOT / "outputs" / "resisc45-screen-kd-resnet18-asf-cmp-v2" / "resisc45-screen-kd-resnet18-asf-cmp-v2.log").read_text()
log_111 = (ROOT / "outputs" / "resisc45-kd-final-seeds-1-2" / "resisc45-kd-final-seeds-1-2.log").read_text()
t110 = "9627ca43dc5a39ca" in log_110b
t111 = "9627ca43dc5a39ca" in log_111
print(f"\nteacher sha prefix trong log 110b: {t110} | trong log 111: {t111}")
OUT["B_teacher_hash_logs"] = {"110b": t110, "111": t111}

# ---------------- C. Git archaeology: refactor _gate_input vs ASF-KD s0 runtime ----------------
print()
print("=" * 70)
print("C. GIT ARCHAEOLOGY — _gate_input nhánh gap lúc ASF-KD s0 chạy (110b)")
print("=" * 70)
# commits chạm src/models.py với dòng liên quan
log_cmd = subprocess.run(
    ["git", "-C", str(ROOT), "log", "--format=%H %cI %s", "-S",
     "return self.gate(stack.flatten(1))", "--", "src/models.py"],
    capture_output=True, text=True)
print("commits giới thiệu/xóa dòng 'return self.gate(stack.flatten(1))':")
print(log_cmd.stdout or "(không có)")
OUT["C_git_log_gate_fix"] = log_cmd.stdout.strip().splitlines()

# commit tại thời điểm 110b push (28/10 ~09:20 local): commit cuối TRƯỚC mốc
log_all = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H %cI %s"],
                         capture_output=True, text=True).stdout.splitlines()
cutoff = "2026-10-28T09:20"
at_push = [l for l in log_all if l.split(" ", 1)[1][:16] <= cutoff]
print(f"\ncommit cuối trước 110b push ({cutoff}): {at_push[0] if at_push else 'n/a'}")
OUT["C_last_commit_before_110b"] = at_push[0] if at_push else None

with open(ROOT / "notes" / "v4_audit.json", "w") as f:
    json.dump(OUT, f, indent=1, ensure_ascii=False)
print("\nSaved → notes/v4_audit.json")
