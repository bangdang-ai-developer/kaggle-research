"""Unit & regression tests cho design patterns và tương thích ngược checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import mean_std, mean_std_dict

from models import (
    MODEL_REGISTRY,
    ASFLiteResNet18,
    ASFResNet18,
    CMPResNet18,
    EarlyStopper,
    KDDistiller,
    TrainConfig,
    build_model,
    count_params,
    register_model,
    train_model,
)


def test_model_registry_contains_all_expected_keys():
    expected = {"smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt", "asf", "asf_lite", "cmp"}
    assert expected.issubset(MODEL_REGISTRY.keys())


def test_build_model_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown model_key"):
        build_model(TrainConfig(model_key="nonexistent_arch"), pretrained=False)


def test_register_model_extension_point():
    dummy_key = "_test_dummy_linear"
    try:
        @register_model(dummy_key)
        def _build_dummy(cfg: TrainConfig, pretrained: bool) -> torch.nn.Module:
            return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 256 * 256, cfg.num_classes))

        m = build_model(TrainConfig(model_key=dummy_key, num_classes=10), pretrained=False)
        out = m(torch.randn(2, 3, 256, 256))
        assert out.shape == (2, 10)
    finally:
        MODEL_REGISTRY.pop(dummy_key, None)


@pytest.mark.parametrize("model_key", ["smallcnn", "resnet18", "mobilenetv3s", "efficientnetb0", "convnextt"])
def test_baseline_models_forward_shape(model_key: str):
    m = build_model(TrainConfig(model_key=model_key, num_classes=45), pretrained=False).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 45)


@pytest.mark.parametrize(
    "overrides",
    [
        {"asf_fixed_fusion": True},
        {"asf_fixed_fusion": False, "asf_gate_descriptor": "gap", "asf_out_init": "small"},
        {"asf_fixed_fusion": False, "asf_gate_descriptor": "energy", "asf_out_init": "small"},
        {"asf_fixed_fusion": False, "asf_detail_mode": "conv", "asf_out_init": "zero"},
    ],
)
def test_asf_strategies_forward_and_entropy(overrides: dict):
    cfg = TrainConfig(model_key="asf", num_classes=45, **overrides)
    m = build_model(cfg, pretrained=False)
    assert isinstance(m, ASFResNet18)
    x = torch.randn(2, 3, 256, 256)
    out = m(x)
    assert out.shape == (2, 45)
    ent = m.gate_entropy_loss()
    assert ent.ndim == 0
    if not cfg.asf_fixed_fusion:
        w = m.gate_weights(x)
        assert w.shape == (2, 8)
        assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("agg", ["lse", "hardmax"])
def test_cmp_strategies_forward_margin_and_losses(agg: str):
    cfg = TrainConfig(
        model_key="cmp",
        num_classes=5,
        cmp_n_proto=2,
        cmp_agg=agg,
        cmp_confusion=True,
        cmp_diversity=True,
        cmp_pair_loss=True,
        cmp_pair_margin=0.15,
    )
    m = build_model(cfg, pretrained=False)
    assert isinstance(m, CMPResNet18)
    w = np.zeros((5, 5), dtype=np.float32)
    w[0, 1] = 0.8
    m.set_confusion(w)

    x = torch.randn(4, 3, 256, 256)
    y = torch.tensor([0, 1, 2, 3])
    logits = m(x, y)
    assert logits.shape == (4, 5)
    e = m.embeddings(x)
    assert m.pair_loss(e, y).item() >= 0.0
    assert m.diversity_loss().item() >= 0.0


def test_kd_distiller_keeps_teacher_frozen_during_train():
    torch.manual_seed(0)
    cfg = TrainConfig(model_key="smallcnn", num_classes=5, epochs=1, batch_size=8, kd_alpha=0.5, kd_T=4.0, amp=False)
    stu = build_model(cfg, pretrained=False)
    tea = build_model(TrainConfig(model_key="smallcnn", num_classes=5), pretrained=False)
    w_before = [p.detach().clone() for p in tea.parameters()]
    bn_before = [b.detach().clone() for b in tea.buffers()]

    dl = DataLoader(TensorDataset(torch.randn(16, 3, 64, 64), torch.randint(0, 5, (16,))), batch_size=8)
    _stu_out, hist = train_model(stu, cfg, dl, None, "cpu", verbose=False, teacher=tea)
    assert all(torch.equal(a, b) for a, b in zip(w_before, tea.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(bn_before, tea.buffers(), strict=True))
    assert hist["history"][0]["aux"]["kd"] > 0.0


def test_kd_distiller_blend_loss_matches_alpha():
    stu_logits = torch.randn(4, 5, requires_grad=True)
    tea = build_model(TrainConfig(model_key="smallcnn", num_classes=5), pretrained=False)
    distiller = KDDistiller(tea, device="cpu", alpha=0.5, temperature=4.0)
    base = torch.tensor(1.0)
    total, kd_l = distiller.blend_loss(base, stu_logits, torch.randn(4, 3, 64, 64))
    assert total.item() == pytest.approx(0.5 * 1.0 + 0.5 * kd_l.item(), rel=1e-5)


def test_early_stopper_tracks_best_and_triggers_patience(tmp_path: Path):
    m = build_model(TrainConfig(model_key="smallcnn", num_classes=5), pretrained=False)
    cfg = TrainConfig(model_key="smallcnn", num_classes=5)
    ckpt = tmp_path / "best.pt"
    stopper = EarlyStopper(patience=2, state_out=str(ckpt))

    assert stopper.update(0.5, 0, m, cfg) is False
    assert stopper.update(0.4, 1, m, cfg) is False
    assert stopper.update(0.3, 2, m, cfg) is True
    assert stopper.best_epoch == 0
    assert ckpt.exists()


def test_existing_checkpoints_strict_load():
    cases = [
        ("resnet18", {}, ROOT / "outputs/resisc45-final-baselines-b0-b4-3-seeds/ckpt_resnet18_s2.pt"),
        (
            "asf",
            {"asf_out_init": "small", "asf_gate_ent": 0.05},
            ROOT / "outputs/resisc45-final-proposed-asf-cmp-3-seeds/ckpt_ASF_A2fix_ent_s1.pt",
        ),
        (
            "asf",
            {"asf_out_init": "small", "asf_gate_ent": 0.05},
            ROOT / "outputs/resisc45-kd-final-seeds-1-2/ckpt_asf_kd_s1.pt",
        ),
        ("cmp", {"cmp_n_proto": 2}, ROOT / "outputs/resisc45-final-proposed-asf-cmp-3-seeds/ckpt_CMP_C2_s1.pt"),
        (
            "cmp",
            {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp"},
            ROOT / "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/ckpt_cmp_exp_ce_s1.pt",
        ),
        (
            "cmp",
            {"cmp_n_proto": 2, "cmp_agg": "hardmax", "cmp_scale_mode": "exp"},
            ROOT / "outputs/resisc45-cmp-exp-3-seeds-and-test-eval/ckpt_cmp_exp_kd_s1.pt",
        ),
    ]
    present = [(mk, ov, p) for mk, ov, p in cases if p.exists()]
    if not present:
        pytest.skip("outputs/ checkpoints not present locally")
    x = torch.randn(1, 3, 256, 256)
    for mk, ov, ck_path in present:
        m = build_model(TrainConfig(model_key=mk, num_classes=45, **ov), pretrained=False)
        sd = torch.load(ck_path, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["state_dict"], strict=True)
        m.eval()
        with torch.no_grad():
            assert m(x).shape == (1, 45)


def test_common_mean_std_helpers():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert s == pytest.approx(1.0)
    d = mean_std_dict([0.95, 0.96, 0.97], digits=4)
    assert d == {"mean": 0.96, "std": 0.01}


def test_corruption_strategies_preserve_rgb_size():
    from PIL import ImageEnhance, ImageFilter

    img = Image.new("RGB", (64, 64), color=(120, 150, 180))
    blurred = img.filter(ImageFilter.GaussianBlur(1.0))
    bright = ImageEnhance.Brightness(img).enhance(1.2)
    assert blurred.size == (64, 64) and bright.size == (64, 64)


def test_asf_lite_forward_gates_and_param_budget():
    m = build_model(TrainConfig(model_key="asf_lite", num_classes=45, asf_out_init="small"), pretrained=False)
    assert isinstance(m, ASFLiteResNet18)
    x = torch.randn(2, 3, 256, 256)
    out = m(x)
    assert out.shape == (2, 45)
    ent = m.gate_entropy_loss()
    assert ent.ndim == 0
    w = m.gate_weights(x)
    assert w.shape == (2, 8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert count_params(m)["total"] < 11_600_000


def test_cmp_exp_scale_breaks_kd_softmax_ceiling():
    m_lin = build_model(TrainConfig(model_key="cmp", num_classes=45, cmp_n_proto=2, cmp_scale_mode="linear"), False)
    m_exp = build_model(TrainConfig(model_key="cmp", num_classes=45, cmp_n_proto=2, cmp_scale_mode="exp"), False)
    assert isinstance(m_lin, CMPResNet18) and isinstance(m_exp, CMPResNet18)
    assert m_lin.current_kappa().item() == pytest.approx(np.log(10.0), rel=1e-4)
    assert m_exp.current_kappa().item() == pytest.approx(10.0, rel=1e-4)

    # Ở cos=1 vs 0 với T=4: linear cap ~13.3%, exp mode vượt xa trần teacher
    e = torch.zeros(1, 512)
    e[0, 0] = 1.0
    m_exp.prototypes.data.zero_()
    m_exp.prototypes.data[0, 0] = 1.0
    with torch.no_grad():
        m_exp.logit_scale.fill_(np.log(30.0))
        probs_t4 = torch.softmax(m_exp.logits_from_emb(e) / 4.0, dim=1)
    assert probs_t4.max().item() > 0.90
