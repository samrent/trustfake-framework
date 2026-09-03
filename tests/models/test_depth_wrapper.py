"""`DepthConsistencyWrapper`: identical to `BaseWrapper` with a probability
score, and the only path to the depth-consistency score otherwise.

The two properties that would silently corrupt numbers if broken: with a
probability-only score the wrapper must reproduce `BaseWrapper` bit for bit
(that is scoring number one of three), and with a depth-aware score the
teacher must be run on the SAME pixels the classifier saw -- the attacked
ones -- which is why `outputs_from_logits` returns None. Stub teacher, no
network, no data.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from trustfake.attacks import PGD, QueryConfidence, UncertaintyFGSM
from trustfake.depth import FakeDepthTeacher
from trustfake.losses.depth import ssi_l1_per_image
from trustfake.metrics.uncertainty import (
    CombinedDepthScore,
    DepthConsistencyScore,
    MultiClassMaxProbability,
)
from trustfake.models.torch import resnet18
from trustfake.models.wrapper import BaseWrapper, DepthConsistencyWrapper

SIZE = 32
NC = 3


class _RecordingTeacher(FakeDepthTeacher):
    """Remembers what it was asked about."""

    def __init__(self, **kw):
        super().__init__(output_size=SIZE // 2, input_size=SIZE, multiple=1, **kw)
        self.calls = 0
        self.last_input = None

    def forward(self, x):
        self.calls += 1
        self.last_input = x.detach().clone()
        return super().forward(x)


def _model():
    torch.manual_seed(0)
    return resnet18(num_classes=NC, depth_head=True, depth_head_width=8).eval()


def _norm():
    return nn.Identity()


def _wrap(score, model=None, teacher=None, **kw):
    model = model if model is not None else _model()
    teacher = teacher if teacher is not None else _RecordingTeacher()
    return DepthConsistencyWrapper(
        normalization_layer=_norm(),
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=score,
        teacher=teacher,
        **kw,
    ).eval()


def _x(n=4, seed=1):
    torch.manual_seed(seed)
    return torch.rand(n, 3, SIZE, SIZE)


def test_probability_score_reproduces_base_wrapper_and_never_runs_the_teacher():
    model = _model()
    teacher = _RecordingTeacher()
    depth = _wrap(MultiClassMaxProbability(), model=model, teacher=teacher)
    base = BaseWrapper(
        normalization_layer=_norm(),
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    ).eval()
    x = _x()
    a, b = depth(x), base(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))
    assert teacher.calls == 0
    assert depth.outputs_from_logits(a[0]) is not None
    assert not depth.consumes_depth


def test_depth_score_is_the_residual_between_head_and_teacher():
    teacher = _RecordingTeacher()
    wrapper = _wrap(DepthConsistencyScore(), teacher=teacher)
    x = _x()
    logits, probs, preds, u = wrapper(x)
    _, head = wrapper.forward_with_depth(x)
    expected = ssi_l1_per_image(head, teacher(x))
    assert torch.allclose(u, expected, atol=1e-6)
    assert u.shape == (4,) and torch.isfinite(u).all()
    assert torch.equal(logits, wrapper.model(x))
    assert torch.equal(preds, probs.argmax(1))
    assert torch.equal(teacher.last_input, x)


def test_depth_score_forces_a_reforward_on_the_perturbed_batch():
    wrapper = _wrap(DepthConsistencyScore())
    assert wrapper.outputs_from_logits(torch.randn(4, NC)) is None


def test_depth_score_needs_a_depth_path():
    torch.manual_seed(0)
    plain = resnet18(num_classes=NC)
    with pytest.raises(ValueError, match="depth path"):
        _wrap(DepthConsistencyScore(), model=plain)


def test_teacher_sees_the_attacked_pixels():
    """The whole point of the score under attack: the reference is the
    teacher's view of the adversarial input, not of the clean one."""
    teacher = _RecordingTeacher()
    wrapper = _wrap(DepthConsistencyScore(), teacher=teacher)
    x = _x(n=6)
    y = torch.randint(0, NC, (6,))
    adv = PGD(eps=0.05, steps=3, seed=0)(wrapper, x, y)
    wrapper(adv)
    assert torch.equal(teacher.last_input, adv)
    assert (adv - x).abs().max() > 0


def test_gradient_through_the_teacher_is_kept_by_default():
    """teacher_grad=True: the score's gradient w.r.t. the input includes the
    teacher's response; with teacher_grad=False it is the student half only.
    Both must be non-zero, and they must differ."""

    def grad(teacher_grad):
        wrapper = _wrap(DepthConsistencyScore(), teacher_grad=teacher_grad)
        x = _x(n=2).requires_grad_(True)
        wrapper(x)[3].sum().backward()
        return x.grad.clone()

    full, student = grad(True), grad(False)
    assert full.abs().sum() > 0 and student.abs().sum() > 0
    assert not torch.allclose(full, student)


def test_attack_contract_essentials_hold():
    """The subset of tests/attacks/test_attack_contracts.py an attack needs
    from a wrapper: deep-copyable, parameters untouched, training mode
    restored, and the confidence attacks actually run on the depth score."""
    wrapper = _wrap(DepthConsistencyScore())
    x, y = _x(n=5), torch.randint(0, NC, (5,))
    before = [p.clone() for p in wrapper.parameters()]
    twin = copy.deepcopy(wrapper)

    adv_a = UncertaintyFGSM(eps=0.05)(wrapper, x, y)
    adv_b = UncertaintyFGSM(eps=0.05)(twin, x, y)
    assert torch.allclose(adv_a, adv_b)
    assert all(
        torch.equal(a, b) for a, b in zip(before, wrapper.parameters(), strict=True)
    )
    assert not wrapper.training

    result = QueryConfidence(eps=0.05, n_queries=8, direction="under", seed=0).run(
        wrapper, x, y
    )
    assert torch.isfinite(wrapper(result.perturbed)[3]).all()


def test_combined_score_end_to_end_through_the_wrapper():
    score = CombinedDepthScore(weight=0.5)
    wrapper = _wrap(score)
    calib = _x(n=16, seed=5)
    msp, residual = wrapper.score_components(calib)
    assert msp.shape == residual.shape == (16,)
    score.fit_reference(msp, residual)
    u = wrapper(_x(n=4))[3]
    assert u.shape == (4,) and (u >= 0).all() and (u <= 1).all()


def test_score_components_respect_the_temperature():
    wrapper = _wrap(DepthConsistencyScore())
    x = _x(n=4)
    wrapper.temperature = 1.0
    msp_cold, res_cold = wrapper.score_components(x)
    wrapper.temperature = 5.0
    msp_warm, res_warm = wrapper.score_components(x)
    assert not torch.allclose(msp_cold, msp_warm)
    assert torch.allclose(res_cold, res_warm)


def test_transfer_mode_scores_by_probability_only_while_attacking():
    """Outside `attacking()` both modes are the same wrapper; inside, transfer
    returns 1 - max prob and never calls the teacher, white-box does both."""
    for mode in ("white_box", "transfer"):
        teacher = _RecordingTeacher()
        wrapper = _wrap(DepthConsistencyScore(), teacher=teacher, attack_scoring=mode)
        x = _x()
        outside = wrapper(x)
        assert teacher.calls == 1
        with wrapper.attacking():
            inside = wrapper(x)
        assert not wrapper._attacking
        if mode == "transfer":
            assert teacher.calls == 1
            assert torch.allclose(inside[3], 1 - inside[1].max(1).values)
        else:
            assert teacher.calls == 2
            assert torch.equal(inside[3], outside[3])
        assert torch.equal(inside[0], outside[0])
    with pytest.raises(ValueError, match="attack_scoring"):
        _wrap(DepthConsistencyScore(), attack_scoring="grey")


def test_teacher_is_not_part_of_the_checkpoint():
    wrapper = _wrap(DepthConsistencyScore())
    assert not any("teacher" in k for k in wrapper.state_dict())
    assert not any("_teacher" in n for n, _ in wrapper.named_modules())
