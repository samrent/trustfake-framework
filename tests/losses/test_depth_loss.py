"""The scale-and-shift-invariant depth loss and the depth frame it shares.

The load-bearing property is affine invariance: the teacher's relative depth
is defined only up to a per-image scale and shift, so a loss that is not
invariant to both would train the head toward an arbitrary gauge and the
depth-consistency residual would then measure the gauge rather than the
geometry. The second property is the fixed floor -- a constant prediction
scores exactly 1.0 -- because that is what makes the residual comparable
across images and against a calibration threshold. No GPU, no data.
"""

from __future__ import annotations

import pytest
import torch

from trustfake.losses import ScaleShiftInvariantL1, normalize_depth, ssi_l1_per_image
from trustfake.losses.depth import resize_depth


def _maps(n=3, size=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, 1, size, size)


def test_normalized_frame_has_zero_median_and_unit_mad():
    d = normalize_depth(_maps() * 5 + 3)
    flat = d.flatten(1)
    assert torch.allclose(flat.median(dim=1).values, torch.zeros(3), atol=1e-6)
    assert torch.allclose(flat.abs().mean(dim=1), torch.ones(3), atol=1e-5)
    assert d.dtype == torch.float32


def test_normalization_is_idempotent():
    """The precomputed targets are stored in the frame and the loss re-applies
    it; applying it twice must equal applying it once."""
    d = _maps()
    once = normalize_depth(d)
    assert torch.allclose(normalize_depth(once), once, atol=1e-5)


def test_loss_is_invariant_to_an_affine_change_of_either_operand():
    t = _maps()
    p = _maps(seed=1)
    base = ssi_l1_per_image(p, t)
    assert torch.allclose(ssi_l1_per_image(3.0 * p + 7.0, t), base, atol=1e-5)
    assert torch.allclose(ssi_l1_per_image(p, 0.5 * t - 2.0), base, atol=1e-5)


def test_prediction_equal_to_affine_target_scores_zero():
    t = _maps()
    assert torch.allclose(ssi_l1_per_image(4.0 * t - 1.0, t), torch.zeros(3), atol=1e-5)


def test_constant_prediction_scores_exactly_one():
    """The floor: a head that has learned nothing scores 1.0 on every image,
    regardless of the image."""
    t = _maps()
    ones = torch.full_like(t, 3.0)
    assert torch.allclose(ssi_l1_per_image(ones, t), torch.ones(3), atol=1e-5)


def test_shape_mismatch_is_refused_not_resampled():
    with pytest.raises(ValueError, match="identical shapes"):
        ssi_l1_per_image(_maps(size=8), _maps(size=16))


def test_three_dimensional_maps_are_accepted_and_others_refused():
    t = _maps()
    assert ssi_l1_per_image(t.squeeze(1), t).shape == (3,)
    with pytest.raises(ValueError, match="single-channel"):
        ssi_l1_per_image(torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8))


def test_float16_target_is_upcast_and_result_is_float32():
    """Targets are stored as float16 on disk; the loss must not compute its
    statistics in half precision."""
    t = _maps()
    out = ssi_l1_per_image(_maps(seed=1), t.half())
    assert out.dtype == torch.float32
    assert torch.allclose(out, ssi_l1_per_image(_maps(seed=1), t), atol=1e-2)


def test_gradient_flows_to_both_operands():
    """An adaptive attack needs d(residual)/d(teacher output) as well as the
    student half; the training loss needs the student half."""
    p = _maps(seed=1).requires_grad_(True)
    t = _maps().requires_grad_(True)
    ssi_l1_per_image(p, t).sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert t.grad is not None and t.grad.abs().sum() > 0


def test_loss_is_computed_in_fp32_under_autocast():
    """Training runs bf16-mixed; a median taken at 8 mantissa bits is not
    the median. The loss must opt out of autocast like the evidential head."""
    p, t = _maps(seed=1), _maps()
    plain = ScaleShiftInvariantL1()(p, t)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed = ScaleShiftInvariantL1()(p, t)
    assert mixed.dtype == torch.float32
    assert torch.allclose(plain, mixed, atol=1e-6)


def test_reduction_modes():
    p, t = _maps(seed=1), _maps()
    per = ScaleShiftInvariantL1(reduction="none")(p, t)
    assert per.shape == (3,)
    assert torch.allclose(ScaleShiftInvariantL1()(p, t), per.mean())
    with pytest.raises(ValueError, match="reduction"):
        ScaleShiftInvariantL1(reduction="sum")


def test_lower_median_is_bit_equal_to_torch_median_including_ties():
    """`Tensor.median(dim=)` is on torch's deterministic-mode throw list on
    CUDA (indices output); the sort-based replacement must be exactly the
    same statistic, on random maps and on maps full of ties."""
    from trustfake.losses.depth import _lower_median

    torch.manual_seed(3)
    random = torch.randn(5, 64)
    ties = torch.randint(0, 3, (5, 63)).float()
    for flat in (random, ties, torch.zeros(2, 10)):
        assert torch.equal(_lower_median(flat), flat.median(dim=1, keepdim=True).values)


def test_normalisation_uses_no_op_from_the_deterministic_throw_list(monkeypatch):
    """Guard against a future edit reintroducing `median(dim=...)`."""
    calls = []
    original = torch.Tensor.median

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "median", spy)
    with torch.autograd.set_detect_anomaly(False):
        normalize_depth(_maps())
    assert not any(a or k for a, k in calls), "median(dim=...) is back"


def test_resize_is_identity_at_the_same_size_and_resamples_otherwise():
    d = _maps(size=16)
    assert torch.equal(resize_depth(d, 16), d.float())
    assert resize_depth(d, 8).shape == (3, 1, 8, 8)
    assert resize_depth(d, (4, 8)).shape == (3, 1, 4, 8)
