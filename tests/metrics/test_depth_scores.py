"""The depth-consistency and combined rejection scores.

What must hold: the residual is the training loss (affine-invariant, floor
of 1.0 for a head that learned nothing), it is a finite per-image vector
with a gradient, and the combined score refuses to run before it has been
fitted on a calib split -- an unnormalised sum would rank like whichever
component is larger and read as a result. No data, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from trustfake.metrics.uncertainty import (
    CombinedDepthScore,
    DepthAwareScore,
    DepthConsistencyScore,
)
from trustfake.metrics.uncertainty.depth import depth_consistency_residual, ecdf_interp


def _maps(n=6, size=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, 1, size, size)


def _probs(n=6, seed=3):
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(n, 3), dim=1)


def test_residual_is_zero_for_an_affine_copy_and_one_for_a_constant():
    t = _maps()
    assert torch.allclose(
        depth_consistency_residual(2 * t + 1, t), torch.zeros(6), atol=1e-5
    )
    assert torch.allclose(
        depth_consistency_residual(torch.ones_like(t), t), torch.ones(6), atol=1e-5
    )


def test_residual_refuses_a_batch_mismatch():
    with pytest.raises(ValueError, match="batch sizes"):
        depth_consistency_residual(_maps(n=4), _maps(n=6))


def test_consistency_score_is_a_finite_per_image_vector():
    score = DepthConsistencyScore()
    u = score(_probs(), _maps(seed=1), _maps())
    assert u.shape == (6,) and u.dtype == torch.float32
    assert torch.isfinite(u).all()
    score.reset()
    assert score.compute().numel() == 0


def test_consistency_score_ignores_probabilities():
    a = DepthConsistencyScore()(_probs(seed=1), _maps(seed=1), _maps())
    b = DepthConsistencyScore()(_probs(seed=2), _maps(seed=1), _maps())
    assert torch.equal(a, b)


def test_consistency_score_carries_a_gradient():
    """UncertaintyFGSM differentiates the wrapper's 4th output w.r.t. the
    input; a score without a graph would make that attack a no-op."""
    pred = _maps(seed=1).requires_grad_(True)
    DepthConsistencyScore()(_probs(), pred, _maps()).sum().backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_random_maps_give_distinct_operating_points():
    u = DepthConsistencyScore()(_probs(n=32), _maps(n=32, seed=1), _maps(n=32))
    assert u.unique().numel() == 32


def test_scores_are_marked_depth_aware():
    assert isinstance(DepthConsistencyScore(), DepthAwareScore)
    assert isinstance(CombinedDepthScore(), DepthAwareScore)


# --------------------------------------------------------------------------
# ECDF and the combined score
# --------------------------------------------------------------------------


def test_ecdf_is_monotone_and_hits_the_reference_points():
    ref = torch.tensor([0.0, 1.0, 2.0, 4.0])
    at_ref = ecdf_interp(ref, ref)
    assert torch.allclose(at_ref, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))
    mid = ecdf_interp(ref, torch.tensor([0.5, 3.0]))
    assert torch.allclose(mid, torch.tensor([1 / 6, 2 / 3 + 1 / 6]))
    outside = ecdf_interp(ref, torch.tensor([-5.0, 9.0]))
    assert torch.allclose(outside, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="empty reference"):
        ecdf_interp(torch.empty(0), torch.tensor([1.0]))


def test_ecdf_has_a_gradient_between_reference_points():
    ref = torch.tensor([0.0, 1.0, 2.0])
    v = torch.tensor([0.25, 1.5], requires_grad=True)
    ecdf_interp(ref, v).sum().backward()
    assert (v.grad > 0).all()


def test_combined_refuses_to_score_before_it_is_fitted():
    with pytest.raises(ValueError, match="fit_reference"):
        CombinedDepthScore()(_probs(), _maps(seed=1), _maps())


def test_combined_refuses_bad_weights_and_bad_references():
    with pytest.raises(ValueError, match="weight"):
        CombinedDepthScore(weight=1.5)
    score = CombinedDepthScore()
    with pytest.raises(ValueError, match="non-empty"):
        score.fit_reference(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError, match="non-finite"):
        score.fit_reference(torch.tensor([0.1, float("nan")]), torch.tensor([1.0, 2.0]))


def _fitted(weight):
    score = CombinedDepthScore(weight=weight)
    torch.manual_seed(9)
    score.fit_reference(torch.rand(200) * 0.7, torch.rand(200) * 2.0)
    return score


def test_combined_values_are_in_unit_interval():
    u = _fitted(0.5)(_probs(n=16), _maps(n=16, seed=1), _maps(n=16))
    assert u.shape == (16,)
    assert (u >= 0).all() and (u <= 1).all()


def test_weight_endpoints_reproduce_each_component_ranking():
    p, dp, dt = _probs(n=16), _maps(n=16, seed=1), _maps(n=16)
    msp = 1 - p.max(dim=1).values
    residual = depth_consistency_residual(dp, dt)
    only_msp = _fitted(1.0)(p, dp, dt)
    only_depth = _fitted(0.0)(p, dp, dt)
    assert torch.equal(only_msp.argsort(), msp.argsort())
    assert torch.equal(only_depth.argsort(), residual.argsort())


def test_combined_gradient_reaches_the_depth_prediction():
    dp = _maps(n=8, seed=1).requires_grad_(True)
    _fitted(0.5)(_probs(n=8), dp, _maps(n=8)).sum().backward()
    assert dp.grad is not None and dp.grad.abs().sum() > 0


def test_calib_reference_never_enters_the_state_dict():
    """The checkpoint is the model's; a calibration fitted at evaluation
    must not travel with it."""
    score = _fitted(0.5)
    assert score.fitted
    assert not any("ref" in k for k in score.state_dict())
