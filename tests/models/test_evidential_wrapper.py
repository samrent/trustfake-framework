"""Tests for the evidential wrapper (arXiv:2607.03075, Sec. 3.2). No data,
no network, no GPU."""

import pytest
import torch
import torch.nn as nn

from trustfake.metrics.uncertainty import EvidentialPredictiveEntropy
from trustfake.models.wrapper import EvidentialWrapper

NUM_CLASSES = 3


def _wrapper(activation="softplus"):
    torch.manual_seed(0)
    return EvidentialWrapper(
        normalization_layer=nn.Identity(),
        model=nn.Linear(8, NUM_CLASSES),
        loss_fn=None,
        uncertainty_score=EvidentialPredictiveEntropy(),
        evidence_activation=activation,
    )


@pytest.mark.parametrize("activation", ["softplus", "exp", "relu"])
def test_dirichlet_construction(activation):
    """alpha = evidence + 1 >= 1, and pi_bar = alpha / S is a valid simplex
    point equal to softmax(log alpha)."""
    m = _wrapper(activation)
    x = torch.randn(16, 8)
    logits, probs, preds, unc = m(x)

    alpha, eta = m.dirichlet(logits)
    assert (alpha >= 1.0 - 1e-6).all()
    assert torch.allclose(eta, torch.log(alpha))
    strength = alpha.sum(1, keepdim=True)
    assert torch.allclose(probs, alpha / strength, atol=1e-6)
    assert torch.allclose(probs, torch.softmax(eta, dim=1), atol=1e-6)
    assert torch.allclose(probs.sum(1), torch.ones(16), atol=1e-6)


def test_prediction_is_argmax_of_evidence():
    m = _wrapper()
    x = torch.randn(16, 8)
    logits, _, preds, _ = m(x)
    assert torch.equal(preds, m.evidence(logits).argmax(1))


def test_uncertainty_is_posterior_entropy():
    """u(x) = H[Cat(pi_bar)] (Eq. 1)."""
    m = _wrapper()
    _, probs, _, unc = m(torch.randn(16, 8))
    expected = -(probs * probs.clamp_min(1e-12).log()).sum(1)
    assert torch.allclose(unc, expected, atol=1e-6)


def test_more_evidence_means_lower_uncertainty():
    """Amplifying the input sharpens the evidence (more peaked pi_bar, larger
    strength), which lowers the predictive entropy. Compared on the SAME
    inputs scaled up, so the mean's direction is held and only its
    concentration changes."""
    m = _wrapper("exp")
    x = torch.randn(64, 8)
    diffuse = m(x)[3]
    concentrated = m(x * 4.0)[3]
    assert concentrated.mean() < diffuse.mean()


def test_temperature_scales_pi_bar_but_not_preds():
    """Temperature divides eta = log alpha before the softmax: valid post-hoc
    calibration of the posterior mean, argmax invariant."""
    m = _wrapper()
    x = torch.randn(16, 8)
    _, p1, pr1, _ = m(x)
    m.temperature = 2.5
    _, p2, pr2, _ = m(x)
    assert torch.equal(pr1, pr2)
    assert not torch.allclose(p1, p2)


def test_outputs_from_logits_matches_forward():
    """The attack accept-check path must reproduce forward for identity norm."""
    m = _wrapper()
    x = torch.randn(16, 8)
    logits, probs, preds, unc = m(x)
    logits2, probs2, preds2, unc2 = m.outputs_from_logits(logits)
    assert torch.allclose(probs, probs2) and torch.equal(preds, preds2)
    assert torch.allclose(unc, unc2)


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown evidence_activation"):
        _wrapper("sigmoid")
