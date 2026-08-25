"""Tests for the evidence-targeted PGD adversary (EV-AT inner max)."""

import copy

import pytest
import torch
import torch.nn as nn

from trustfake.attacks import EvidenceTargetedPGD
from trustfake.losses import EvidentialLoss, LogDirichletDivergence
from trustfake.metrics.uncertainty import (
    EvidentialPredictiveEntropy,
    MultiClassMaxProbability,
)
from trustfake.models.wrapper import BaseWrapper, EvidentialWrapper

EPS = 0.05
INPUT_SHAPE = (3, 8, 8)


@pytest.fixture(scope="module")
def evidential_model():
    """A small evidential model trained enough to carry real evidence."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    m = EvidentialWrapper(
        normalization_layer=nn.Identity(),
        model=net,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=EvidentialPredictiveEntropy(),
    )
    x = torch.rand(48, *INPUT_SHAPE)
    y = x.flatten(1)[:, :3].argmax(1)  # a learnable rule
    loss = EvidentialLoss(3)
    loss.set_epoch(5)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        alpha, _ = m.dirichlet(m(x)[0])
        loss(alpha, y).backward()
        opt.step()
    m.eval()
    return m, x, y


@pytest.mark.parametrize("mode", ["ikl", "kl", "l2"])
def test_stays_in_eps_ball_and_range(mode, evidential_model):
    m, x, y = evidential_model
    adv = EvidenceTargetedPGD(eps=EPS, steps=10, mode=mode)(m, x, y)
    assert (adv - x).abs().max().item() <= EPS + 1e-6
    assert adv.min() >= -1e-6 and adv.max() <= 1 + 1e-6


@pytest.mark.parametrize("mode", ["ikl", "kl", "l2"])
def test_increases_posterior_drift(mode, evidential_model):
    """The attack must raise the discrepancy D between the clean and
    adversarial posteriors -- that is its objective."""
    m, x, y = evidential_model
    adv = EvidenceTargetedPGD(eps=EPS, steps=10, mode=mode)(m, x, y)
    d = LogDirichletDivergence(3, mode=mode)
    with torch.no_grad():
        _, eta_clean = m.dirichlet(m(x)[0])
        _, eta_adv = m.dirichlet(m(adv)[0])
    assert float(d(eta_clean, eta_adv, y)) > float(d(eta_clean, eta_clean, y))


def test_does_not_mutate_inputs_or_weights(evidential_model):
    m, x, y = evidential_model
    x0 = x.clone()
    params = [p.clone() for p in m.parameters()]
    EvidenceTargetedPGD(eps=EPS, steps=5)(m, x, y)
    assert torch.equal(x, x0)
    for a, b in zip(params, m.parameters(), strict=True):
        assert torch.equal(a, b)


def test_output_detached_and_deterministic(evidential_model):
    m, x, y = evidential_model
    atk = EvidenceTargetedPGD(eps=EPS, steps=5, seed=1)
    adv = atk(m, x, y)
    assert not adv.requires_grad
    assert torch.allclose(adv, atk(copy.deepcopy(m), x, y))


def test_zero_eps_returns_input(evidential_model):
    m, x, y = evidential_model
    assert torch.allclose(EvidenceTargetedPGD(eps=0.0)(m, x, y), x)


def test_requires_evidential_wrapper():
    b = BaseWrapper(
        normalization_layer=nn.Identity(),
        model=nn.Linear(4, 3),
        loss_fn=None,
        uncertainty_score=MultiClassMaxProbability(),
    )
    with pytest.raises(TypeError, match="not an evidential wrapper"):
        EvidenceTargetedPGD()(b, torch.randn(3, 4), torch.tensor([0, 1, 2]))


def test_shared_divergence_is_used(evidential_model):
    """When a divergence is injected (as the training module does to share
    IKL global stats), the attack uses that instance, not a fresh one."""
    m, x, y = evidential_model
    shared = LogDirichletDivergence(3, mode="ikl")
    atk = EvidenceTargetedPGD(eps=EPS, steps=3, divergence=shared)
    atk(m, x, y)
    assert atk._divergence is shared
