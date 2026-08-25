"""Tests for the evidential clean loss L_EV. No data, no network, no GPU."""

import pytest
import torch
from torch.distributions import Dirichlet, kl_divergence

from trustfake.losses import EvidentialLoss
from trustfake.losses.evidential import dirichlet_uniform_kl


def test_dirichlet_uniform_kl_matches_reference():
    """Closed-form KL[Dir(alpha) || Dir(1)] must match torch.distributions."""
    torch.manual_seed(0)
    alpha = torch.rand(16, 5) * 5 + 1.0
    ref = kl_divergence(Dirichlet(alpha), Dirichlet(torch.ones(5)))
    assert torch.allclose(dirichlet_uniform_kl(alpha), ref, atol=1e-5)


def test_kl_zero_at_uniform():
    assert torch.allclose(
        dirichlet_uniform_kl(torch.ones(4, 3)), torch.zeros(4), atol=1e-6
    )


def test_nll_lower_for_correct_confident_evidence():
    loss = EvidentialLoss(num_classes=3)
    loss.set_epoch(0)  # kl weight 0 -> pure NLL
    y = torch.tensor([0, 1, 2])
    good = torch.tensor([[50.0, 1, 1], [1, 50.0, 1], [1, 1, 50.0]])
    bad = good[[1, 2, 0]]
    assert loss(good, y) < loss(bad, y)


def test_kl_weight_anneals():
    loss = EvidentialLoss(num_classes=3, lambda_max=1.0, anneal_epochs=10)
    loss.set_epoch(0)
    assert loss.kl_weight == 0.0
    loss.set_epoch(5)
    assert loss.kl_weight == pytest.approx(0.5)
    loss.set_epoch(10)
    assert loss.kl_weight == pytest.approx(1.0)
    loss.set_epoch(50)  # capped
    assert loss.kl_weight == pytest.approx(1.0)


def test_kl_regulariser_penalises_wrong_class_evidence():
    """With the KL term on, evidence spread onto wrong classes costs more."""
    loss = EvidentialLoss(num_classes=3, lambda_max=1.0, anneal_epochs=1)
    loss.set_epoch(1)  # full weight
    y = torch.tensor([0])
    concentrated = torch.tensor([[20.0, 1.0, 1.0]])  # evidence only on the truth
    spread = torch.tensor([[20.0, 10.0, 10.0]])  # plus misleading evidence
    assert loss(concentrated, y) < loss(spread, y)


def test_gradients_flow():
    loss = EvidentialLoss(num_classes=3)
    loss.set_epoch(5)
    alpha = (torch.rand(8, 3) * 3 + 1).requires_grad_(True)
    loss(alpha, torch.randint(0, 3, (8,))).backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
