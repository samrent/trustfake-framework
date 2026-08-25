"""Tests for the log-Dirichlet discrepancy D (EV-AT L_REA / adversary)."""

import pytest
import torch

from trustfake.losses import LogDirichletDivergence


@pytest.mark.parametrize("mode", ["l2", "kl"])
def test_metric_modes_vanish_at_equality(mode):
    """l2 and kl are proper divergences: zero iff the inputs match."""
    d = LogDirichletDivergence(4, mode=mode)
    eta = torch.randn(8, 4)
    assert float(d(eta, eta.clone())) == pytest.approx(0.0, abs=1e-6)
    assert float(d(eta, eta + 0.5 * torch.randn(8, 4))) > 0.0


def test_ikl_minimised_at_equality_over_adversary():
    """IKL is a training objective, not a metric: its CE term equals the
    entropy at s_n = s_m, so it does not vanish -- but perturbing the
    adversarial side away from the clean side only increases it."""
    d = LogDirichletDivergence(4, mode="ikl")
    eta = torch.randn(8, 4)
    at_equal = float(d(eta, eta.clone()))
    perturbed = float(d(eta, eta + 0.6 * torch.randn(8, 4)))
    assert perturbed >= at_equal


def test_ikl_gradient_pushes_adversary_toward_clean():
    """Minimising IKL w.r.t. the adversarial branch must reduce the gap to
    the clean posterior mean."""
    d = LogDirichletDivergence(3, mode="ikl")
    eta_m = torch.randn(16, 3)
    eta_n = (eta_m + torch.randn(16, 3)).clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([eta_n], lr=0.5)
    gap0 = (torch.softmax(eta_n, 1) - torch.softmax(eta_m, 1)).abs().sum().item()
    for _ in range(50):
        opt.zero_grad()
        d(eta_m, eta_n).backward()
        opt.step()
    gap1 = (torch.softmax(eta_n, 1) - torch.softmax(eta_m, 1)).abs().sum().item()
    assert gap1 < gap0


def test_pairwise_diff_is_antisymmetric():
    d = LogDirichletDivergence(4, mode="ikl")
    o = torch.randn(5, 4)
    diff = d._pairwise_diff(o)
    assert torch.allclose(diff, -diff.transpose(1, 2))
    assert torch.allclose(diff[:, 0, 1], o[:, 0] - o[:, 1])


def test_global_stats_update_and_use():
    d = LogDirichletDivergence(4, mode="ikl")
    assert not bool(d.stats_ready)
    probs = torch.softmax(torch.randn(64, 4), 1)
    targets = torch.randint(0, 4, (64,))
    d.update_global_stats(probs, targets)
    assert bool(d.stats_ready)
    # each class row stays a valid distribution
    assert torch.allclose(d.class_means.sum(1), torch.ones(4), atol=1e-5)
    # forward with targets uses the global weight and stays finite
    eta = torch.randn(8, 4)
    val = d(eta, eta + 0.3 * torch.randn(8, 4), targets[:8])
    assert torch.isfinite(val)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        LogDirichletDivergence(4, mode="js")
