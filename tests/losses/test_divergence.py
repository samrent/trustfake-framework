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


@pytest.mark.parametrize("mode", ["ikl", "kl", "l2"])
def test_self_discrepancy_is_zero(mode):
    """D(eta, eta) must be 0, or `beta` means a different thing per mode and
    an ablation across modes compares scales rather than mechanisms. IKL's
    cross-entropy term does not vanish at identity on its own -- it becomes
    the entropy of the clean side, about ln(C) -- so its floor is subtracted."""
    torch.manual_seed(0)
    eta = torch.rand(8, 3).abs() + 0.5
    divergence = LogDirichletDivergence(num_classes=3, mode=mode)

    assert float(divergence(eta, eta)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("mode", ["ikl", "kl", "l2"])
def test_discrepancy_is_positive_when_the_sides_differ(mode):
    """Subtracting the floor must not flatten the signal it sits on."""
    torch.manual_seed(0)
    eta = torch.rand(8, 3).abs() + 0.5
    divergence = LogDirichletDivergence(num_classes=3, mode=mode)

    assert float(divergence(eta, eta * 2.0)) > 1e-4


def test_subtracting_the_floor_leaves_every_gradient_untouched():
    """The floor is built from the already-detached clean side, so training is
    bit-identical -- only the reported number changes. If this ever fails, the
    fix has started altering optimisation, which it must not."""
    torch.manual_seed(0)
    divergence = LogDirichletDivergence(num_classes=3, mode="ikl")
    eta_m = torch.randn(8, 3, requires_grad=True)
    eta_n = torch.randn(8, 3, requires_grad=True)

    loss = divergence(eta_m, eta_n)
    grads = torch.autograd.grad(loss, [eta_m, eta_n], retain_graph=True)

    s = torch.softmax(eta_m, dim=1).detach()
    floor = -(s * torch.log(s + 1e-12)).sum(dim=1).mean()
    shifted = torch.autograd.grad(loss + floor, [eta_m, eta_n])

    for a, b in zip(grads, shifted, strict=True):
        assert torch.equal(a, b)
