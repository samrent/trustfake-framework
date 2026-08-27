"""QueryConfidence is the gradient-free confidence attack -- the instrument
that tells real evidential robustness from gradient masking. Its guarantees
are load-bearing, so they are pinned here."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trustfake.attacks import QueryConfidence
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper

NC = 3


def _trained_model():
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 6, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(6 * 16 * 16, NC),
    )
    model = BaseWrapper(
        normalization_layer=nn.Identity(),
        model=net,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )
    x = torch.rand(256, 3, 16, 16)
    y = x.flatten(1)[:, :NC].argmax(1)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(150):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x)[0], y).backward()
        opt.step()
    model.eval()
    return model


def test_uses_no_gradients():
    """The whole point: if it touched the model's gradient it could be fooled
    by the same masking it exists to detect."""
    model = _trained_model()
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    x = torch.rand(16, 3, 16, 16)

    QueryConfidence(n_queries=50, seed=1).run(model, x)

    assert all(p.grad is None for p in model.parameters())


def test_preserves_the_argmax():
    """It is a CONFIDENCE attack: accuracy is unchanged by construction, so
    any selective-risk change is attributable to the confidence axis alone."""
    model = _trained_model()
    x = torch.rand(32, 3, 16, 16)
    with torch.no_grad():
        clean_preds = model(x)[2]

    adv = QueryConfidence(n_queries=200, direction="over", seed=1).run(model, x)
    with torch.no_grad():
        adv_preds = model(adv.perturbed)[2]

    assert torch.equal(adv_preds, clean_preds)


@pytest.mark.parametrize("direction", ["over", "under"])
def test_moves_uncertainty_the_named_way(direction):
    model = _trained_model()
    x = torch.rand(32, 3, 16, 16)
    with torch.no_grad():
        u_before = model(x)[3].mean()

    adv = QueryConfidence(eps=8 / 255, n_queries=300, direction=direction, seed=1).run(
        model, x
    )
    with torch.no_grad():
        u_after = model(adv.perturbed)[3].mean()

    if direction == "over":
        assert u_after < u_before  # confidence up
    else:
        assert u_after > u_before  # confidence down


def test_respects_the_budget():
    model = _trained_model()
    x = torch.rand(16, 3, 16, 16)
    eps = 8 / 255

    adv = QueryConfidence(eps=eps, n_queries=100, seed=1).run(model, x)

    assert (adv.perturbed - x).abs().max() <= eps + 1e-6


def test_is_deterministic_for_a_seed():
    model = _trained_model()
    x = torch.rand(16, 3, 16, 16)

    a = QueryConfidence(n_queries=80, seed=7).run(model, x).perturbed
    b = QueryConfidence(n_queries=80, seed=7).run(model, x).perturbed

    assert torch.equal(a, b)


def test_name_encodes_direction():
    assert QueryConfidence(direction="over").name == "query_overconf"
    assert QueryConfidence(direction="under").name == "query_underconf"
    assert QueryConfidence().uses_labels is False
