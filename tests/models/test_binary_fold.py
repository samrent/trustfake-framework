"""The 3-class -> binary fold used to score a SID-Set model on FakeClue.

The property everything else rests on is that `softmax(log p) == p`, so the
harness downstream sees exactly the probabilities the fold intended. If that
breaks, every FakeClue metric is computed on a quietly different
distribution.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trustfake.attacks import PGD
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.torch.binary_fold import BinaryFoldClassifier
from trustfake.models.wrapper import BaseWrapper


class _ThreeClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x):
        return self.lin(x.flatten(1))


def _wrapped(model):
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    ).eval()


def test_fold_is_one_minus_p_real():
    """p_fake = 1 - P(real), the repo-wide definition in
    trustfake.metrics.moderation."""
    torch.manual_seed(0)
    inner = _ThreeClass()
    folded = BinaryFoldClassifier(inner)
    x = torch.rand(16, 4)

    inner_probs = torch.softmax(inner(x), dim=1)
    out = torch.softmax(folded(x), dim=1)

    assert torch.allclose(out[:, 0], inner_probs[:, 0], atol=1e-6)
    assert torch.allclose(out[:, 1], 1.0 - inner_probs[:, 0], atol=1e-6)


def test_softmax_of_log_probs_recovers_the_probabilities():
    """THE invariant. The fold returns log-probabilities as 'logits', which is
    only harmless because softmax inverts it exactly."""
    torch.manual_seed(1)
    folded = BinaryFoldClassifier(_ThreeClass())
    x = torch.rand(32, 4)

    logits = folded(x)
    probs = torch.softmax(logits, dim=1)

    assert torch.allclose(probs, logits.exp(), atol=1e-6)
    assert torch.allclose(probs.sum(1), torch.ones(32), atol=1e-6)


def test_output_has_two_columns_with_real_first():
    folded = BinaryFoldClassifier(_ThreeClass())
    assert folded(torch.rand(8, 4)).shape == (8, 2)


def test_prediction_agrees_with_the_three_class_argmax():
    """A model that calls an image real must still call it real after folding,
    and one that calls it synthetic or tampered must call it fake."""
    torch.manual_seed(2)
    inner = _ThreeClass()
    folded = _wrapped(BinaryFoldClassifier(inner))
    x = torch.rand(64, 4)

    three = torch.softmax(inner(x), dim=1).argmax(1)
    binary = folded(x)[2]

    assert torch.equal(binary, (three != 0).long())


def test_saturated_probabilities_do_not_produce_nan():
    """P(real) == 1 exactly would give log(0) = -inf on the fake column and
    NaN gradients, and a gradient attack would fail with no diagnostic."""

    class _Saturated(nn.Module):
        def forward(self, x):
            out = torch.full((x.shape[0], 3), -1e9)
            out[:, 0] = 1e9
            return out

    folded = BinaryFoldClassifier(_Saturated())
    logits = folded(torch.rand(4, 4))
    assert torch.isfinite(logits).all()


def test_gradients_flow_through_the_fold():
    """Attacks differentiate the folded logits w.r.t. the input."""
    folded = BinaryFoldClassifier(_ThreeClass())
    x = torch.rand(8, 4, requires_grad=True)
    folded(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_pgd_runs_against_a_folded_model():
    torch.manual_seed(3)
    model = _wrapped(BinaryFoldClassifier(_ThreeClass()))
    x = torch.rand(16, 4)
    y = torch.randint(0, 2, (16,))

    adv = PGD(eps=8 / 255, steps=5, seed=0).run(model, x, y).perturbed

    assert torch.isfinite(adv).all()
    assert (adv - x).abs().max() <= 8 / 255 + 1e-6


def test_rejects_a_one_column_model():
    class _OneClass(nn.Module):
        def forward(self, x):
            return torch.rand(x.shape[0], 1)

    with pytest.raises(ValueError, match="C>=2"):
        BinaryFoldClassifier(_OneClass())(torch.rand(4, 4))


def test_rejects_out_of_range_real_class():
    with pytest.raises(ValueError, match="real_class"):
        BinaryFoldClassifier(_ThreeClass(), real_class=5)(torch.rand(4, 4))
