"""The CLIP probe detector -- Track B's backbone.

A different use of CLIP from `test_clip_zeroshot.py`: there the encoder is
the thing being attacked, here it is a feature extractor for the detector.
The properties that matter are that the head trains, the backbone does not,
and -- the one that would silently invalidate every robustness number -- that
freezing the backbone still lets input gradients through.

Stub encoder, no network, no `open_clip`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trustfake.attacks import PGD, QueryConfidence
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.torch.clip import CLIPProbeClassifier
from trustfake.models.wrapper import BaseWrapper

DIM, NC, SIZE = 16, 3, 16


class _StubVisual(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32, dim),
        )

    def forward(self, x):
        return self.net(x)


def _probe(**kwargs) -> CLIPProbeClassifier:
    torch.manual_seed(0)
    return CLIPProbeClassifier(
        visual=_StubVisual(), feature_dim=DIM, num_classes=NC, **kwargs
    )


def _wrapped(clf) -> BaseWrapper:
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=clf,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    ).eval()


# --------------------------------------------------------------------------
# The freeze must not break attacks
# --------------------------------------------------------------------------


def test_frozen_backbone_still_passes_input_gradients():
    """THE test. Freezing is requires_grad_(False) on parameters, not a
    no_grad region: a no_grad backbone cuts d(logits)/d(input) too, and the
    probe would then report perfect PGD robustness while being trivially
    attackable. That is a silent, catastrophic wrong number."""
    clf = _probe(freeze_backbone=True)
    x = torch.rand(4, 3, SIZE, SIZE, requires_grad=True)

    clf(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0, "no input gradient: every gradient attack is dead"


def test_pgd_actually_moves_a_frozen_probe():
    """The end-to-end consequence of the test above: a gradient attack must
    change the prediction of at least some samples."""
    clf = _probe(freeze_backbone=True)
    model = _wrapped(clf)
    x = torch.rand(32, 3, SIZE, SIZE)
    y = torch.randint(0, NC, (32,))

    adv = PGD(eps=8 / 255, steps=10, seed=0).run(model, x, y).perturbed

    assert (adv - x).abs().max() > 0, "PGD returned the input unchanged"
    assert (adv - x).abs().max() <= 8 / 255 + 1e-6


def test_frozen_backbone_takes_no_parameter_gradient():
    clf = _probe(freeze_backbone=True)
    x = torch.rand(4, 3, SIZE, SIZE)
    clf(x).sum().backward()

    assert all(p.grad is None for p in clf.visual.parameters())
    assert clf.head.weight.grad is not None, "the head must still train"


def test_unfrozen_backbone_takes_parameter_gradients():
    clf = _probe(freeze_backbone=False)
    x = torch.rand(4, 3, SIZE, SIZE)
    clf(x).sum().backward()

    assert any(p.grad is not None for p in clf.visual.parameters())


# --------------------------------------------------------------------------
# Shape, normalisation, guards
# --------------------------------------------------------------------------


def test_features_are_unit_norm_when_asked():
    clf = _probe(normalize_features=True)
    f = clf.encode(torch.rand(8, 3, SIZE, SIZE))
    assert torch.allclose(f.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_unnormalised_features_are_left_alone():
    clf = _probe(normalize_features=False)
    f = clf.encode(torch.rand(8, 3, SIZE, SIZE))
    assert not torch.allclose(f.norm(dim=-1), torch.ones(8), atol=1e-3)


def test_satisfies_the_wrapper_contract():
    model = _wrapped(_probe())
    logits, probs, preds, unc = model(torch.rand(8, 3, SIZE, SIZE))
    assert logits.shape == (8, NC)
    assert torch.allclose(probs.sum(1), torch.ones(8), atol=1e-6)
    assert torch.equal(preds, probs.argmax(1))
    assert unc.shape == (8,)


def test_head_learns_while_backbone_stays_put():
    """A probe is only cheap if the encoder really is fixed."""
    clf = _probe(freeze_backbone=True)
    before = [p.detach().clone() for p in clf.visual.parameters()]
    head_before = clf.head.weight.detach().clone()

    opt = torch.optim.Adam([p for p in clf.parameters() if p.requires_grad], 1e-2)
    x = torch.rand(32, 3, SIZE, SIZE)
    y = torch.randint(0, NC, (32,))
    for _ in range(5):
        opt.zero_grad()
        nn.functional.cross_entropy(clf(x), y).backward()
        opt.step()

    assert not torch.equal(clf.head.weight, head_before), "the head did not move"
    for p, b in zip(clf.visual.parameters(), before, strict=True):
        assert torch.equal(p, b), "a frozen backbone parameter moved"


def test_query_confidence_runs_against_the_probe():
    """The gradient-free instrument must work here too, so Track B models can
    be audited on the confidence axis like Track A's."""
    clf = _probe()
    model = _wrapped(clf)
    x = torch.rand(8, 3, SIZE, SIZE)
    clean = model(x)[2]

    adv = QueryConfidence(eps=8 / 255, n_queries=40, seed=0).run(model, x).perturbed

    assert torch.equal(clean, model(adv)[2]), "argmax must be preserved"


def test_rejects_degenerate_class_count():
    with pytest.raises(ValueError, match="num_classes"):
        CLIPProbeClassifier(visual=_StubVisual(), feature_dim=DIM, num_classes=1)
