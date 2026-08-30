"""The seam-1 port: a CLIP-family encoder presented as the framework's
`x -> (logits, probs, preds, uncertainty)` contract.

The claim these tests exist to pin is not "CLIP works" -- it is that the
confidence-axis apparatus transfers to a component it was never written for,
without touching the apparatus. So the load-bearing test here is the one that
runs `QueryConfidence` -- the gradient-free instrument -- against a zero-shot
head and gets the same guarantees it gives on the forgery detector.

No data, no network, no GPU, no `open_clip`: the encoder is a stub, which is
the point -- if these tests needed the real checkpoint they could not pin the
contract.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from trustfake.attacks import QueryConfidence
from trustfake.metrics.calibration import fit_temperature
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.torch.clip import (
    CLIP_MEAN,
    CLIP_STD,
    OPENAI_CLIP_LOGIT_SCALE,
    CLIPZeroShotClassifier,
    build_text_prototypes,
)
from trustfake.models.wrapper import BaseWrapper

NC = 3
DIM = 12
SIZE = 16


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class _StubVisual(nn.Module):
    """Stands in for `open_clip`'s `model.visual`: (B, 3, H, W) -> (B, DIM),
    not unit-norm, so the classifier's own normalization is exercised."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * 3.0


def _classifier(logit_scale: float = 5.0, **kwargs) -> CLIPZeroShotClassifier:
    torch.manual_seed(0)
    prototypes = torch.randn(NC, DIM)
    return CLIPZeroShotClassifier(
        visual=_StubVisual(),
        prototypes=prototypes,
        logit_scale=logit_scale,
        **kwargs,
    )


def _wrapped(clf: CLIPZeroShotClassifier, temperature: float = 1.0) -> BaseWrapper:
    return BaseWrapper(
        # Identity is REQUIRED here: the CLIP module normalizes internally.
        normalization_layer=nn.Identity(),
        model=clf,
        loss_fn=None,
        uncertainty_score=MultiClassMaxProbability(),
        temperature=temperature,
    ).eval()


# --------------------------------------------------------------------------
# The head is what it claims to be
# --------------------------------------------------------------------------


def test_logits_are_scaled_cosine_similarity():
    """`logits = logit_scale * cos(f(x), P)`, so every logit is bounded by
    +/- logit_scale -- which is what makes the scale, not the encoder, the
    thing that sets the confidence distribution."""
    clf = _classifier(logit_scale=5.0)
    x = torch.rand(8, 3, SIZE, SIZE)

    logits = clf(x)
    expected = 5.0 * clf.encode(x) @ clf.prototypes.T

    assert logits.shape == (8, NC)
    assert torch.allclose(logits, expected, atol=1e-6)
    assert logits.abs().max() <= 5.0 + 1e-5


def test_prototypes_are_unit_norm_and_carry_no_gradient():
    clf = _classifier()
    assert torch.allclose(clf.prototypes.norm(dim=-1), torch.ones(NC), atol=1e-6)
    assert not clf.prototypes.requires_grad
    assert all(not p.requires_grad for p in clf.visual.parameters())


def test_encoder_freeze_does_not_block_input_gradients():
    """Seam 1 audits a *pretrained* encoder, so its parameters are frozen --
    but a gradient attack differentiates w.r.t. the input, not the weights.
    If freezing broke that, half the attack battery would silently stop
    working against this model."""
    clf = _classifier()
    x = torch.rand(4, 3, SIZE, SIZE, requires_grad=True)

    clf(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# The normalization trap
# --------------------------------------------------------------------------


def test_normalization_happens_inside_the_model_exactly_once():
    """The encoder's preprocessing travels with the encoder. This is the one
    deviation from the framework's convention (datamodule owns normalization)
    and it is deliberate: CLIP is not free to be normalized any way you like.
    """
    clf = _classifier()
    x = torch.rand(4, 3, SIZE, SIZE)

    mean = torch.tensor(CLIP_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, -1, 1, 1)
    manual = F.normalize(clf.visual((x - mean) / std).float(), dim=-1)

    assert torch.allclose(clf.encode(x), manual, atol=1e-6)


def test_double_normalization_is_a_silent_number_change():
    """Pins the failure the config comment warns about: putting an ImageNet
    Normalize in the datamodule on top of this model raises nothing, returns
    the right shapes, and returns different numbers. Nothing downstream can
    detect it, which is why it is written down here."""
    clf = _classifier()
    x = torch.rand(8, 3, SIZE, SIZE)

    correct = _wrapped(clf)
    doubled = BaseWrapper(
        normalization_layer=nn.Identity(),
        model=clf,
        loss_fn=None,
        uncertainty_score=MultiClassMaxProbability(),
    ).eval()
    imagenet = nn.Sequential(
        # what configs/training/datamodule/sid_set.yaml applies by default
        _Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    doubled.normalization_layer = imagenet

    assert not torch.allclose(correct(x)[0], doubled(x)[0], atol=1e-3)


class _Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------
# Checkpointing: the text tower is a build-time dependency, never load-time
# --------------------------------------------------------------------------


def test_state_dict_round_trips_without_the_text_tower():
    clf = _classifier()
    x = torch.rand(4, 3, SIZE, SIZE)
    before = clf(x)

    state = clf.state_dict()
    assert "prototypes" in state, "prototypes must be a buffer, not rebuilt"

    reloaded = CLIPZeroShotClassifier(
        visual=_StubVisual(),
        prototypes=torch.zeros(NC, DIM).fill_(1e-3),
        logit_scale=clf.logit_scale,
    )
    reloaded.load_state_dict(state)

    assert torch.allclose(reloaded(x), before, atol=1e-6)


def test_rejects_malformed_construction():
    with pytest.raises(ValueError, match="prototypes must be"):
        CLIPZeroShotClassifier(visual=_StubVisual(), prototypes=torch.randn(DIM))
    with pytest.raises(ValueError, match="logit_scale must be"):
        CLIPZeroShotClassifier(
            visual=_StubVisual(), prototypes=torch.randn(NC, DIM), logit_scale=0.0
        )


def test_prompt_ensembling_averages_normalized_embeddings():
    """A verbose template must not dominate a terse one by embedding
    magnitude, so the mean is taken over normalized embeddings."""

    class _FakeCLIP:
        def encode_text(self, tokens):
            # token[i] scales a fixed direction: same direction, wildly
            # different magnitudes across templates.
            return torch.tensor([[1.0, 0.0], [100.0, 0.0]])[: len(tokens)]

    prototypes = build_text_prototypes(
        _FakeCLIP(),
        tokenizer=lambda prompts: torch.zeros(len(prompts), 4),
        classnames=["a"],
        templates=("{}", "a photo of {}."),
    )

    assert torch.allclose(prototypes, torch.tensor([[1.0, 0.0]]), atol=1e-6)


# --------------------------------------------------------------------------
# The transfer claim: the apparatus runs unchanged
# --------------------------------------------------------------------------


def test_satisfies_the_wrapper_contract():
    model = _wrapped(_classifier())
    x = torch.rand(8, 3, SIZE, SIZE)

    logits, probs, preds, unc = model(x)

    assert logits.shape == (8, NC)
    assert probs.shape == (8, NC)
    assert torch.allclose(probs.sum(1), torch.ones(8), atol=1e-6)
    assert preds.shape == (8,)
    assert torch.equal(preds, probs.argmax(1))
    assert unc.shape == (8,)
    assert ((unc >= 0) & (unc <= 1)).all()


def test_query_confidence_transfers_to_the_zero_shot_head():
    """THE transfer test. `QueryConfidence` was written against a forgery
    detector and knows nothing about CLIP; it needs only forward passes and
    the wrapper's own uncertainty. If its three guarantees hold here -- no
    gradients touched, argmax preserved, uncertainty moved -- then the
    confidence axis is measurable on a vision encoder with no change to the
    instrument."""
    model = _wrapped(_classifier(logit_scale=5.0))
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    x = torch.rand(16, 3, SIZE, SIZE)

    clean_preds, clean_unc = model(x)[2], model(x)[3]
    adv = (
        QueryConfidence(eps=8 / 255, n_queries=200, direction="over", seed=0)
        .run(model, x)
        .perturbed
    )
    adv_preds, adv_unc = model(adv)[2], model(adv)[3]

    # gradient-free
    assert all(p.grad is None for p in model.parameters())
    # a CONFIDENCE attack: the prediction axis does not move
    assert torch.equal(clean_preds, adv_preds)
    # ... while the confidence axis does, in the requested direction
    assert adv_unc.mean() < clean_unc.mean()
    # and stays inside the budget
    assert (adv - x).abs().max() <= 8 / 255 + 1e-6


# --------------------------------------------------------------------------
# logit_scale is a confidence-axis variable
# --------------------------------------------------------------------------


def test_native_logit_scale_degenerates_the_confidence_signal():
    """CLIP's shipped scale (~100) saturates the softmax: 1-MSP collapses
    toward zero and a failure-detection AUROC computed on it is measuring the
    temperature, not the model. Temperature is monotone, so accuracy is
    identical either way -- which is exactly why this is invisible to an
    accuracy check and has to be pinned as a test."""
    clf = _classifier(logit_scale=OPENAI_CLIP_LOGIT_SCALE)
    x = torch.rand(64, 3, SIZE, SIZE)

    saturated = _wrapped(clf, temperature=1.0)
    _, _, hot_preds, hot_unc = saturated(x)

    # A temperature fitted on a calib split restores a usable spread. The
    # labels have to be labels the model gets *wrong* sometimes -- fitting
    # against the model's own argmax is a no-op by construction, because a
    # model is never wrong about its own prediction, so NLL is already
    # minimal at T = 1. A real calib split contains errors; this stands in
    # for one.
    logits = saturated(x)[0]
    targets = torch.randint(0, NC, (x.shape[0],), generator=_gen())
    temperature = fit_temperature(logits, targets)
    calibrated = _wrapped(clf, temperature=temperature)
    _, _, cold_preds, cold_unc = calibrated(x)

    assert hot_unc.mean() < 1e-3, "expected saturation at the native scale"
    assert temperature > 10.0, "a saturated head needs a large T to spread out"
    # degenerate -> non-degenerate. The magnitude of the spread is a property
    # of the encoder (a stub barely varies across random inputs); what is
    # pinned here is that at the native scale there is no signal to rank at
    # all, and after calibration there is one.
    assert hot_unc.std() < 1e-6, "1-MSP is constant, so FD-AUROC ranks nothing"
    assert cold_unc.mean() > 0.1, "calibrated confidence must leave the floor"
    assert cold_unc.std() > hot_unc.std()
    assert torch.equal(hot_preds, cold_preds), "temperature cannot move accuracy"
