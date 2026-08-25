"""Shared fixtures for adversarial attack contract tests."""

import pytest
import torch
import torch.nn as nn

from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper

NUM_CLASSES = 4
INPUT_SHAPE = (3, 8, 8)
BATCH_SIZE = 5
FLAT_FEATURES = 16


class _Identity(nn.Module):
    """Stand-in normalization layer that changes nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyClassifier(nn.Module):
    """Small model with a real gradient path, cheap enough to run every test."""

    def __init__(self):
        super().__init__()
        in_features = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]
        self.linear = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class _TinyConvNet(nn.Module):
    """Small conv net: a *curved* decision boundary with a real logit scale.

    `_TinyClassifier` is one `nn.Linear`, which is the regime in which a
    whole class of attack bugs hides. A linear model's decision boundary is
    a hyperplane, so a linearising attack solves it in one step and a
    badly-scaled one still lands on it; the margin is exactly linear in the
    input, so a gradient whose magnitude is wrong by a constant factor still
    points at the answer. Anything that depends on step size relative to the
    logit scale, or on the boundary curving away from the linearisation,
    reads as correct there and fails on the first real model.
    """

    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain
        channels, height, width = INPUT_SHAPE
        self.net = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * (height // 2) * (width // 2), NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * self.net(x)


def _build_wrapper(module: nn.Module) -> BaseWrapper:
    """Wrap a bare module the way every fixture here does."""
    return BaseWrapper(
        normalization_layer=_Identity(),
        model=module,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


@pytest.fixture
def model() -> BaseWrapper:
    """A minimal, real TrustFakeWrapper: identity normalization + linear
    classifier. Big enough to exercise gradients, small enough to be a fast
    unit-test fixture."""
    torch.manual_seed(0)
    return _build_wrapper(_TinyClassifier())


@pytest.fixture
def make_conv_model():
    """Factory for seeded conv-net wrappers, optionally with scaled logits.

    `gain` multiplies every logit by a constant. That cannot move any
    decision boundary -- argmax is invariant to a positive scale -- so every
    quantity a minimum-norm attack reports must be invariant to it too, which
    makes `gain` a free probe for scale-dependence.
    """

    def _make(seed: int = 0, gain: float = 1.0) -> BaseWrapper:
        torch.manual_seed(seed)
        return _build_wrapper(_TinyConvNet(gain=gain))

    return _make


@pytest.fixture
def conv_model(make_conv_model) -> BaseWrapper:
    """Conv-net wrapper -- use this wherever a curved boundary matters."""
    return make_conv_model()


@pytest.fixture
def conv_inputs() -> torch.Tensor:
    """A batch wide enough for a success *rate* to mean something."""
    torch.manual_seed(1)
    return torch.rand(32, *INPUT_SHAPE)


@pytest.fixture
def flat_model() -> BaseWrapper:
    """A model over rank-2 inputs, for the attacks' rank-agnosticism."""
    torch.manual_seed(0)
    return _build_wrapper(nn.Linear(FLAT_FEATURES, NUM_CLASSES))


@pytest.fixture
def flat_inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(6, FLAT_FEATURES)


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(BATCH_SIZE, *INPUT_SHAPE)


@pytest.fixture
def targets() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
