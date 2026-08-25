"""Shared fixtures for adversarial attack contract tests."""

import pytest
import torch
import torch.nn as nn

from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper

NUM_CLASSES = 4
INPUT_SHAPE = (3, 8, 8)
BATCH_SIZE = 5


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


@pytest.fixture
def model() -> BaseWrapper:
    """A minimal, real TrustFakeWrapper: identity normalization + linear
    classifier. Big enough to exercise gradients, small enough to be a fast
    unit-test fixture."""
    torch.manual_seed(0)
    return BaseWrapper(
        normalization_layer=_Identity(),
        model=_TinyClassifier(),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(BATCH_SIZE, *INPUT_SHAPE)


@pytest.fixture
def targets() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
