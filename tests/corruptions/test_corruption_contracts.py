"""Generic contract tests for `ImageCorruption` implementations.

Deliberately NOT the attack battery in `tests/attacks/test_attack_contracts.py`:
that one asserts the epsilon ball, and a corruption has no budget to respect.
What a corruption promises instead is checked here -- it stays a valid image,
it leaves its input alone, it is model-independent, it actually changes
something, and its name carries its parameter so two rungs of the same ladder
cannot overwrite each other's metrics.
"""

import copy

import pytest
import torch

from trustfake.attacks import AttackFamily
from trustfake.corruptions import (
    Downscale,
    GaussianBlur,
    GaussianNoise,
    ImageCorruption,
    JPEGCompression,
    WebPCompression,
)

CLIP_MIN, CLIP_MAX = 0.0, 1.0

CORRUPTIONS = [
    JPEGCompression(quality=40),
    WebPCompression(quality=80),
    Downscale(factor=2),
    GaussianNoise(sigma=0.05, seed=0),
    GaussianBlur(sigma=1.0),
]

CORRUPTION_IDS = [corruption.name for corruption in CORRUPTIONS]


@pytest.fixture(params=CORRUPTIONS, ids=CORRUPTION_IDS)
def corruption(request):
    # A fresh instance per test: GaussianNoise carries a generator that
    # advances across calls, so a shared one would make these tests
    # order-dependent.
    return copy.deepcopy(request.param)


@pytest.fixture
def images() -> torch.Tensor:
    """A batch with real structure: a smooth ramp plus mild texture, so a
    codec has something to quantise and a blur something to remove."""
    torch.manual_seed(0)
    ramp = torch.linspace(0.0, 1.0, 32).view(1, 1, 1, 32).expand(4, 3, 32, 32)
    return (ramp + 0.05 * torch.randn(4, 3, 32, 32)).clamp(0.0, 1.0).contiguous()


def test_is_not_eps_bounded(corruption):
    """The declaration that separates a corruption from an attack: no budget.

    `eps` is infinite because a corruption is a distributional-shift
    condition, not a point in a ball -- reading a corruption row against an
    epsilon would be a category error.
    """
    assert corruption.eps == float("inf")
    assert corruption.family is AttackFamily.CORRUPTION
    assert corruption.norm == "none"
    assert corruption.uses_labels is False
    assert isinstance(corruption, ImageCorruption)


def test_name_encodes_its_parameter(corruption):
    """Two rungs of one ladder share a metric prefix and a log directory
    unless the parameter is in the name -- the second run would overwrite the
    first with no error anywhere."""
    assert corruption.name != type(corruption).__name__
    assert any(character.isdigit() for character in corruption.name)
    assert corruption.name == corruption.name.lower()
    assert "." not in corruption.name and "/" not in corruption.name


@pytest.mark.parametrize(
    ("factory", "first", "second"),
    [
        (JPEGCompression, {"quality": 40}, {"quality": 90}),
        (WebPCompression, {"quality": 50}, {"quality": 80}),
        (Downscale, {"factor": 2}, {"factor": 4}),
        (GaussianNoise, {"sigma": 0.05}, {"sigma": 0.1}),
        (GaussianBlur, {"sigma": 1.0}, {"sigma": 2.0}),
    ],
)
def test_two_settings_do_not_collide(factory, first, second):
    assert factory(**first).name != factory(**second).name


def test_stays_a_valid_image(corruption, images):
    corrupted = corruption(None, images)

    assert corrupted.shape == images.shape
    assert corrupted.dtype == images.dtype
    assert corrupted.device == images.device
    assert corrupted.min().item() >= CLIP_MIN - 1e-6
    assert corrupted.max().item() <= CLIP_MAX + 1e-6


def test_actually_corrupts(corruption, images):
    """A condition that changes nothing is a clean row wearing another name."""
    corrupted = corruption(None, images)
    assert not torch.allclose(corrupted, images, atol=1e-4)


def test_leaves_the_input_untouched(corruption, images):
    original = images.clone()
    corruption(None, images)
    assert torch.equal(images, original)


def test_is_model_independent(corruption, images):
    """A corruption never touches the model -- that is what makes it a
    transferable condition rather than an attack on one detector. Passing
    None as the model would explode if any implementation reached for it."""
    corrupted = corruption(None, images, None)
    assert corrupted.shape == images.shape


def test_run_returns_a_diagnostic_magnitude(corruption, images):
    """`effective_eps` is filled in as a measurement (how far the condition
    moved the batch), never as a budget it respected -- which is why it is
    allowed to exceed any epsilon an adversarial row would report."""
    result = corruption.run(None, images)

    assert result.perturbed.shape == images.shape
    assert result.effective_eps.shape == (images.shape[0],)
    expected = (result.perturbed - images).abs().flatten(1).amax(dim=1)
    assert torch.allclose(result.effective_eps, expected)
    # No accept-check forward exists, so the eval pipe must fall back to a
    # re-forward on `perturbed` rather than trusting stale logits.
    assert result.accepted_logits is None
    assert result.clean_preds is None


def test_run_matches_call(corruption, images):
    """`__call__` must be exactly `run(...).perturbed` -- the eval pipe uses
    `run`, adversarial training uses `__call__`, and a divergence would make
    the two report different conditions under one name."""
    first = corruption.run(None, images).perturbed
    if hasattr(corruption, "reset"):  # stochastic condition: same draw twice
        corruption.reset()
    assert torch.equal(first, corruption(None, images))


def test_survives_a_deepcopy(corruption, images):
    """The eval module clones metric collections per condition; a corruption
    carrying an unpicklable handle would only fail there, at run time."""
    clone = copy.deepcopy(corruption)
    assert clone.name == corruption.name
    assert clone(None, images).shape == images.shape


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (JPEGCompression, {"quality": 0}),
        (JPEGCompression, {"quality": 101}),
        (WebPCompression, {"quality": 0}),
        (Downscale, {"factor": 0.5}),
        (GaussianNoise, {"sigma": -0.1}),
        (GaussianBlur, {"sigma": 0.0}),
    ],
)
def test_rejects_a_meaningless_parameter(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
