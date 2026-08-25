"""Tests for the resampling and sensor-side conditions.

Each one is checked on the property it exists for, not merely on "the tensor
changed": downscale and blur must REMOVE high-frequency content (an aliasing
downsample would add some instead, which on a forensic task is the opposite
of the intended condition), and the noise condition must be reproducible
without borrowing the global RNG and without depending on how many times it
has already been called.
"""

import pytest
import torch

from trustfake.corruptions import Downscale, GaussianBlur, GaussianNoise


def _high_frequency_energy(images: torch.Tensor) -> float:
    """Mean squared horizontal first difference: a cheap stand-in for the
    detail a resampler or a blur destroys."""
    return images.diff(dim=-1).pow(2).mean().item()


def _checkerboard(size: int = 32) -> torch.Tensor:
    """Maximal high-frequency content -- the hardest case for a resampler,
    and the one that aliases if the downsample is not antialiased."""
    grid = torch.arange(size)
    board = ((grid[:, None] + grid[None, :]) % 2).to(torch.float32)
    return board.view(1, 1, size, size).expand(2, 3, size, size).contiguous()


def test_downscale_destroys_high_frequency_detail():
    board = _checkerboard()
    corrupted = Downscale(factor=2)(None, board)

    assert _high_frequency_energy(corrupted) < 0.1 * _high_frequency_energy(board)


def test_downscale_does_not_alias_the_detail_back_in():
    """Without antialiasing the shrink point-samples the grid and folds the
    checkerboard back as a spurious low-frequency pattern of full contrast --
    a corruption that CREATES structure a forensic detector might read. The
    antialiased shrink must leave a nearly flat image instead."""
    board = _checkerboard()
    corrupted = Downscale(factor=2)(None, board)

    assert corrupted.std().item() < 0.1
    assert abs(corrupted.mean().item() - board.mean().item()) < 0.05


def test_downscale_severity_is_monotone_in_the_factor():
    board = _checkerboard()
    energies = [
        _high_frequency_energy(Downscale(factor=f)(None, board)) for f in (2, 4, 8)
    ]

    assert energies[0] >= energies[1] >= energies[2]


def test_downscale_preserves_the_input_grid():
    images = torch.rand(2, 3, 30, 21)
    assert Downscale(factor=3)(None, images).shape == images.shape


def test_gaussian_noise_is_reproducible_without_the_global_rng():
    """The row must not move because something else in the run consumed
    randomness first -- otherwise adding a dropout layer elsewhere silently
    changes a reported corruption number."""
    images = torch.full((2, 3, 16, 16), 0.5)

    corruption = GaussianNoise(sigma=0.05, seed=7)
    first = corruption(None, images)
    torch.manual_seed(1234)  # global stream churned in between
    _ = torch.randn(1000)
    second = corruption(None, images)

    assert torch.equal(first, second)


def test_gaussian_noise_repeats_itself_on_a_second_call():
    """THE determinism property, and the one an advancing generator broke: a
    second `trainer.test()` on the same eval module calls the same instance
    again, so a stream carried across calls reports a different accuracy for
    the same model, data and condition (0.2917 -> 0.3750 on a smoke run).
    The condition has to be a pure function of its input, like every other
    stochastic component here (`attacks/pgd.py::_random_start` re-seeds
    inside the call for the same reason)."""
    images = torch.full((2, 3, 16, 16), 0.5)
    corruption = GaussianNoise(sigma=0.05, seed=0)

    first = corruption(None, images)
    second = corruption(None, images)
    third = corruption.run(None, images).perturbed

    assert torch.equal(first, second)
    assert torch.equal(first, third)


def test_gaussian_noise_seed_still_selects_the_field():
    """Deterministic must not mean fixed: a different seed is a different
    draw, or the `seed` argument would be decoration."""
    images = torch.full((2, 3, 16, 16), 0.5)

    assert not torch.equal(
        GaussianNoise(sigma=0.05, seed=0)(None, images),
        GaussianNoise(sigma=0.05, seed=1)(None, images),
    )


def test_gaussian_noise_has_the_requested_scale():
    images = torch.full((8, 3, 32, 32), 0.5)
    corrupted = GaussianNoise(sigma=0.05, seed=0)(None, images)

    # Clipping at 0.5 +- 10 sigma is inactive, so the empirical std is the
    # parameter, not a truncated version of it.
    assert abs((corrupted - images).std().item() - 0.05) < 5e-3


def test_gaussian_noise_is_clipped_into_the_valid_range():
    images = torch.full((4, 3, 16, 16), 1.0)
    corrupted = GaussianNoise(sigma=0.5, seed=0)(None, images)

    assert corrupted.max().item() <= 1.0
    assert corrupted.min().item() >= 0.0


def test_gaussian_blur_removes_detail_and_keeps_the_mean():
    board = _checkerboard()
    corrupted = GaussianBlur(sigma=1.5)(None, board)

    assert _high_frequency_energy(corrupted) < 0.1 * _high_frequency_energy(board)
    # Reflection padding, not zeros: a zero-padded blur darkens the border
    # and adds a frame a detector could learn instead of the blur.
    assert abs(corrupted.mean().item() - board.mean().item()) < 0.02


def test_gaussian_blur_kernel_covers_two_standard_deviations():
    for sigma, expected in ((0.5, 3), (1.0, 5), (1.5, 7), (2.0, 9)):
        blur = GaussianBlur(sigma=sigma)
        assert blur.kernel_size == expected
        assert blur.kernel_size % 2 == 1


def test_gaussian_blur_severity_is_monotone_in_sigma():
    board = _checkerboard()
    energies = [
        _high_frequency_energy(GaussianBlur(sigma=s)(None, board))
        for s in (0.5, 1.0, 2.0)
    ]

    assert energies[0] > energies[1] > energies[2]


# --------------------------------------------------------------------------
# no-op parameterisations: the clean row wearing a corruption's name
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [(Downscale, {"factor": 1.0}), (GaussianNoise, {"sigma": 0.0})],
    ids=["downscale_factor_1", "gaussian_noise_sigma_0"],
)
def test_a_no_op_parameter_is_refused(factory, kwargs):
    """`factor=1` and `sigma=0` are not mild rungs of a ladder, they are the
    CLEAN condition under a distinct metric prefix. A results table then
    gains a corruption row that is bit-identical to the clean one, and a
    reader cannot tell it from a detector that shrugged off the condition --
    the most flattering error a robustness table can make. `codec.py` guards
    the same shape of no-op by pinning WebP to `lossless=False`."""
    with pytest.raises(ValueError, match="no-op"):
        factory(**kwargs)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [(Downscale, {"factor": 1.0}), (GaussianNoise, {"sigma": 0.0})],
    ids=["downscale_factor_1", "gaussian_noise_sigma_0"],
)
def test_the_refused_parameter_would_really_have_been_a_no_op(factory, kwargs):
    """The premise behind the refusal, asserted rather than assumed: run the
    corruption's own arithmetic at the rejected parameter and the output is
    bit-identical to the input. If a future change made these parameters
    genuinely lossy, this test fails and the guard should be revisited."""
    images = torch.rand(2, 3, 16, 16)

    if factory is Downscale:
        corruption = Downscale(factor=2.0)
        corruption.factor = 1.0
    else:
        corruption = GaussianNoise(sigma=0.05, seed=0)
        corruption.sigma = 0.0

    assert torch.equal(corruption.corrupt(images), images)
