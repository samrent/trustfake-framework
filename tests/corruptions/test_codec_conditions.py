"""Tests for the lossy re-encoding conditions (JPEG, WebP).

The properties that matter are the ones a wrong implementation would break
silently: the uint8 grid snap that a real file always costs, monotonicity in
quality, and near-identity at q=100 -- because a round-trip that changed
nothing (lossless mode, or a decode that returned its input) would report a
corruption column identical to clean and nobody would notice.
"""

import numpy as np
import pytest
import torch

from trustfake.corruptions import JPEGCompression, WebPCompression


def _smooth_ramp(size: int = 32) -> torch.Tensor:
    """A grey horizontal ramp: R == G == B, so chroma subsampling costs
    nothing and the residual error is the codec's luma quantisation alone.
    Random noise is the worst possible probe here -- it is unencodable, and
    even q=100 mangles it."""
    ramp = torch.linspace(0.0, 1.0, size).view(1, 1, 1, size)
    return ramp.expand(2, 3, size, size).contiguous()


def _mean_error(corruption, images: torch.Tensor) -> float:
    return (corruption(None, images) - images).abs().mean().item()


def test_jpeg_low_quality_changes_the_tensor():
    images = _smooth_ramp()
    corrupted = JPEGCompression(quality=10)(None, images)

    assert not torch.allclose(corrupted, images, atol=1e-3)
    assert (corrupted - images).abs().mean().item() > 5e-3


def test_jpeg_q100_is_near_identity_on_an_encodable_image():
    """q=100 is still lossy (DCT rounding survives it), so this is
    "idempotent-ish", not idempotent -- but a q=100 round-trip that moved a
    smooth ramp by more than a couple of grey levels would mean the encoder
    is not receiving what it was handed."""
    images = _smooth_ramp()
    corrupted = JPEGCompression(quality=100)(None, images)

    assert (corrupted - images).abs().max().item() <= 0.01  # ~2.5 grey levels
    assert not torch.equal(corrupted, images)


def test_jpeg_error_grows_as_quality_falls():
    images = _smooth_ramp()
    errors = [_mean_error(JPEGCompression(quality=q), images) for q in (100, 40, 10)]

    assert errors[0] < errors[1] < errors[2]


def test_codec_output_lands_on_the_uint8_lattice():
    """The quantisation is part of the condition, not an artifact of it: a
    codec cannot carry more than 8 bits per channel, so an output off the
    1/255 grid would mean the round-trip never touched a real encoder."""
    images = _smooth_ramp()
    for corruption in (JPEGCompression(quality=40), WebPCompression(quality=60)):
        corrupted = corruption(None, images) * 255.0
        assert torch.allclose(corrupted, corrupted.round(), atol=1e-4), corruption.name


def test_webp_lossy_is_actually_lossy():
    """`lossless=False` is explicit in the implementation for this reason: a
    lossless WebP round-trip is a no-op and would report a corruption row
    that is a copy of clean."""
    images = _smooth_ramp()
    corrupted = WebPCompression(quality=30)(None, images)

    assert not torch.allclose(corrupted, images, atol=1e-3)


def test_codecs_are_deterministic():
    images = _smooth_ramp()
    for corruption in (JPEGCompression(quality=40), WebPCompression(quality=80)):
        assert torch.equal(corruption(None, images), corruption(None, images))


def test_greyscale_batch_round_trips():
    """WebP has no greyscale profile; a 1-channel batch must be promoted and
    reduced rather than raising deep inside the encoder."""
    images = _smooth_ramp()[:, :1]
    for corruption in (JPEGCompression(quality=50), WebPCompression(quality=50)):
        corrupted = corruption(None, images)
        assert corrupted.shape == images.shape


def test_codec_rejects_a_batch_that_is_not_an_image():
    with pytest.raises(ValueError, match="1 .*or 3.*channels"):
        JPEGCompression(quality=50)(None, torch.rand(2, 5, 8, 8))
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        JPEGCompression(quality=50)(None, torch.rand(3, 8, 8))


def test_codec_preserves_dtype_and_moves_back_from_cpu():
    """The round-trip runs on CPU in uint8; the caller must get its own
    dtype back, or a float64 pipeline silently becomes float32."""
    images = _smooth_ramp().to(torch.float64)
    corrupted = JPEGCompression(quality=70)(None, images)

    assert corrupted.dtype == torch.float64
    assert np.isfinite(corrupted.numpy()).all()
