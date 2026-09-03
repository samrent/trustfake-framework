"""The auxiliary depth decoder: shapes, no dropout, gradients. No data."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trustfake.models.torch import DepthHead


def test_output_is_at_half_the_input_resolution():
    head = DepthHead()
    # layer3 / layer4 maps of a 224 input
    out = head(torch.randn(2, 256, 14, 14), torch.randn(2, 512, 7, 7))
    assert out.shape == (2, 1, 112, 112)
    # ... and of a 64 input
    out = head(torch.randn(2, 256, 4, 4), torch.randn(2, 512, 2, 2))
    assert out.shape == (2, 1, 32, 32)


def test_head_has_no_dropout():
    """`MCDropoutWrapper._enable_dropout` flips every dropout module under
    the model into train mode at evaluation; a stochastic head would turn the
    depth-consistency score into noise."""
    assert not any(
        isinstance(m, nn.modules.dropout._DropoutNd) for m in DepthHead().modules()
    )


def test_bottleneck_widths_are_accepted():
    head = DepthHead(in_channels_l3=1024, in_channels_l4=2048, width=32)
    assert head(torch.randn(1, 1024, 4, 4), torch.randn(1, 2048, 2, 2)).shape == (
        1,
        1,
        32,
        32,
    )


def test_invalid_widths_are_refused():
    with pytest.raises(ValueError, match="width"):
        DepthHead(width=0)
    with pytest.raises(ValueError, match="num_upsamples"):
        DepthHead(num_upsamples=-1)


def test_head_upsamples_with_nearest_only(monkeypatch):
    """Bilinear/bicubic/linear interpolation has no deterministic CUDA
    backward; under the trainer's `deterministic: true` the first GPU step
    would raise. The CPU suite cannot see that, so the mode is pinned here."""
    import torch.nn.functional as F  # noqa: N812

    modes = []
    original = F.interpolate

    def spy(*args, **kwargs):
        modes.append(kwargs.get("mode", "nearest"))
        return original(*args, **kwargs)

    monkeypatch.setattr("trustfake.models.torch.depth_head.F.interpolate", spy)
    head = DepthHead(width=8)
    head(torch.randn(1, 256, 4, 4), torch.randn(1, 512, 2, 2)).sum().backward()
    assert modes and set(modes) == {"nearest"}


def test_head_runs_under_deterministic_algorithms():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        head = DepthHead(width=8)
        head(torch.randn(2, 256, 4, 4), torch.randn(2, 512, 2, 2)).sum().backward()
    finally:
        torch.use_deterministic_algorithms(prev)


def test_gradients_reach_both_feature_maps():
    head = DepthHead(width=16)
    f3 = torch.randn(2, 256, 4, 4, requires_grad=True)
    f4 = torch.randn(2, 512, 2, 2, requires_grad=True)
    head(f3, f4).sum().backward()
    assert f3.grad.abs().sum() > 0 and f4.grad.abs().sum() > 0
