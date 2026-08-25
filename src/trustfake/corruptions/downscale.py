"""Resampling condition: shrink and restore.

The thumbnail round-trip. A platform stores one master and serves several
derivatives, so an image that reaches a moderation model has usually been
through at least one resample it did not ask for. The condition destroys
high-frequency content irreversibly while leaving the image's semantics
intact -- which is why it separates a detector reading synthesis traces
from one reading a resampling signature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from trustfake.corruptions._common import param_tag
from trustfake.corruptions.abc import ImageCorruption

__all__ = ["Downscale"]


class Downscale(ImageCorruption):
    """Bilinearly shrink by `factor`, then restore the original size.

    Named `downscale_<factor>`, e.g. `downscale_2`.

    `antialias=True` on the shrinking leg is load-bearing: without it the
    downsample point-samples the grid and *aliases*, folding high
    frequencies back into the image as a new, spurious pattern instead of
    removing them. That is a different corruption from the one a real
    resizer applies, and on a forensic task it can create the very
    high-frequency structure the detector is looking for. With it, the
    operation matches PIL's `Image.resize(..., BILINEAR)` -- what WP1's
    ladder used (`wp1/src/features.py:79`). The restoring leg needs no
    antialiasing: upsampling invents no frequencies to alias.

    Unlike the codec conditions this stays in float: it is a resampling
    condition, not a file round-trip, so no uint8 grid snap is implied.

    Args:
        factor (float): Shrink divisor, >= 1. 2 means "halve, then double".
        clip_min (float): Minimum valid value for a corrupted input.
        clip_max (float): Maximum valid value for a corrupted input.
    """

    def __init__(
        self, factor: float = 2.0, clip_min: float = 0.0, clip_max: float = 1.0
    ):
        super().__init__(clip_min=clip_min, clip_max=clip_max)
        if float(factor) < 1.0:
            msg = f"Downscale factor must be >= 1, got {factor}."
            raise ValueError(msg)
        self.factor = float(factor)

    @property
    def name(self) -> str:
        return f"downscale_{param_tag(self.factor)}"

    def corrupt(self, images: torch.Tensor) -> torch.Tensor:
        height, width = int(images.shape[-2]), int(images.shape[-1])
        small = (
            max(1, int(round(height / self.factor))),
            max(1, int(round(width / self.factor))),
        )
        shrunk = F.interpolate(
            images.to(torch.float32),
            size=small,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return F.interpolate(
            shrunk,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
