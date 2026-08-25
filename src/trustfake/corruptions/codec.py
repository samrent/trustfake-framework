"""Lossy re-encoding conditions: JPEG and WebP.

The two rungs of the ladder that a platform actually applies to everything
it serves. They matter more than usual for a deepfake detector: the
forensic evidence a detector learns lives in exactly the high-frequency
residue a lossy codec is designed to throw away, so a detector that has
learned compression history rather than synthesis traces falls apart here
while a real one degrades gracefully.
"""

from __future__ import annotations

import torch

from trustfake.corruptions._common import codec_roundtrip
from trustfake.corruptions.abc import ImageCorruption

__all__ = ["JPEGCompression", "WebPCompression"]


class JPEGCompression(ImageCorruption):
    """Re-encode the model input as JPEG at a fixed quality.

    Named `jpeg_q<quality>` so the whole ladder can be reported side by
    side without two rungs colliding in a metric prefix or a log directory.

    Quality is not a severity index: the mapping from libjpeg quality to
    distortion is codec-specific and non-linear, and q100 is still lossy
    (chroma subsampling and DCT rounding survive it). Report the quality,
    not a rank.

    Args:
        quality (int): libjpeg quality, 1 (worst) to 100.
        clip_min (float): Minimum valid value for a corrupted input.
        clip_max (float): Maximum valid value for a corrupted input.
    """

    def __init__(self, quality: int = 40, clip_min: float = 0.0, clip_max: float = 1.0):
        super().__init__(clip_min=clip_min, clip_max=clip_max)
        if not 1 <= int(quality) <= 100:
            msg = f"JPEG quality must be in [1, 100], got {quality}."
            raise ValueError(msg)
        self.quality = int(quality)

    @property
    def name(self) -> str:
        return f"jpeg_q{self.quality}"

    def corrupt(self, images: torch.Tensor) -> torch.Tensor:
        return codec_roundtrip(images, "JPEG", quality=self.quality)


class WebPCompression(ImageCorruption):
    """Re-encode the model input as lossy WebP at a fixed quality.

    WebP is what the large platforms actually serve, and its artifacts are
    not JPEG's: it is a block-prediction codec, so it smooths where JPEG
    rings. A detector tuned to JPEG blocking can be blind to it, which is
    the reason both rungs are reported rather than one standing in for the
    other.

    Named `webp_q<quality>`. WP1's ladder pinned quality 80 and left it out
    of the condition string (`wp1/src/features.py:84`); it is in the name
    here because a second quality would otherwise overwrite the first.

    Args:
        quality (int): WebP quality, 1 (worst) to 100.
        clip_min (float): Minimum valid value for a corrupted input.
        clip_max (float): Maximum valid value for a corrupted input.
    """

    def __init__(self, quality: int = 80, clip_min: float = 0.0, clip_max: float = 1.0):
        super().__init__(clip_min=clip_min, clip_max=clip_max)
        if not 1 <= int(quality) <= 100:
            msg = f"WebP quality must be in [1, 100], got {quality}."
            raise ValueError(msg)
        self.quality = int(quality)

    @property
    def name(self) -> str:
        return f"webp_q{self.quality}"

    def corrupt(self, images: torch.Tensor) -> torch.Tensor:
        # lossless=False is explicit: a lossless round-trip is a no-op and
        # would silently report a corruption row identical to clean.
        return codec_roundtrip(images, "WEBP", quality=self.quality, lossless=False)
