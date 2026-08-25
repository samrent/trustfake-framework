"""Sensor-side conditions: additive Gaussian noise and Gaussian blur.

The two ImageNet-C staples the WP1 ladder did not carry. They are not
platform processing -- nothing re-encodes an image as "blurred" -- but they
are the standard low-level corruptions a robustness table is read against,
and for a forensic detector they probe the two opposite failure modes:
noise *adds* high-frequency content that can be mistaken for synthesis
residue, blur *removes* the residue a real detector depends on.
"""

from __future__ import annotations

import math

import torch
from torchvision.transforms import functional as TF  # noqa: N812

from trustfake.corruptions._common import param_tag
from trustfake.corruptions.abc import ImageCorruption

__all__ = ["GaussianNoise", "GaussianBlur"]


class GaussianNoise(ImageCorruption):
    """Add i.i.d. N(0, sigma^2) noise to every pixel, then clip.

    Named `gaussian_noise_sigma<sigma>`, with sigma in input units (so
    sigma=0.05 is ~13 grey levels).

    The noise is drawn from an instance-owned CPU generator rather than the
    global RNG, and that is a reproducibility decision, not a style one. A
    condition drawing from the global stream would produce different noise
    depending on how much randomness the rest of the run had already
    consumed -- so re-running an evaluation after adding a single dropout
    layer elsewhere would silently change the reported row. Drawing on CPU
    also makes the row identical on CUDA, MPS and CPU. The generator is
    advanced across calls, so consecutive batches get independent noise
    while the whole pass stays deterministic.

    Args:
        sigma (float): Noise standard deviation, in input units.
        seed (int): Seed for the instance's generator.
        clip_min (float): Minimum valid value for a corrupted input.
        clip_max (float): Maximum valid value for a corrupted input.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        seed: int = 0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(clip_min=clip_min, clip_max=clip_max)
        if float(sigma) < 0.0:
            msg = f"Gaussian noise sigma must be >= 0, got {sigma}."
            raise ValueError(msg)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self._generator = torch.Generator().manual_seed(self.seed)

    @property
    def name(self) -> str:
        return f"gaussian_noise_sigma{param_tag(self.sigma)}"

    def reset(self) -> None:
        """Rewind the noise stream, so a second pass over the same loader
        reproduces the first one batch for batch."""
        self._generator.manual_seed(self.seed)

    def corrupt(self, images: torch.Tensor) -> torch.Tensor:
        base = images.to(device="cpu", dtype=torch.float32)
        noise = torch.randn(base.shape, generator=self._generator, dtype=torch.float32)
        return base + self.sigma * noise


class GaussianBlur(ImageCorruption):
    """Convolve with an isotropic Gaussian kernel of width `sigma`.

    Named `gaussian_blur_sigma<sigma>`, with sigma in pixels of the model
    input grid (224x224 by default) -- not of the original image, which is
    the whole point of applying corruptions post-resize (see the package
    docstring).

    The kernel is truncated at two standard deviations either side
    (`2*ceil(2*sigma)+1` taps, always odd): far enough that the truncation
    error is well under the 1/255 quantisation floor, near enough that the
    condition stays cheap. Torchvision pads by reflection, so the border
    does not darken -- zero padding would add a frame artifact that a
    detector could learn instead of the blur.

    Args:
        sigma (float): Gaussian standard deviation, in input pixels. > 0.
        clip_min (float): Minimum valid value for a corrupted input.
        clip_max (float): Maximum valid value for a corrupted input.
    """

    def __init__(
        self, sigma: float = 1.0, clip_min: float = 0.0, clip_max: float = 1.0
    ):
        super().__init__(clip_min=clip_min, clip_max=clip_max)
        if float(sigma) <= 0.0:
            msg = f"Gaussian blur sigma must be > 0, got {sigma}."
            raise ValueError(msg)
        self.sigma = float(sigma)
        self.kernel_size = 2 * math.ceil(2.0 * self.sigma) + 1

    @property
    def name(self) -> str:
        return f"gaussian_blur_sigma{param_tag(self.sigma)}"

    def corrupt(self, images: torch.Tensor) -> torch.Tensor:
        return TF.gaussian_blur(
            images.to(torch.float32),
            kernel_size=[self.kernel_size, self.kernel_size],
            sigma=[self.sigma, self.sigma],
        )
