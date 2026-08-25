"""Common-corruption evaluation conditions.

See `trustfake.corruptions.abc` for what a corruption is, why it is not an
eps-bounded perturbation, and why the ladder is applied to the model input
rather than to the original file bytes.
"""

from .abc import ImageCorruption
from .codec import JPEGCompression, WebPCompression
from .downscale import Downscale
from .gaussian import GaussianBlur, GaussianNoise

__all__ = [
    "ImageCorruption",
    # platform processing (the WP1 ladder)
    "JPEGCompression",
    "WebPCompression",
    "Downscale",
    # sensor-side (ImageNet-C staples)
    "GaussianNoise",
    "GaussianBlur",
]
