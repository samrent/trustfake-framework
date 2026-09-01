"""JPEG quality estimation from quantization tables -- a G1 nuisance feature.

The G1 gate scores a headers-only classifier on (aspect, resolution, JPEG
quality) per environment; CASIA v2 proves the point, where the format split
alone reaches ~0.92 AUC. Quality is not stored in a JPEG file -- what is
stored is the pair of quantization tables, which libjpeg derives from the
quality knob by scaling the IJG reference tables. Inverting that scaling
recovers an estimate of the knob, which is the standard trick (identical in
spirit to ImageMagick's heuristic).

Non-JPEG images return None: "no quality because not JPEG" is itself a
signal, and the G1 classifier receives it as a separate missing-indicator
rather than a fake number.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_jpeg_quality"]

# IJG reference luminance quantization table (quality 50), zigzag-free order.
_IJG_LUMA = np.array(
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ],
    dtype=np.float64,
)


def estimate_jpeg_quality(quantization: dict[int, list[int]] | None) -> int | None:
    """Estimate the libjpeg quality setting from PIL's ``image.quantization``.

    Inverts the IJG scaling: quality >= 50 uses ``scale = 200 - 2q`` and
    below uses ``scale = 5000 / q`` (in percent of the reference table). The
    estimate is the quality whose scaled reference table is nearest (L1) to
    the observed luminance table. Exact for images written by libjpeg-family
    encoders; approximate-but-monotone for custom tables, which is all a
    nuisance feature needs.
    """
    if not quantization or 0 not in quantization:
        return None
    observed = np.asarray(quantization[0], dtype=np.float64)
    if observed.size != 64:
        return None

    qualities = np.arange(1, 101, dtype=np.float64)
    scales = np.where(qualities < 50, 5000.0 / qualities, 200.0 - 2.0 * qualities)
    # (100, 64): reference table under every quality's scale, libjpeg rounding.
    tables = np.clip(np.floor((_IJG_LUMA[None] * scales[:, None] + 50.0) / 100.0), 1, 255)
    distances = np.abs(tables - observed[None]).sum(axis=1)
    return int(qualities[int(np.argmin(distances))])
