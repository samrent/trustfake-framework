"""Shared helpers for common-corruption implementations."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

__all__ = ["param_tag", "codec_roundtrip"]

#: Channel count -> PIL mode. Anything else has no image interpretation and
#: must not be silently reshaped into one.
_PIL_MODES = {1: "L", 3: "RGB"}

#: Codecs with no greyscale profile. A 1-channel batch is promoted to RGB
#: for the encode and reduced back after, so the condition stays defined
#: instead of raising deep inside the encoder.
_RGB_ONLY_FORMATS = frozenset({"WEBP"})


def param_tag(value: float) -> str:
    """Render a numeric parameter for use inside a corruption's `name`.

    The name becomes a metric prefix and a log-directory component, so it
    must survive both: `%g` drops trailing zeros (2.0 -> "2") and the
    decimal point becomes "p" (1.5 -> "1p5") rather than a dot that reads
    as a filename extension.
    """
    return f"{value:g}".replace(".", "p").replace("-", "m")


def codec_roundtrip(
    images: torch.Tensor,
    image_format: str,
    **save_kwargs: object,
) -> torch.Tensor:
    """Encode each image of a batch with a lossy codec and decode it back.

    Quantisation to uint8 is part of the condition, not an implementation
    detail: a codec cannot see more than 8 bits per channel, so skipping
    the rounding would report a condition no real file can produce and
    would hide the grid snap that saving an image always costs.

    The round-trip runs on CPU through PIL because that is where the real
    encoders live; the caller (`ImageCorruption.run`) moves the result back
    to the input's device and dtype.

    Args:
        images (torch.Tensor): Batch of model inputs, shape (B, C, H, W),
            values in [0, 1]. C must be 1 or 3.
        image_format (str): PIL format name, e.g. "JPEG" or "WEBP".
        **save_kwargs: Passed to `PIL.Image.save`, e.g. ``quality=40``.

    Returns:
        torch.Tensor: The decoded batch, shape (B, C, H, W), float32 on
            CPU, on the 1/255 lattice.
    """
    if images.ndim != 4:
        msg = f"Expected a (B, C, H, W) batch, got shape {tuple(images.shape)}."
        raise ValueError(msg)
    channels = int(images.shape[1])
    mode = _PIL_MODES.get(channels)
    if mode is None:
        msg = (
            f"A {image_format} round-trip needs 1 (greyscale) or 3 (RGB) "
            f"channels, got {channels}."
        )
        raise ValueError(msg)
    encode_mode = "RGB" if image_format.upper() in _RGB_ONLY_FORMATS else mode

    quantised = (
        images.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )

    decoded = np.empty_like(quantised)
    for index in range(quantised.shape[0]):
        plane = quantised[index]
        source = Image.fromarray(plane[:, :, 0] if channels == 1 else plane, mode=mode)
        with io.BytesIO() as buffer:
            source.convert(encode_mode).save(buffer, format=image_format, **save_kwargs)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                restored = np.asarray(encoded.convert(mode))
        decoded[index] = restored.reshape(plane.shape[0], plane.shape[1], channels)

    return torch.from_numpy(decoded).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
