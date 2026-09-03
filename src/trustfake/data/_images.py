"""One decoder for a parquet image cell, shared by the datamodule and the
depth-target precompute.

HF `datasets` decodes an Image feature by opening the bytes, loading, and
applying `ImageOps.exif_transpose` when the file carries an EXIF orientation.
The datamodule gets its rows through that path on the real (HF-written)
shards. A precompute that read the same bytes with a bare `Image.open` would
hand an EXIF-rotated photo's target to a differently oriented image -- and
the loss would decrease regardless. So both sides decode through this
function, which mirrors the HF rule.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps

__all__ = ["decode_image_cell"]


def _open(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    if image.getexif().get(Image.ExifTags.Base.Orientation) is not None:
        image = ImageOps.exif_transpose(image)
    return image


def decode_image_cell(cell: Any) -> Image.Image:
    """Any encoding a shard may carry -> a PIL RGB image, EXIF-corrected."""
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if isinstance(cell, dict) and "bytes" in cell and cell["bytes"] is not None:
        return _open(cell["bytes"]).convert("RGB")
    if isinstance(cell, dict) and cell.get("path"):
        with open(cell["path"], "rb") as fh:
            return _open(fh.read()).convert("RGB")
    if isinstance(cell, bytes | bytearray):
        return _open(bytes(cell)).convert("RGB")
    return Image.fromarray(np.asarray(cell, dtype=np.uint8)).convert("RGB")
