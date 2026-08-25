"""Trivial metadata baselines: the floor every detector accuracy is read against.

SID-Set carries a geometry artifact. Fully-synthetic and tampered images are
essentially always square (generated at a fixed resolution); only a small
fraction of real images are. So a rule with no learning in it,

    width == height  ->  fake,

can score well above chance -- on the original SID-Set, above a CLIP probe. A
detector accuracy is therefore not evidence of forensic capability until it is
read against this rule, not against the majority class. (Grommelt et al.,
"Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets".)

The framework resizes every image to a square before the model sees it, so the
model cannot use geometry directly -- but these baselines are computed from the
ORIGINAL image bytes (before resize), so they measure the shortcut that lives
in the data. The honest fix is a geometry-controlled evaluation subset (a WP1
protocol change), not a better model.

These require no model and no GPU: they read image dimensions and format from
the parquet shards the manifest assigns to a split.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from trustfake.data.manifest import DEFAULT_MANIFEST_SEED, build_manifest
from trustfake.logging import get_logger

logger = get_logger("baselines")

__all__ = ["compute_trivial_baselines", "headline"]


def _dims_and_format(image_cell) -> tuple[int, int, str]:
    """(width, height, format) from a parquet image cell (HF Image struct or bytes)."""
    if isinstance(image_cell, dict) and "bytes" in image_cell:
        data = image_cell["bytes"]
    elif isinstance(image_cell, (bytes, bytearray)):
        data = image_cell
    else:
        # Already-decoded array: no original format, infer dims from shape.
        arr = np.asarray(image_cell)
        h, w = arr.shape[0], arr.shape[1]
        return int(w), int(h), ""
    with Image.open(io.BytesIO(data)) as im:
        return int(im.width), int(im.height), (im.format or "")


def compute_trivial_baselines(
    data_dir: str | Path,
    profile: str = "full",
    split_role: str = "test",
    real_class: int = 0,
    manifest_seed: int = DEFAULT_MANIFEST_SEED,
    image_column: str = "image",
    label_column: str = "label",
) -> dict:
    """Compute the metadata baselines over the shards assigned to `split_role`.

    Returns accuracies of: majority class, ``width == height -> fake``,
    ``PNG -> fake``, and their OR, plus per-label square/PNG rates. ``fake`` is
    ``label != real_class``.
    """
    manifest = build_manifest(data_dir, profile, manifest_seed)
    if split_role not in manifest:
        raise ValueError(
            f"Role '{split_role}' not in profile '{profile}': {list(manifest)}"
        )

    square, png, y_binary, labels = [], [], [], []
    for shard in manifest[split_role]:
        table = pq.read_table(shard, columns=[image_column, label_column])
        images = table.column(image_column).to_pylist()
        labs = table.column(label_column).to_pylist()
        for image_cell, lab in zip(images, labs, strict=True):
            w, h, fmt = _dims_and_format(image_cell)
            square.append(int(w == h))
            png.append(int(fmt == "PNG"))
            labels.append(int(lab))
            y_binary.append(int(int(lab) != real_class))

    square = np.asarray(square)
    png = np.asarray(png)
    y = np.asarray(y_binary)
    labels = np.asarray(labels)
    if y.size == 0:
        raise ValueError(f"No rows found for role '{split_role}'.")

    fake_prior = float(y.mean())
    unique_labels = sorted(set(labels.tolist()))
    return {
        "split_role": split_role,
        "n": int(y.size),
        "fake_prior": fake_prior,
        "majority_class": float(max(fake_prior, 1.0 - fake_prior)),
        "square_is_fake": float((square == y).mean()),
        "png_is_fake": float((png == y).mean()),
        "square_or_png_is_fake": float(((square | png) == y).mean()),
        "square_rate_by_label": {
            str(k): float(square[labels == k].mean()) for k in unique_labels
        },
        "png_rate_by_label": {
            str(k): float(png[labels == k].mean()) for k in unique_labels
        },
        "note": (
            "Read every model accuracy against 'square_is_fake', not against "
            "0.5. See trustfake.data.baselines."
        ),
    }


def headline(baselines: dict) -> str:
    return (
        f"TRIVIAL BASELINE ({baselines['split_role']}, n={baselines['n']}): "
        f"'width==height -> fake' = {baselines['square_is_fake']:.4f} accuracy "
        f"(majority class {baselines['majority_class']:.4f}). "
        "Read every model accuracy against that number."
    )
