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
protocol change), not a better model. The framework ships two such controls --
`SIDSetDataModule(geometry_filter=...)` at row level and
`SIDSetDataModule(squarecrop=True)` at pixel level -- and `headline` is
protocol-aware so a controlled row is never printed next to the uncontrolled
0.98.

One baseline here does NOT yield to either control, and it is the reason the
controls must not be oversold: ``short side == 1024 -> fake``. Every fake in
SID-Set is generated at 1024x1024 and only a few percent of reals share that
short side, so the rule scores well above chance -- and a centre crop to the
short side preserves the short side exactly. The residue survives into
resampling history, which a CNN can read. A geometry-controlled row is
evidence about the ``width == height`` shortcut and about nothing else.

These require no model and no GPU: they read image dimensions and format from
the parquet shards the manifest assigns to a split.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from trustfake.data.manifest import DEFAULT_MANIFEST_SEED, build_manifest
from trustfake.logging import get_logger

logger = get_logger("baselines")

__all__ = ["compute_trivial_baselines", "headline", "image_dims_and_format"]

#: Protocol tags under which geometry carries no label information, so the
#: raw ``width == height`` accuracy would misstate what was measured. These
#: are the `SIDSetDataModule` controls: the row filters
#: (`trustfake.data.manifest.GEOMETRY_FILTERS`) and the squarecrop
#: pre-transform.
_GEOMETRY_CONTROLLED_TAGS = frozenset({"squarecrop", "square", "nonsquare", "matched"})

#: The resolution SID-Set's generators emit at. A centre crop preserves the
#: short side, so this residue outlives the geometry controls.
GENERATION_SHORT_SIDE = 1024


def image_dims_and_format(image_cell) -> tuple[int, int, str]:
    """(width, height, format) of an image cell, without decoding its pixels.

    Accepts every shape a SID-Set image column takes: the HF Image struct
    ``{bytes, path}``, raw encoded bytes, a decoded PIL image, or an array.
    For the encoded forms `Image.open` reads the header only, which is what
    makes it cheap enough for `SIDSetDataModule` to call once per row when a
    geometry filter is on. A decoded cell has no original format, so the
    format comes back empty rather than guessed.
    """
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
    ``PNG -> fake``, their OR, and ``short side == 1024 -> fake``, plus
    per-label square/PNG/short-side rates. ``fake`` is ``label != real_class``.
    """
    manifest = build_manifest(data_dir, profile, manifest_seed)
    if split_role not in manifest:
        raise ValueError(
            f"Role '{split_role}' not in profile '{profile}': {list(manifest)}"
        )

    square, png, short_side, y_binary, labels = [], [], [], [], []
    for shard in manifest[split_role]:
        table = pq.read_table(shard, columns=[image_column, label_column])
        images = table.column(image_column).to_pylist()
        labs = table.column(label_column).to_pylist()
        for image_cell, lab in zip(images, labs, strict=True):
            w, h, fmt = image_dims_and_format(image_cell)
            square.append(int(w == h))
            png.append(int(fmt == "PNG"))
            short_side.append(min(w, h))
            labels.append(int(lab))
            y_binary.append(int(int(lab) != real_class))

    square = np.asarray(square)
    png = np.asarray(png)
    short_side = np.asarray(short_side)
    y = np.asarray(y_binary)
    labels = np.asarray(labels)
    if y.size == 0:
        raise ValueError(f"No rows found for role '{split_role}'.")
    shortside_1024 = (short_side == GENERATION_SHORT_SIDE).astype(int)

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
        # The decode-scale residue. Every SID-Set fake is generated at
        # 1024x1024 and only a few percent of reals share that short side.
        # Unlike width==height, a centre crop to the short side does NOT
        # remove it -- the crop preserves the short side exactly -- so it
        # survives into resampling history, where a CNN can still read it.
        # This is why the squarecrop control cannot exonerate a model of
        # every geometry shortcut, only of the width==height one.
        "shortside1024_is_fake": float((shortside_1024 == y).mean()),
        "shortside1024_rate_by_label": {
            str(k): float(shortside_1024[labels == k].mean()) for k in unique_labels
        },
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


def _is_geometry_controlled(condition: str) -> bool:
    """True when `condition` names a protocol under which geometry carries
    no label information -- a squarecrop pre-transform or a geometry row
    filter. Tokenised rather than substring-matched, so 'nonsquare' is not
    read as 'square' and a compound tag like 'squarecrop+matched' still
    resolves."""
    tokens = set(re.split(r"[^a-z0-9]+", condition.lower()))
    return bool(tokens & _GEOMETRY_CONTROLLED_TAGS)


def headline(baselines: dict, condition: str | None = None) -> str:
    """One line to print above any accuracy computed on this split.

    Protocol-aware, and that is the point. Under a geometry control --
    `squarecrop`, or a `square` / `nonsquare` / `matched` row filter -- the
    ``width == height`` artifact is removed BY CONSTRUCTION: the rule is
    constant (or independent of the label), so its accuracy on the
    controlled subset is the majority class, not 0.98. Printing the raw
    0.98 beside a controlled row would misstate the protocol, and in the
    direction that flatters nobody: it makes an honest, controlled result
    look like a failure to clear a bar it was never being read against.

    Args:
        baselines: The dict from `compute_trivial_baselines`.
        condition: The evaluation protocol this line will sit next to,
            e.g. "squarecrop" or a `geometry_filter` mode. None means the
            raw, uncontrolled protocol.

    Returns:
        A single line naming the floor the accuracy must be read against.
    """
    residue = (
        "The decode-scale residue 'short side == 1024 -> fake' scores "
        f"{baselines['shortside1024_is_fake']:.4f} and survives centre-cropping."
    )
    if condition and _is_geometry_controlled(condition):
        return (
            f"GEOMETRY-CONTROLLED protocol ('{condition}'): geometry carries no "
            "label information under it, so 'width==height -> fake' scores the "
            f"majority class ({baselines['majority_class']:.4f}). On the RAW "
            f"{baselines['split_role']} split (n={baselines['n']}) the same rule "
            f"scores {baselines['square_is_fake']:.4f} -- this row is not riding "
            f"that artifact. {residue}"
        )
    return (
        f"TRIVIAL BASELINE ({baselines['split_role']}, n={baselines['n']}): "
        f"'width==height -> fake' = {baselines['square_is_fake']:.4f} accuracy "
        f"(majority class {baselines['majority_class']:.4f}). "
        f"Read every model accuracy against that number. {residue}"
    )
