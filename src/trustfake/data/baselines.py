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

from trustfake.data.manifest import (
    DEFAULT_MANIFEST_SEED,
    build_manifest,
    geometry_selection,
)
from trustfake.logging import get_logger

logger = get_logger("baselines")

__all__ = ["compute_trivial_baselines", "headline", "image_dims_and_format"]

#: Protocol tags under which geometry carries no label information, so the
#: raw ``width == height`` accuracy would misstate what was measured. These
#: are the `SIDSetDataModule` controls: the row filters
#: (`trustfake.data.manifest.GEOMETRY_FILTERS`) and the squarecrop
#: pre-transform.
#: Row filters. These change WHICH rows are reported, and therefore the
#: class prior and every floor computed from it -- so a headline naming
#: one is only valid against baselines computed under the same filter.
_ROW_FILTER_TAGS = frozenset({"square", "nonsquare", "matched"})
#: `squarecrop` is a pre-transform, not a row filter: the reported rows
#: are unchanged, so the raw split's floor is still the right one.
_GEOMETRY_CONTROLLED_TAGS = _ROW_FILTER_TAGS | {"squarecrop"}

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
    geometry_filter: str = "none",
) -> dict:
    """Compute the metadata baselines over the shards assigned to `split_role`.

    Returns accuracies of: majority class, ``width == height -> fake``,
    ``PNG -> fake``, their OR, and ``short side == 1024 -> fake``, plus
    per-label square/PNG/short-side rates. ``fake`` is ``label != real_class``.

    `geometry_filter` MUST match the `SIDSetDataModule` row filter the model
    was evaluated under. A row filter changes the class prior, so it changes
    every one of these numbers -- the floor for a `nonsquare` subset of
    SID-Set is 1.0000 (that subset is almost all real), not the raw split's
    0.6657. Computing the floor on the raw split and printing it beside a
    filtered accuracy is how a model that beat nothing comes to look like it
    cleared a bar. When a filter is set, the returned numbers describe the
    FILTERED subset and `raw` carries the unfiltered ones for contrast.
    """
    manifest = build_manifest(data_dir, profile, manifest_seed)
    if split_role not in manifest:
        raise ValueError(
            f"Role '{split_role}' not in profile '{profile}': {list(manifest)}"
        )

    square, png, short_side, y_binary, labels = [], [], [], [], []
    widths, heights = [], []
    for shard in manifest[split_role]:
        table = pq.read_table(shard, columns=[image_column, label_column])
        images = table.column(image_column).to_pylist()
        labs = table.column(label_column).to_pylist()
        for image_cell, lab in zip(images, labs, strict=True):
            w, h, fmt = image_dims_and_format(image_cell)
            square.append(int(w == h))
            png.append(int(fmt == "PNG"))
            short_side.append(min(w, h))
            widths.append(w)
            heights.append(h)
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

    def _summarise(idx: np.ndarray) -> dict:
        sq, pg, ss = square[idx], png[idx], shortside_1024[idx]
        yy, lab = y[idx], labels[idx]
        prior = float(yy.mean())
        present = sorted(set(lab.tolist()))
        return {
            "n": int(yy.size),
            "fake_prior": prior,
            "majority_class": float(max(prior, 1.0 - prior)),
            "square_is_fake": float((sq == yy).mean()),
            "png_is_fake": float((pg == yy).mean()),
            "square_or_png_is_fake": float(((sq | pg) == yy).mean()),
            "shortside1024_is_fake": float((ss == yy).mean()),
            "shortside1024_rate_by_label": {
                str(k): float(ss[lab == k].mean()) for k in present
            },
            "square_rate_by_label": {
                str(k): float(sq[lab == k].mean()) for k in present
            },
            "png_rate_by_label": {str(k): float(pg[lab == k].mean()) for k in present},
        }

    all_rows = np.arange(y.size)
    raw = _summarise(all_rows)
    if geometry_filter != "none":
        selected = np.asarray(
            geometry_selection(
                np.asarray(widths),
                np.asarray(heights),
                labels,
                mode=geometry_filter,
                seed=manifest_seed,
            ),
            dtype=int,
        )
        if selected.size == 0:
            raise ValueError(
                f"geometry_filter={geometry_filter!r} selected no rows from "
                f"'{split_role}'; there is no subset to read a floor against."
            )
        summary = _summarise(selected)
    else:
        summary = raw

    return {
        "split_role": split_role,
        "geometry_filter": geometry_filter,
        # The unfiltered split, kept so a controlled headline can quote the
        # artifact it removed without recomputing it.
        "raw": raw,
        **summary,
        # The decode-scale residue lives in `shortside1024_is_fake`. Every
        # SID-Set fake is generated at 1024x1024 and only a few percent of
        # reals share that short side. Unlike width==height, a centre crop to
        # the short side does NOT remove it -- the crop preserves the short
        # side exactly -- so it survives into resampling history, where a CNN
        # can still read it. This is why the squarecrop control cannot
        # exonerate a model of every geometry shortcut, only the width==height
        # one.
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
        applied = baselines.get("geometry_filter", "none")
        row_filter = condition.lower() in _ROW_FILTER_TAGS
        if row_filter and applied != condition.lower():
            # Refuse rather than print a plausible wrong number. A row filter
            # changes the class prior, so every floor here is a different
            # number on the filtered subset -- quoting the raw split's floor
            # beside a filtered accuracy is exactly the misreading this
            # function exists to prevent.
            raise ValueError(
                f"headline(condition={condition!r}) needs baselines computed "
                f"under that same row filter, but they were computed with "
                f"geometry_filter={applied!r}. Re-run "
                f"compute_trivial_baselines(..., geometry_filter={condition!r})."
            )
        raw = baselines.get("raw", baselines)
        if row_filter:
            # The rows themselves changed, so every floor was recomputed on
            # the subset; quote those.
            return (
                f"GEOMETRY-CONTROLLED protocol ('{condition}'): geometry "
                "carries no label information under it. On the controlled "
                f"subset (n={baselines['n']}) 'width==height -> fake' scores "
                f"{baselines['square_is_fake']:.4f} and the majority class "
                f"{baselines['majority_class']:.4f} -- read the model against "
                f"the HIGHER of those, not against 0.5. On the RAW "
                f"{baselines['split_role']} split (n={raw['n']}) the same rule "
                f"scores {raw['square_is_fake']:.4f}, which is the artifact "
                f"this protocol removes. {residue}"
            )
        # squarecrop: the rows are unchanged, but every image is square by the
        # time the model sees it, so the rule degenerates to the constant
        # "fake" and scores the fake prior. The floor a model must clear is
        # the best constant rule, i.e. the majority class.
        return (
            f"GEOMETRY-CONTROLLED protocol ('{condition}'): every image is "
            "square once cropped, so 'width==height -> fake' degenerates to a "
            f"constant and scores the fake prior ({baselines['fake_prior']:.4f}). "
            f"Read the model against the majority class "
            f"({baselines['majority_class']:.4f}). On the RAW "
            f"{baselines['split_role']} split (n={baselines['n']}) the rule "
            f"scores {baselines['square_is_fake']:.4f} -- the artifact this "
            f"protocol removes. {residue}"
        )
    return (
        f"TRIVIAL BASELINE ({baselines['split_role']}, n={baselines['n']}): "
        f"'width==height -> fake' = {baselines['square_is_fake']:.4f} accuracy "
        f"(majority class {baselines['majority_class']:.4f}). "
        f"Read every model accuracy against that number. {residue}"
    )
