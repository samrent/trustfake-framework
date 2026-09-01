"""Per-dataset shard readers for the TB-E3 embed-once pipeline.

A reader yields ``(shard_name, rows)`` per source shard, where each row is a
dict with at least:

    uid          "<dataset>:<source_split>:<id>" -- the project row key; the
                 source_split prefix is load-bearing (SID-Set img_ids repeat
                 across splits with different bytes, see trustfake.data.manifest).
    image        raw encoded bytes (decoded in the embed workers)
    label3       0 real / 1 fully-synthetic / 2 tampered / -1 unmappable
                 (a binary fake whose modality the source does not state)
    label_bin    0 real / 1 fake -- always defined
    generator    per-image generator/manipulation method, or None
    source_split the source dataset's own split name

plus any columns worth carrying (dataset-declared width/height and the like).
Actual width/height/format/JPEG-quality are re-measured from the bytes in the
embed workers -- the bytes are the ground truth, dataset metadata is not.

Environment tag == dataset key: the spec's (class x environment) cells are
per-dataset, so the reader name is the environment name.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trustfake.logging import get_logger

logger = get_logger("curation.readers")

__all__ = ["READERS", "iter_dataset"]

Rows = list[dict[str, Any]]


def _image_bytes(cell: Any) -> bytes:
    """Unwrap an HF Image struct ({bytes, path}) or raw bytes."""
    if isinstance(cell, dict):
        return cell["bytes"]
    return cell


def _iter_sid_set(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """SID-Set: train-*/validation-* parquet, columns img_id/image/label.

    Every shard is embedded (train AND validation); the fit/calib/test role
    of a row is re-derived downstream from `trustfake.data.manifest` so the
    cache stays role-agnostic and the TB-E2 protocol stays reproducible.
    """
    shards = sorted(data_dir.rglob("train-*.parquet")) + sorted(
        data_dir.rglob("validation-*.parquet")
    )
    for shard in shards:
        split = shard.name.split("-")[0]
        table = pq.read_table(shard, columns=["img_id", "image", "label"])
        rows: Rows = []
        for img_id, image, label in zip(
            table["img_id"].to_pylist(),
            table["image"].to_pylist(),
            table["label"].to_pylist(),
            strict=True,
        ):
            label3 = int(label)
            rows.append(
                {
                    "uid": f"sid_set:{split}:{img_id}",
                    "image": _image_bytes(image),
                    "label3": label3,
                    "label_bin": int(label3 > 0),
                    "generator": None,
                    "source_split": split,
                }
            )
        yield shard.name, rows


def _iter_so_fake_ood(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """So-Fake-OOD: test_image-* parquet. Eval-only (leg L2) by policy.

    Ships string labels; the mapping is SID-Set's ids, imported from the
    datamodule so the two can never disagree. `generator` names the synthesis
    model, `editor` the manipulation tool; whichever applies is the per-image
    method label.
    """
    from trustfake.data.so_fake_ood import STRING_LABEL_TO_ID

    for shard in sorted(data_dir.rglob("test_image-*.parquet")):
        table = pq.read_table(
            shard, columns=["id", "label", "generator", "editor", "image"]
        )
        rows: Rows = []
        for rid, label, generator, editor, image in zip(
            table["id"].to_pylist(),
            table["label"].to_pylist(),
            table["generator"].to_pylist(),
            table["editor"].to_pylist(),
            table["image"].to_pylist(),
            strict=True,
        ):
            label3 = STRING_LABEL_TO_ID[label]
            rows.append(
                {
                    "uid": f"so_fake_ood:test:{rid}",
                    "image": _image_bytes(image),
                    "label3": label3,
                    "label_bin": int(label3 > 0),
                    "generator": generator or editor or None,
                    "source_split": "test",
                }
            )
        yield shard.name, rows


def _iter_community_forensics_small(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """Community Forensics (Small): data/*.parquet, per-image model_name.

    Fakes are fully generated (label3=1); reals carry their source dataset
    (FFHQ / VISION / COCO / LHQ) in `real_source`, which G2 needs -- those
    sources overlap AUDITS (COCO) and a native VISION ingest.
    """
    for shard in sorted((data_dir / "data").glob("*.parquet")):
        table = pq.read_table(
            shard,
            columns=[
                "image_name",
                "image_data",
                "model_name",
                "real_source",
                "subset",
                "split",
                "label",
                "architecture",
            ],
        )
        rows: Rows = []
        for name, image, model, real_source, subset, split, label, arch in zip(
            table["image_name"].to_pylist(),
            table["image_data"].to_pylist(),
            table["model_name"].to_pylist(),
            table["real_source"].to_pylist(),
            table["subset"].to_pylist(),
            table["split"].to_pylist(),
            table["label"].to_pylist(),
            table["architecture"].to_pylist(),
            strict=True,
        ):
            fake = int(label) == 1
            rows.append(
                {
                    "uid": f"community_forensics:{split}:{subset}:{name}",
                    "image": _image_bytes(image),
                    "label3": 1 if fake else 0,
                    "label_bin": int(fake),
                    "generator": model if fake else None,
                    "source_split": split,
                    "subset": subset,
                    "architecture": arch,
                    "real_source": real_source,
                }
            )
        yield shard.name, rows


READERS = {
    "sid_set": _iter_sid_set,
    "so_fake_ood": _iter_so_fake_ood,
    "community_forensics_small": _iter_community_forensics_small,
}


def iter_dataset(dataset: str, data_dir: str | Path) -> Iterator[tuple[str, Rows]]:
    if dataset not in READERS:
        msg = f"No reader for '{dataset}'. Available: {sorted(READERS)}"
        logger.error(msg)
        raise ValueError(msg)
    data_dir = Path(data_dir)
    if not data_dir.exists():
        msg = f"Data directory does not exist: {data_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    return READERS[dataset](data_dir)
