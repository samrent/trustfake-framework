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
        stem = shard.name.removesuffix(".parquet")
        columns = zip(
            table["image_name"].to_pylist(),
            table["image_data"].to_pylist(),
            table["model_name"].to_pylist(),
            table["real_source"].to_pylist(),
            table["subset"].to_pylist(),
            table["split"].to_pylist(),
            table["label"].to_pylist(),
            table["architecture"].to_pylist(),
            strict=True,
        )
        for position, (name, image, model, real_source, subset, split, label, arch) in (
            enumerate(columns)
        ):
            fake = int(label) == 1
            rows.append(
                {
                    # image_name repeats across generators and shards -- the
                    # shard stem + parquet row is the stable, unique key.
                    "uid": f"community_forensics:{split}:{stem}:{position}",
                    "image_name": name,
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


def _chunks(rows: Rows, prefix: str, size: int = 2000) -> Iterator[tuple[str, Rows]]:
    """Synthetic shards for file-tree datasets, so embed resume still works."""
    for start in range(0, len(rows), size):
        yield f"{prefix}-{start // size:04d}", rows[start : start + size]


#: The L4 held-out tool inside AUDITS: present ONLY in its test split (the
#: dataset's own OOD design), so it has zero training presence by
#: construction. Frozen here at ingestion, per the spec.
AUDITS_L4_METHOD = "PowerPaint"


def _iter_audits(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """AUDITS (DivyaApp/AUDITS): metadata parquet + extracted train/val/test.

    Pool candidates: the dataset's own train+val splits (4 methods +
    Authentic). Leg L4: test-split Authentic + the held-out method -- the
    other test-only methods are not embedded (nothing in the ladder reads
    them). `generator` carries manipulation_type; Authentic rows are reals.
    """
    table = pq.read_table(
        data_dir / "data" / "train-00000-of-00001.parquet"
    ).to_pandas()
    wanted = table[
        table["training"].isin(["train", "val"])
        | (
            (table["training"] == "test")
            & table["manipulation_type"].isin(["Authentic", AUDITS_L4_METHOD])
        )
    ]
    rows: Rows = []
    for record in wanted.itertuples(index=False):
        authentic = record.manipulation_type == "Authentic"
        # Zip layouts differ per split (verified against the zips 2026-09-01):
        #   train/val: {split}/{split}_{subset}/{method}/{manipulated|original}/{id}.jpg
        #   test:      test/test_{subset}/{method}/{id}.jpg, with a SHARED
        #              original/ dir -- test Authentic rows are the pristine
        #              twins of the manipulated test rows (same ids), which is
        #              also why the uid must carry the method.
        prefix = f"{record.training}/{record.training}_{record.subset.lower()}"
        # COCO-subset files keep COCO's 12-digit zero-padded names in every
        # split; NEWS files use the bare id (verified on disk 2026-09-01).
        stem = f"{int(record.id):012d}" if record.subset == "COCO" else str(record.id)
        if record.training == "test":
            folder = "original" if authentic else record.manipulation_type
            relative = f"{prefix}/{folder}/{stem}.jpg"
        else:
            kind = "original" if authentic else "manipulated"
            relative = f"{prefix}/{record.manipulation_type}/{kind}/{stem}.jpg"
        path = data_dir / relative
        # NEWS and COCO id spaces overlap numerically, so the subset is part
        # of the key (2,402 collisions without it, measured on the cache).
        uid = (
            f"audits:{record.training}:{record.subset.lower()}:"
            f"{record.manipulation_type}:{record.id}"
        )
        rows.append(
            {
                "uid": uid,
                "path": str(path),
                "label3": 0 if authentic else 2,
                "label_bin": int(not authentic),
                "generator": None if authentic else record.manipulation_type,
                "source_split": record.training,
                "subset": record.subset,
                "distribution": record.distribution,
            }
        )
    rows.sort(key=lambda r: r["uid"])
    missing = [r for r in rows if not Path(r["path"]).exists()]
    if missing:
        msg = (
            f"audits: {len(missing)} of {len(rows)} files missing "
            f"(e.g. {missing[0]['path']}) -- extract the zips first"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
    yield from _chunks(rows, "audits")


def _iter_synthbuster(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """Synthbuster: nine per-generator folders of 1,000 PNGs, all synthetic.

    No paired reals in the archive (RAISE-1k is a separate, form-gated
    download); a single-class environment, which C2 passes through.
    """
    rows: Rows = []
    for image in sorted(data_dir.rglob("*")):
        suffixes = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        if image.suffix.lower() not in suffixes:
            continue
        generator = image.parent.name
        rows.append(
            {
                "uid": f"synthbuster:all:{generator}/{image.name}",
                "path": str(image),
                "label3": 1,
                "label_bin": 1,
                "generator": generator,
                "source_split": "all",
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"synthbuster: no images under {data_dir} -- extract first"
        )
    yield from _chunks(rows, "synthbuster")


def _iter_vision(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """VISION: per-device folders, images only; every row is camera real.

    `subset` records the collection (flat / nat / natFBH / natFBL / natWA),
    which is the native-vs-web-recompressed axis the spec wants covered;
    the device id rides in `device`.
    """
    rows: Rows = []
    for image in sorted(data_dir.rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "/images/" not in str(image):
            continue
        collection = image.parent.name
        parts = image.parts
        device = parts[parts.index("images") - 1] if "images" in parts else "unknown"
        rows.append(
            {
                "uid": f"vision:all:{device}/{collection}/{image.name}",
                "path": str(image),
                "label3": 0,
                "label_bin": 0,
                "generator": None,
                "source_split": "all",
                "subset": collection,
                "device": device,
            }
        )
    if not rows:
        raise FileNotFoundError(f"vision: no images under {data_dir}")
    yield from _chunks(rows, "vision")


def _iter_tgif(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """TGIF/TGIF2 from `extracted/{tool}/{split}/{category}/{file}.png`.

    `orig` is the real environment; every other tool is tampered with
    `generator` = tool name. `tool` rides as its own column because the
    pool filter excludes ps-sp (frozen L4) by it. Masks are never
    extracted into this tree.
    """
    extracted = data_dir / "extracted"
    rows: Rows = []
    for image in sorted(extracted.rglob("*.png")):
        relative = image.relative_to(extracted)
        # Forgery FILENAMES embed the mask name that produced them
        # (`..._mask_segm.png_ps_0.png` IS a forged image); mask archives
        # were never fetched, so no name filter -- the first embed filtered
        # on 'mask' and silently kept only the originals.
        if len(relative.parts) < 4:
            continue
        tool, split, category = relative.parts[0], relative.parts[1], relative.parts[2]
        real = tool == "orig"
        rows.append(
            {
                "uid": f"tgif:{split}:{tool}/{category}/{image.name}",
                "path": str(image),
                "label3": 0 if real else 2,
                "label_bin": int(not real),
                "generator": None if real else tool,
                "source_split": split,
                "tool": tool,
                "category": category,
            }
        )
    if not rows:
        raise FileNotFoundError(f"tgif: nothing under {extracted} -- extract first")
    yield from _chunks(rows, "tgif")


def _iter_imd2020(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """IMD2020 from three extracted subtrees.

    real_life/<id>/: `*_orig.jpg` real + human-made manipulations (the
    irreplaceable in-the-wild env); camera_real/<brand>/<model>/: reals;
    gan_inpaint/: pre-diffusion GAN inpaintings (tampered). Masks skipped.
    """
    rows: Rows = []
    image_suffixes = {".jpg", ".jpeg", ".png"}
    for image in sorted((data_dir / "real_life").rglob("*")):
        if image.suffix.lower() not in image_suffixes or "mask" in image.name:
            continue
        real = image.stem.endswith("_orig")
        rows.append(
            {
                "uid": f"imd2020:real_life:{image.parent.name}/{image.name}",
                "path": str(image),
                "label3": 0 if real else 2,
                "label_bin": int(not real),
                "generator": None if real else "human_manual",
                "source_split": "real_life",
            }
        )
    for image in sorted((data_dir / "camera_real").rglob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rows.append(
            {
                "uid": (
                    f"imd2020:camera_real:{image.parent.parent.name}/"
                    f"{image.parent.name}/{image.name}"
                ),
                "path": str(image),
                "label3": 0,
                "label_bin": 0,
                "generator": None,
                "source_split": "camera_real",
            }
        )
    gan_root = data_dir / "gan_inpaint"
    if gan_root.exists():
        for image in sorted(gan_root.rglob("*")):
            if image.suffix.lower() not in image_suffixes or "mask" in image.name:
                continue
            rows.append(
                {
                    "uid": f"imd2020:gan_inpaint:{image.parent.name}/{image.name}",
                    "path": str(image),
                    "label3": 2,
                    "label_bin": 1,
                    "generator": "yu2018_gan_inpainting",
                    "source_split": "gan_inpaint",
                }
            )
    if not rows:
        raise FileNotFoundError(f"imd2020: nothing under {data_dir} -- extract first")
    yield from _chunks(rows, "imd2020")


def _iter_sagi_d(data_dir: Path) -> Iterator[tuple[str, Rows]]:
    """SAGI-D (giakop/sagi-d): metadata-CSV-driven, modern inpainting tools.

    Fakes carry `generator` = inpainting_model (comma-joined for the
    sequential multi-tool edits); the unique originals referenced by
    src_path are the in-env reals (COCO / RAISE / OpenImages sources --
    the RAISE lineage is why G2 must see this dataset before any RAISE
    rows join a leg). PowerPaint rows are NOT filtered here: the reader
    reports what ships, the pool filter enforces the frozen L4 rule.
    """
    import pandas as pd

    frame = pd.read_csv(data_dir / "sagid.csv")

    def resolve(raw: str) -> Path:
        relative = raw.replace("\\", "/").removeprefix("sagid/")
        direct = data_dir / relative
        return direct if direct.exists() else data_dir / "sagid" / relative

    rows: Rows = []
    for record in frame.itertuples(index=False):
        relative = record.img_path.replace("\\", "/").removeprefix("sagid/")
        source = relative.split("/")[1] if len(relative.split("/")) > 1 else "unknown"
        rows.append(
            {
                "uid": f"sagi_d:{record.split}:{relative}",
                "path": str(resolve(record.img_path)),
                "label3": 2,
                "label_bin": 1,
                "generator": record.inpainting_model,
                "source_split": record.split,
                "real_source": source,
                "diffusion_model": record.diffusion_model,
            }
        )
    originals = frame.drop_duplicates("src_path")
    for record in originals.itertuples(index=False):
        relative = record.src_path.replace("\\", "/").removeprefix("sagid/")
        source = relative.split("/")[1] if len(relative.split("/")) > 1 else "unknown"
        rows.append(
            {
                "uid": f"sagi_d:{record.split}:{relative}",
                "path": str(resolve(record.src_path)),
                "label3": 0,
                "label_bin": 0,
                "generator": None,
                "source_split": record.split,
                "real_source": source,
                "diffusion_model": None,
            }
        )
    rows.sort(key=lambda r: r["uid"])
    missing = sum(1 for r in rows if not Path(r["path"]).exists())
    if missing > len(rows) * 0.01:
        msg = f"sagi_d: {missing} of {len(rows)} files missing -- extract first"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if missing:
        logger.warning(f"sagi_d: {missing} files missing, skipped")
        rows = [r for r in rows if Path(r["path"]).exists()]
    yield from _chunks(rows, "sagi_d")


READERS = {
    "sid_set": _iter_sid_set,
    "so_fake_ood": _iter_so_fake_ood,
    "community_forensics_small": _iter_community_forensics_small,
    "audits": _iter_audits,
    "synthbuster": _iter_synthbuster,
    "vision": _iter_vision,
    "tgif": _iter_tgif,
    "imd2020": _iter_imd2020,
    "sagi_d": _iter_sagi_d,
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
