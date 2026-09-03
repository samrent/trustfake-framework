"""Precomputed depth targets for Track C, keyed to the manifest.

The auxiliary depth head trains against maps from a frozen teacher (see
`trustfake.depth.teacher`). Running that teacher inside every training step
would multiply the cost of every arm, so the maps are computed ONCE, offline,
and stored beside the dataset. This module is the store and the writer;
`src/precompute_depth.py` is the command line around it.

**Keyed to the row identity the firewall tracks.** SID-Set's `img_id`
restarts its counter per source split, so the key is the manifest's
``uid = "<source_split>:<img_id>"`` (see `trustfake.data.manifest`), never a
dataset position: the fit split is shuffled by the seeded train/val carve
and filtered by `select`, so positions are not stable across a run's own
setup, let alone across runs. The store keeps one array per SOURCE SHARD in
that shard's parquet row order plus a parallel list of `img_id`s, and the
datamodule resolves uids through it. A fit uid that the store does not
cover is refused at `setup()`, not at the first batch that happens to need it.

**One instrument, recorded.** `manifest.json` in the store's directory
records which teacher produced the maps and under what preprocessing
(`image_size`, `squarecrop`, `input_mode`, teacher input size, output grid,
frame). The datamodule refuses a store whose recorded pre-transform differs
from its own configuration: a target computed on a different view of the
pixels than the model gets is the kind of mismatch that trains and reports
without a symptom.

Layout::

    <depth_dir>/manifest.json
    <depth_dir>/<shard stem>.depth.npy   float16 (N, size, size), memmap-able
    <depth_dir>/<shard stem>.ids.json    ["img_id", ...], parquet row order

float16 at 112x112 is 25 KB per image (2.5 GB per 100k images); the arrays
are memory-mapped, so forked dataloader workers share pages.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from trustfake.data.manifest import DEFAULT_MANIFEST_SEED, build_manifest
from trustfake.logging import get_logger

logger = get_logger("depth-targets")

__all__ = [
    "STORE_MANIFEST",
    "DepthTargetStore",
    "depth_target_paths",
    "read_store_manifest",
    "write_store_manifest",
    "check_store_manifest",
    "precompute_shard",
    "precompute_depth_targets",
]

STORE_MANIFEST = "manifest.json"

#: Which source split each manifest role's rows come from (the uid prefix).
ROLE_SOURCE_SPLIT = {
    "fit": "train",
    "holdout": "train",
    "calib": "validation",
    "test": "validation",
}

#: The pre-transform settings a store is tied to; the datamodule checks them.
CHECKED_KEYS = ("image_size", "squarecrop", "input_mode", "output_size", "frame")


def depth_target_paths(depth_dir: str | Path, shard: str | Path) -> tuple[Path, Path]:
    """(`<stem>.depth.npy`, `<stem>.ids.json`) for a source shard."""
    depth_dir = Path(depth_dir)
    stem = Path(shard).stem
    return depth_dir / f"{stem}.depth.npy", depth_dir / f"{stem}.ids.json"


def read_store_manifest(depth_dir: str | Path) -> dict[str, Any]:
    path = Path(depth_dir) / STORE_MANIFEST
    if not path.exists():
        msg = (
            f"No {STORE_MANIFEST} under {depth_dir}: not a depth-target store. "
            "Run src/precompute_depth.py first."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def write_store_manifest(depth_dir: str | Path, meta: dict[str, Any]) -> Path:
    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    path = depth_dir / STORE_MANIFEST
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def check_store_manifest(meta: dict[str, Any], **expected: Any) -> None:
    """Refuse a store whose recorded settings differ from `expected`.

    Only the keys in `expected` are compared, so a caller states exactly
    what it depends on. Missing keys count as mismatches.
    """
    mismatched = {
        key: (meta.get(key, "<missing>"), value)
        for key, value in expected.items()
        if meta.get(key, "<missing>") != value
    }
    if mismatched:
        detail = ", ".join(
            f"{k}: store has {have!r}, run needs {want!r}"
            for k, (have, want) in mismatched.items()
        )
        msg = (
            f"Depth-target store was precomputed under different settings ({detail}). "
            "The targets would describe a different view of the pixels than the "
            "model sees. Precompute a store for these settings instead."
        )
        logger.error(msg)
        raise ValueError(msg)


class DepthTargetStore:
    """Read side: uid -> float32 (1, S, S) tensor, from memory-mapped shards.

    Args:
        depth_dir: The store directory.
        shards: Source shard paths whose targets this store must serve (the
            manifest's fit shards). A shard with no targets is refused here.
        source_split: uid prefix for these shards ("train" for fit).
    """

    def __init__(
        self, depth_dir: str | Path, shards: Iterable[str | Path], source_split: str
    ):
        self.depth_dir = Path(depth_dir)
        self.source_split = source_split
        self.meta = read_store_manifest(self.depth_dir)
        self._paths: list[Path] = []
        self._index: dict[str, tuple[int, int]] = {}
        self._arrays: dict[int, np.ndarray] = {}
        for shard in shards:
            npy_path, ids_path = depth_target_paths(self.depth_dir, shard)
            if not npy_path.exists() or not ids_path.exists():
                msg = (
                    f"No depth targets for shard {Path(shard).name} under "
                    f"{self.depth_dir} (expected {npy_path.name} and "
                    f"{ids_path.name}). Precompute them, or use a profile whose "
                    "fit shards are covered."
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
            ids = json.loads(ids_path.read_text())
            shard_index = len(self._paths)
            self._paths.append(npy_path)
            for row, img_id in enumerate(ids):
                self._index[f"{source_split}:{img_id}"] = (shard_index, row)
        self.size = int(self.meta.get("output_size", 0))

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, uid: str) -> bool:
        return uid in self._index

    def _array(self, shard_index: int) -> np.ndarray:
        # Opened lazily and per process: a memmap handle does not survive a
        # trip through a dataloader worker's pickle, the paths do.
        array = self._arrays.get(shard_index)
        if array is None:
            array = np.load(self._paths[shard_index], mmap_mode="r")
            self._arrays[shard_index] = array
        return array

    def __getitem__(self, uid: str) -> torch.Tensor:
        try:
            shard_index, row = self._index[uid]
        except KeyError:
            msg = f"No depth target for uid {uid!r} in {self.depth_dir}"
            raise KeyError(msg) from None
        depth = np.asarray(self._array(shard_index)[row], dtype=np.float32)
        return torch.from_numpy(depth).unsqueeze(0)

    def assert_covers(self, uids: Iterable[str]) -> None:
        """Refuse at setup, not mid-epoch, when a row has no target."""
        missing = [uid for uid in uids if uid not in self._index]
        if missing:
            shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            msg = (
                f"{len(missing)} fit rows have no depth target in {self.depth_dir} "
                f"(e.g. {shown}). The store must cover every fit row; precompute "
                "the missing shards."
            )
            logger.error(msg)
            raise ValueError(msg)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state


# --------------------------------------------------------------------------
# Write side
# --------------------------------------------------------------------------


def _to_pil(cell: Any) -> Image.Image:
    """Decode one parquet image cell the way `SIDSetTorchDataset` does."""
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if isinstance(cell, dict) and "bytes" in cell:
        return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    if isinstance(cell, bytes | bytearray):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    return Image.fromarray(np.asarray(cell, dtype=np.uint8)).convert("RGB")


def _pre_transform(image_size: int, squarecrop: bool) -> transforms.Compose:
    """The datamodule's resize-mode view of the pixels, exactly."""
    from trustfake.data.sid_set import CentreSquareCrop

    return transforms.Compose(
        [
            *([CentreSquareCrop()] if squarecrop else []),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def precompute_shard(
    shard: str | Path,
    depth_dir: str | Path,
    teacher: nn.Module,
    *,
    image_size: int = 224,
    squarecrop: bool = False,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    image_column: str = "image",
    id_column: str = "img_id",
) -> int:
    """Write the teacher's maps for every row of one source shard.

    Returns the number of rows written. Reads the shard with pyarrow rather
    than through the HF cache: only the image and id columns are touched
    (the real shards also carry a mask column nobody needs here).
    """
    shard = Path(shard)
    table = pq.read_table(shard, columns=[image_column, id_column])
    ids = [str(v) for v in table.column(id_column).to_pylist()]
    images = table.column(image_column).to_pylist()
    transform = _pre_transform(image_size, squarecrop)
    device = torch.device(device)
    teacher = teacher.to(device).eval()

    maps: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [
                    transform(_to_pil(cell))
                    for cell in images[start : start + batch_size]
                ]
            ).to(device)
            depth = teacher(batch)  # (B, 1, S, S) float32, normalised
            half = depth[:, 0].to(torch.float16)
            if not torch.isfinite(half).all():
                # A non-finite target trains toward nothing and, at
                # evaluation, aborts the selective metrics -- refuse now.
                bad = ids[start : start + batch_size]
                msg = (
                    f"non-finite depth target in shard {shard.name} (rows "
                    f"{start}-{start + len(bad) - 1}, e.g. {bad[:3]}); the "
                    "teacher produced NaN/inf or the map overflowed float16."
                )
                logger.error(msg)
                raise ValueError(msg)
            maps.append(half.cpu().numpy())

    npy_path, ids_path = depth_target_paths(depth_dir, shard)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        np.concatenate(maps, axis=0)
        if maps
        else np.empty((0, teacher.output_size, teacher.output_size), np.float16)
    )
    np.save(npy_path, array)
    ids_path.write_text(json.dumps(ids))
    return len(ids)


def precompute_depth_targets(
    data_dir: str | Path,
    depth_dir: str | Path,
    teacher: nn.Module,
    *,
    profile: str = "train",
    roles: Iterable[str] = ("fit",),
    manifest_seed: int = DEFAULT_MANIFEST_SEED,
    image_size: int = 224,
    squarecrop: bool = False,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    limit_shards: int | None = None,
) -> dict[str, Any]:
    """Precompute targets for every shard the manifest assigns to `roles`.

    Resumable per shard: a shard whose two files already exist is skipped.
    An existing store whose manifest disagrees with these settings is
    refused rather than overwritten. `limit_shards` caps how many shards are
    WRITTEN this call (for timing a first run); skipped shards do not count.
    """
    depth_dir = Path(depth_dir)
    describe = getattr(teacher, "describe", None)
    meta = {
        **(describe() if callable(describe) else {"teacher": type(teacher).__name__}),
        "image_size": int(image_size),
        "squarecrop": bool(squarecrop),
        "input_mode": "resize",
        "dtype": "float16",
        "source": str(data_dir),
    }
    if (depth_dir / STORE_MANIFEST).exists():
        existing = read_store_manifest(depth_dir)
        check_store_manifest(existing, **{k: meta[k] for k in meta})
        logger.info(f"Resuming depth-target store at {depth_dir}")
    else:
        meta["created"] = datetime.now(UTC).isoformat(timespec="seconds")
        write_store_manifest(depth_dir, meta)
        logger.info(f"Created depth-target store at {depth_dir}: {meta}")

    manifest = build_manifest(data_dir, profile, manifest_seed)
    summary: dict[str, Any] = {"written": 0, "skipped": 0, "rows": 0, "shards": []}
    for role in roles:
        if role not in manifest:
            msg = f"Role {role!r} is not in profile {profile!r}: {list(manifest)}"
            logger.error(msg)
            raise ValueError(msg)
        for shard in manifest[role]:
            npy_path, ids_path = depth_target_paths(depth_dir, shard)
            if npy_path.exists() and ids_path.exists():
                summary["skipped"] += 1
                summary["shards"].append((shard.name, "skipped"))
                continue
            if limit_shards is not None and summary["written"] >= limit_shards:
                summary["shards"].append((shard.name, "not reached (limit_shards)"))
                continue
            rows = precompute_shard(
                shard,
                depth_dir,
                teacher,
                image_size=image_size,
                squarecrop=squarecrop,
                batch_size=batch_size,
                device=device,
            )
            logger.info(f"{shard.name}: {rows} depth targets written")
            summary["written"] += 1
            summary["rows"] += rows
            summary["shards"].append((shard.name, "written"))
    return summary
