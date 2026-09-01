"""Embed-once feature cache for TB-E3 (spec instrument, held fixed).

Encoder: standard ViT-B/16 (open_clip, laion2B-s34B-b88K), frozen. The
preprocessing mirrors `SIDSetDataModule` in resize mode exactly -- PIL RGB,
bilinear resize to 224x224, ToTensor -- and CLIP's own mean/std is applied
here (it lives with the model, never the datamodule; see
gotchas/clip-normalization-lives-in-the-model.md). Features are the visual
tower's output, L2-normalized (the probe regime: `normalize_features=True`
in `CLIPProbeClassifier`), stored fp16.

Cache layout, per dataset:

    <out>/<dataset>/features-<shard>.npy    (N, 512) fp16, L2-normalized
    <out>/<dataset>/index-<shard>.parquet   row-aligned metadata
    <out>/<dataset>/manifest.json           preprocessing policy + per-shard
                                            provenance (source size/mtime,
                                            row and failure counts)

Resumable at shard level: a shard whose .npy and index rows agree is
skipped. Rows whose bytes fail to decode are dropped from BOTH files and
counted in the manifest -- the index is the ledger of what was embedded,
not of what the source claimed to hold.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from trustfake.curation.jpegq import estimate_jpeg_quality
from trustfake.curation.readers import iter_dataset
from trustfake.logging import get_logger
from trustfake.models.torch.clip import CLIP_MEAN, CLIP_STD

logger = get_logger("curation.embed")

__all__ = ["PREPROCESSING_POLICY", "embed_dataset"]

MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
IMAGE_SIZE = 224
FEATURE_DIM = 512

#: The one preprocessing policy, recorded in every manifest (spec: the route
#: to 224 is itself a per-dataset signature). Changing ANY step changes the
#: cache identity -- the hash below travels with the features.
PREPROCESSING_POLICY = (
    "PIL.convert(RGB) -> torchvision.Resize((224,224), bilinear on PIL) -> "
    "ToTensor -> normalize(CLIP mean/std, in-model convention) -> "
    f"open_clip {MODEL_NAME}/{PRETRAINED} .visual -> float32 -> "
    "L2-normalize -> fp16"
)
POLICY_HASH = hashlib.sha256(PREPROCESSING_POLICY.encode()).hexdigest()[:16]

# Written by dataloader workers into each meta dict; promoted to index columns.
_MEASURED = ("width", "height", "format", "jpeg_q")


class _RowsDataset(Dataset):
    """Map-style view over one shard's rows; decode happens in workers."""

    def __init__(self, rows: list[dict[str, Any]], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            source = (
                io.BytesIO(row["image"])
                if row.get("image") is not None
                else row["path"]
            )
            with Image.open(source) as pil:
                measured = {
                    "width": int(pil.width),
                    "height": int(pil.height),
                    "format": pil.format,
                    "jpeg_q": estimate_jpeg_quality(getattr(pil, "quantization", None)),
                }
                tensor = self.transform(pil.convert("RGB"))
        except Exception:  # noqa: BLE001 -- a corrupt row must not kill the shard
            return index, None, None
        return index, tensor, measured


def _collate(batch):
    kept = [(i, t, m) for i, t, m in batch if t is not None]
    failed = [i for i, t, _ in batch if t is None]
    if not kept:
        return failed, None, [], []
    indices = [i for i, _, _ in kept]
    tensors = torch.stack([t for _, t, _ in kept])
    metas = [m for _, _, m in kept]
    return failed, tensors, indices, metas


def _load_encoder(device: torch.device):
    import open_clip
    from torchvision import transforms

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    visual = model.visual.to(device).eval()
    visual.requires_grad_(False)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, -1, 1, 1)
    return visual, transform, mean, std


def _index_table(
    rows: list[dict[str, Any]],
    metas: dict[int, dict[str, Any]],
    dataset: str,
    shard: str,
) -> pa.Table:
    """Row-aligned index for the embedded (non-failed) rows, in cache order."""
    columns: dict[str, list[Any]] = {
        "uid": [],
        "dataset": [],
        "shard": [],
        "row": [],
        "label3": [],
        "label_bin": [],
        "generator": [],
        "source_split": [],
        "width": [],
        "height": [],
        "format": [],
        "jpeg_q": [],
    }
    extra_keys = sorted(
        set().union(*(set(r) for r in rows))
        - {"image", "path", *columns.keys()}
    )
    for key in extra_keys:
        columns[key] = []
    for row_index in sorted(metas):
        row, measured = rows[row_index], metas[row_index]
        columns["uid"].append(row["uid"])
        columns["dataset"].append(dataset)
        columns["shard"].append(shard)
        columns["row"].append(row_index)
        columns["label3"].append(int(row["label3"]))
        columns["label_bin"].append(int(row["label_bin"]))
        columns["generator"].append(row.get("generator"))
        columns["source_split"].append(row.get("source_split"))
        for key in _MEASURED:
            columns[key].append(measured[key])
        for key in extra_keys:
            columns[key].append(row.get(key))
    return pa.table(columns)


def embed_dataset(
    dataset: str,
    data_dir: str | Path,
    out_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 6,
    limit_shards: int | None = None,
    device: str = "cuda",
) -> None:
    out = Path(out_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    manifest.setdefault("dataset", dataset)
    manifest.setdefault("model", f"{MODEL_NAME}/{PRETRAINED}")
    manifest.setdefault("preprocessing", PREPROCESSING_POLICY)
    manifest.setdefault("preprocessing_hash", POLICY_HASH)
    manifest.setdefault("feature_dim", FEATURE_DIM)
    manifest.setdefault("shards", {})

    torch_device = torch.device(device)
    visual = transform = mean = std = None  # lazy: skip-only runs load nothing

    n_shards = 0
    for shard_name, rows in iter_dataset(dataset, data_dir):
        if limit_shards is not None and n_shards >= limit_shards:
            break
        n_shards += 1
        stem = shard_name.removesuffix(".parquet")
        feature_file = out / f"features-{stem}.npy"
        index_file = out / f"index-{stem}.parquet"
        if feature_file.exists() and index_file.exists():
            cached = pq.read_metadata(index_file).num_rows
            stored = np.load(feature_file, mmap_mode="r").shape[0]
            if cached == stored and shard_name in manifest["shards"]:
                logger.info(f"SKIP {shard_name} ({cached} rows cached)")
                continue

        if visual is None:
            visual, transform, mean, std = _load_encoder(torch_device)

        loader = DataLoader(
            _RowsDataset(rows, transform),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=True,
        )
        features = np.zeros((len(rows), FEATURE_DIM), dtype=np.float16)
        metas: dict[int, dict[str, Any]] = {}
        failures = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for failed, tensors, indices, batch_metas in loader:
                failures += len(failed)
                if tensors is None:
                    continue
                x = (tensors.to(torch_device, non_blocking=True) - mean) / std
                emb = visual(x).float()
                emb = torch.nn.functional.normalize(emb, dim=-1)
                features[indices] = emb.half().cpu().numpy()
                for i, m in zip(indices, batch_metas, strict=True):
                    metas[i] = m

        order = sorted(metas)
        features = features[order]
        table = _index_table(rows, metas, dataset, shard_name)

        tmp = feature_file.with_suffix(".npy.tmp")
        with open(tmp, "wb") as handle:  # np.save on a path would append .npy
            np.save(handle, features)
        os.replace(tmp, feature_file)
        pq.write_table(table, index_file)

        manifest["shards"][shard_name] = {
            "n_rows": len(order),
            "n_failed": failures,
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info(
            f"embedded {shard_name}: {len(order)} rows, {failures} failed decodes"
        )
    logger.info(f"{dataset}: done ({n_shards} shards visited)")
