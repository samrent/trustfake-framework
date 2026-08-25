"""PyTorch Lightning datamodule for the SID_Set dataset.

Splits are governed by the shard-level manifest in `trustfake.data.manifest`:
fit comes from train shards, calib and test from disjoint validation shards,
and the model-selection slice (what Lightning sees as `val`) is carved from
fit at row level. See the manifest module docstring for why this is structural
rather than disciplinary.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import lightning as L  # noqa
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from trustfake.data.manifest import (
    DEFAULT_MANIFEST_SEED,
    SPLIT_PROVENANCE,
    build_manifest,
)
from trustfake.logging import get_logger

logger = get_logger("sid-set")

__all__ = ["SIDSetDataModule"]


def _find_column(columns: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a {kind} column in: {columns}")


class SIDSetTorchDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset wrapper for a Hugging Face split from SID_Set."""

    def __init__(
        self,
        hf_dataset: Dataset,
        image_column: str,
        label_column: str,
        transform: transforms.Compose,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict[str, Any] = self.hf_dataset[index]
        image = sample[self.image_column]
        label = int(sample[self.label_column])

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, dict) and "bytes" in image:
            # Raw parquet rows carry the HF Image struct {bytes, path}.
            pil_image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        elif isinstance(image, torch.Tensor):
            pil_image = transforms.ToPILImage()(image)
        else:
            pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert(
                "RGB"
            )

        image_tensor = self.transform(pil_image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class SIDSetDataModule(L.LightningDataModule):
    """Lightning datamodule for SID_Set with images resized to 224x224.

    Args:
        data_dir: Directory holding the SID_Set parquet shards
            (train-*.parquet / validation-*.parquet, searched recursively).
            See jobs/download_sidset.sh.
        profile: Manifest profile choosing how many shards each role gets
            (see `trustfake.data.manifest.PROFILES`).
        manifest_seed: Seed for the shard-to-role assignment. A project
            constant -- deliberately independent of the experiment seed, so
            the reported split never moves with it.
        val_fraction: Fraction of fit carved out (at row level) as the
            model-selection slice served by `val_dataloader`.
        seed: Experiment seed; governs the row-level fit/val carve only.
    """

    IMAGE_COLUMN_CANDIDATES = ("image", "img", "pixel_values")
    LABEL_COLUMN_CANDIDATES = ("label", "labels", "target", "class", "y")
    ID_COLUMN_CANDIDATES = ("img_id", "id", "image_id")

    def __init__(
        self,
        data_dir: str | Path = "data/sid_set",
        profile: str = "full",
        manifest_seed: int = DEFAULT_MANIFEST_SEED,
        val_fraction: float = 0.05,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        seed: int = 42,
        pin_memory: bool = True,
        normalization_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = str(data_dir)
        self.profile = profile
        self.manifest_seed = manifest_seed
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.seed = seed
        self.pin_memory = pin_memory
        self.normalization_layer = (
            normalization_layer if normalization_layer is not None else nn.Identity()
        )

        self.num_classes: int = 3
        self.split_provenance: str = SPLIT_PROVENANCE
        self.image_column: str | None = None
        self.label_column: str | None = None

        self._train_ds: SIDSetTorchDataset | None = None
        self._val_ds: SIDSetTorchDataset | None = None
        self._calib_ds: SIDSetTorchDataset | None = None
        self._test_ds: SIDSetTorchDataset | None = None
        self._holdout_ds: SIDSetTorchDataset | None = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def _check_ids(self, dataset: dict[str, Dataset], id_column: str) -> None:
        """Row-key tripwire: img_id must be unique within each role, and
        across calib+test jointly (they share the 'validation' uid prefix)."""
        for role, ds in dataset.items():
            ids = ds[id_column]
            if len(set(ids)) != len(ids):
                msg = f"Duplicate {id_column} within role '{role}' -- key broken."
                logger.error(msg)
                raise ValueError(msg)
        joint = list(dataset["calib"][id_column]) + list(dataset["test"][id_column])
        if len(set(joint)) != len(joint):
            msg = (
                f"Duplicate {id_column} across calib and test -- the same image "
                "would be used to fit post-hoc quantities and to report."
            )
            logger.error(msg)
            raise ValueError(msg)

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            # Already setup
            return

        manifest = build_manifest(self.data_dir, self.profile, self.manifest_seed)
        data_files = {role: [str(p) for p in paths] for role, paths in manifest.items()}
        dataset = load_dataset("parquet", data_files=data_files)

        columns = list(dataset["fit"].column_names)
        self.image_column = _find_column(
            columns,
            self.IMAGE_COLUMN_CANDIDATES,
            "image",
        )
        self.label_column = _find_column(
            columns,
            self.LABEL_COLUMN_CANDIDATES,
            "label",
        )
        id_column = next((c for c in self.ID_COLUMN_CANDIDATES if c in columns), None)
        if id_column is not None:
            self._check_ids(dataset, id_column)

        # Model selection sees a row-level slice of fit; calib and test never.
        fit_split = dataset["fit"].train_test_split(
            test_size=self.val_fraction, seed=self.seed
        )

        def _wrap(hf_dataset: Dataset) -> SIDSetTorchDataset:
            return SIDSetTorchDataset(
                hf_dataset,
                image_column=self.image_column,
                label_column=self.label_column,
                transform=self.transform,
            )

        self._train_ds = _wrap(fit_split["train"])
        self._val_ds = _wrap(fit_split["test"])
        self._calib_ds = _wrap(dataset["calib"])
        self._test_ds = _wrap(dataset["test"])
        if "holdout" in dataset:
            self._holdout_ds = _wrap(dataset["holdout"])

        logger.info(
            "SID_Set splits -- "
            f"train: {len(self._train_ds)}, val (selection, from fit): "
            f"{len(self._val_ds)}, calib: {len(self._calib_ds)}, "
            f"test: {len(self._test_ds)}"
            + (
                f", holdout (sealed): {len(self._holdout_ds)}"
                if self._holdout_ds
                else ""
            )
        )
        logger.info(f"Split provenance: {self.split_provenance}")

    def _loader(
        self, dataset: SIDSetTorchDataset, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._train_ds is None:
            raise RuntimeError("Call setup() before requesting the train dataloader.")
        return self._loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._val_ds is None:
            raise RuntimeError("Call setup() before requesting the val dataloader.")
        return self._loader(self._val_ds, shuffle=False)

    def calib_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._calib_ds is None:
            raise RuntimeError("Call setup() before requesting the calib dataloader.")
        return self._loader(self._calib_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before requesting the test dataloader.")
        return self._loader(self._test_ds, shuffle=False)

    @property
    def train_dataset(self) -> SIDSetTorchDataset:
        if self._train_ds is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_ds

    @property
    def val_dataset(self) -> SIDSetTorchDataset:
        if self._val_ds is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_ds

    @property
    def calib_dataset(self) -> SIDSetTorchDataset:
        if self._calib_ds is None:
            raise RuntimeError("Call setup() before accessing calib_dataset.")
        return self._calib_ds

    @property
    def test_dataset(self) -> SIDSetTorchDataset:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_ds

    @property
    def holdout_dataset(self) -> SIDSetTorchDataset:
        """The sealed holdout. No dataloader on purpose: touching it is a
        deliberate, one-time act, not part of any pipeline."""
        if self._holdout_ds is None:
            raise RuntimeError(
                "No holdout in this profile (or setup() not called). "
                "Use profile 'train_holdout'."
            )
        return self._holdout_ds
