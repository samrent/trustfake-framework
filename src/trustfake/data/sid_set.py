"""PyTorch Lightning datamodule for the SID_Set dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L  # noqa
import torch
import torch.nn as nn
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

__all__ = ["SIDSetDataModule"]


def _find_column(columns: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a {kind} column in: {columns}")


def _resolve_splits(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
) -> tuple[Dataset, Dataset, Dataset]:
    if isinstance(dataset, DatasetDict):
        train_ds = dataset["train"]
        val_ds = dataset["validation"]
        test_ds = dataset[
            "validation"
        ]  # TODO Separated test split available in the official git repository
        return train_ds, val_ds, test_ds
    else:
        raise ValueError("Expected a DatasetDict with train/val/test splits.")


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
        elif isinstance(image, torch.Tensor):
            pil_image = transforms.ToPILImage()(image)
        else:
            pil_image = Image.fromarray(image).convert("RGB")

        image_tensor = self.transform(pil_image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class SIDSetDataModule(L.LightningDataModule):
    """Lightning datamodule for SID_Set with images resized to 224x224."""

    IMAGE_COLUMN_CANDIDATES = ("image", "img", "pixel_values")
    LABEL_COLUMN_CANDIDATES = ("label", "labels", "target", "class", "y")

    def __init__(
        self,
        dataset_name: str = "saberzl/SID_Set",
        cache_dir: str | Path = "data/sid_set",
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        seed: int = 42,
        pin_memory: bool = True,
        normalization_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.cache_dir = str(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.seed = seed
        self.pin_memory = pin_memory
        self.normalization_layer = (
            normalization_layer if normalization_layer is not None else nn.Identity()
        )

        self.num_classes: int = 3
        self.image_column: str | None = None
        self.label_column: str | None = None

        self._train_ds: SIDSetTorchDataset | None = None
        self._val_ds: SIDSetTorchDataset | None = None
        self._test_ds: SIDSetTorchDataset | None = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            # Already setup
            return

        dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        train_hf, val_hf, test_hf = _resolve_splits(dataset)

        columns = list(train_hf.column_names)
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

        self._train_ds = SIDSetTorchDataset(
            train_hf,
            image_column=self.image_column,
            label_column=self.label_column,
            transform=self.transform,
        )
        self._val_ds = SIDSetTorchDataset(
            val_hf,
            image_column=self.image_column,
            label_column=self.label_column,
            transform=self.transform,
        )
        self._test_ds = SIDSetTorchDataset(
            test_hf,
            image_column=self.image_column,
            label_column=self.label_column,
            transform=self.transform,
        )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._train_ds is None:
            raise RuntimeError("Call setup() before requesting the train dataloader.")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._val_ds is None:
            raise RuntimeError("Call setup() before requesting the val dataloader.")
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before requesting the test dataloader.")
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

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
    def test_dataset(self) -> SIDSetTorchDataset:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_ds
