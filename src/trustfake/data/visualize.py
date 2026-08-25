"""Visualization utilities for SID_Set samples."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .sid_set import SIDSetDataModule

__all__ = ["visualize_sample"]

_SPLIT_TO_DATASET_ATTR = {
    "train": "train_dataset",
    "val": "val_dataset",
    "test": "test_dataset",
}


def visualize_sample(
    datamodule: SIDSetDataModule, index: int, split: str = "train"
) -> None:
    """Display the image and label at `index` for the given split of `datamodule`."""
    if split not in _SPLIT_TO_DATASET_ATTR:
        raise ValueError(
            f"split must be one of {list(_SPLIT_TO_DATASET_ATTR)}, got {split!r}"
        )

    dataset = getattr(datamodule, _SPLIT_TO_DATASET_ATTR[split])
    image, label = dataset[index]

    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"{split} sample #{index} — label: {label.item()}")
    plt.axis("off")
    plt.show()
