"""So-Fake-OOD: the distribution-shift condition.

A test-only out-of-distribution benchmark. Its string labels map onto
SID-Set's class ids exactly -- ``REAL -> 0``, ``FULL_SYNTHETIC -> 1``,
``TAMPERED -> 2`` -- which makes it the better shift condition than FakeClue:
FakeClue is binary and collapses the two forgery modalities, so evaluating
there throws away ``detection_auroc_tampered`` / ``detection_auroc_synthetic``.
Here the breakout survives, and "does the tampered class degrade faster under
shift than the synthetic one" becomes a question you can actually ask.

**There is deliberately no calib split here, and that is the point.**
Thresholds and temperature must be fitted on the IN-DOMAIN calib split and
then applied unchanged to this data, because that is the deployment
situation: you calibrate on what you have and serve on what arrives.
Refitting on shifted data would hide the exact failure this condition exists
to expose -- selective prediction and conformal risk control rest on
exchangeability, distribution shift breaks it, and a threshold refitted after
the shift papers over the breakage. Any So-Fake-OOD number must say which
calib split its thresholds came from.

The dataset is ~135 GB across 46 test shards. Shards are the unit of
selection here, as in :mod:`trustfake.data.manifest`: taking whole shards
(deterministically, seeded by ``manifest_seed``) keeps the sample a property
of the protocol rather than of whatever happened to be downloaded, and a
partial download is a legitimate sample rather than a broken one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L  # noqa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from trustfake.data.manifest import DEFAULT_MANIFEST_SEED
from trustfake.logging import get_logger

__all__ = [
    "SO_FAKE_OOD_PROVENANCE",
    "STRING_LABEL_TO_ID",
    "SoFakeOODDataModule",
    "SoFakeOODTorchDataset",
    "select_shards",
]

logger = get_logger("so-fake-ood")

#: So-Fake-OOD ships string labels. These are SID-Set's ids, so a SID-Set
#: checkpoint is scored here without any remapping -- and, unlike FakeClue,
#: without collapsing synthetic and tampered into one "fake" class.
STRING_LABEL_TO_ID: dict[str, int] = {
    "REAL": 0,
    "FULL_SYNTHETIC": 1,
    "TAMPERED": 2,
}

SO_FAKE_OOD_PROVENANCE = (
    "So-Fake-OOD test split (shard sample); test-only by construction -- "
    "thresholds and temperature come from the in-domain calib split, never "
    "from this data"
)


def select_shards(available: list[Path], n_shards: int | None, seed: int) -> list[Path]:
    """Deterministically choose ``n_shards`` of the shards on disk.

    Seeded by ``manifest_seed`` (a project constant, not the experiment seed)
    so the OOD sample does not move when a training run does. ``None`` takes
    everything present. Sorted first, so the choice depends on shard identity
    rather than on filesystem order.
    """
    ordered = sorted(available)
    if not ordered:
        raise FileNotFoundError("no So-Fake-OOD parquet shards found")
    if n_shards is None or n_shards >= len(ordered):
        return ordered
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    generator = torch.Generator().manual_seed(seed)
    picked = torch.randperm(len(ordered), generator=generator)[:n_shards].tolist()
    return [ordered[i] for i in sorted(picked)]


class SoFakeOODTorchDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    """``(image, label)`` pairs, matching the project contract.

    So-Fake-OOD carries a ``mask`` column, but nothing downstream consumes a
    mask yet, so it is not returned -- adding a third element would break
    every pipe that unpacks two. It is left in the shards for the
    localization work TODO 4 describes.
    """

    def __init__(self, hf_dataset: Any, transform: transforms.Compose) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.hf_dataset[index]
        raw = sample["label"]
        if isinstance(raw, str):
            key = raw.upper()
            if key not in STRING_LABEL_TO_ID:
                raise KeyError(
                    f"unknown So-Fake-OOD label {raw!r}; known: "
                    f"{sorted(STRING_LABEL_TO_ID)}"
                )
            label = STRING_LABEL_TO_ID[key]
        else:
            label = int(raw)
        image = sample["image"].convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


class SoFakeOODDataModule(L.LightningDataModule):
    """So-Fake-OOD as a test-only shift condition.

    Args:
        data_dir: Directory holding the ``test_image-*.parquet`` shards.
        n_shards: How many shards to sample (``None`` = every shard present).
            Each is ~3 GB, so a partial download is the normal case.
        manifest_seed: Seeds the shard choice. A project constant.
        input_mode: ``"resize"`` or ``"crop"``, matching
            :class:`~trustfake.data.sid_set.SIDSetDataModule`. Evaluate under
            the mode the checkpoint was fitted under.
        limit_test: Prefix cap, so an expensive attack is comparable to a
            cheap one.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/so_fake_ood",
        n_shards: int | None = None,
        manifest_seed: int = DEFAULT_MANIFEST_SEED,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        input_mode: str = "resize",
        normalization_layer: nn.Module | None = None,
        pin_memory: bool = True,
        limit_test: int | None = None,
    ) -> None:
        super().__init__()
        if input_mode not in ("resize", "crop"):
            raise ValueError(
                f"input_mode must be 'resize' or 'crop', got {input_mode!r}"
            )
        self.data_dir = Path(data_dir)
        self.n_shards = n_shards
        self.manifest_seed = manifest_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.input_mode = input_mode
        self.normalization_layer = (
            normalization_layer if normalization_layer is not None else nn.Identity()
        )
        self.pin_memory = pin_memory
        self.limit_test = limit_test
        self._test_ds: SoFakeOODTorchDataset | None = None
        self.shards: list[Path] = []

    def _transform(self) -> transforms.Compose:
        if self.input_mode == "crop":
            return transforms.Compose(
                [transforms.CenterCrop(self.image_size), transforms.ToTensor()]
            )
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        from datasets import load_dataset

        self.shards = select_shards(
            list(self.data_dir.rglob("*.parquet")), self.n_shards, self.manifest_seed
        )
        logger.info(
            f"So-Fake-OOD: {len(self.shards)} shard(s) selected "
            f"(seed {self.manifest_seed}) -- {SO_FAKE_OOD_PROVENANCE}"
        )
        dataset = load_dataset(
            "parquet", data_files=[str(p) for p in self.shards], split="train"
        )
        if self.limit_test is not None:
            dataset = dataset.select(range(min(self.limit_test, len(dataset))))
        self._test_ds = SoFakeOODTorchDataset(dataset, self._transform())

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before requesting the test dataloader.")
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def test_dataset(self) -> SoFakeOODTorchDataset:
        if self._test_ds is None:
            raise RuntimeError("Call setup() first.")
        return self._test_ds

    def calib_dataloader(self) -> DataLoader:
        """Refused, deliberately.

        Fitting a threshold on shifted data hides the failure this condition
        exists to measure. Use the in-domain calib split and report which one.
        """
        raise NotImplementedError(
            "So-Fake-OOD is test-only by construction. Fit temperature and "
            "moderation thresholds on the IN-DOMAIN calib split and apply them "
            "unchanged here -- refitting on shifted data hides the "
            "exchangeability violation this condition exists to expose."
        )
