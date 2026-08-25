"""PyTorch Lightning datamodule for the SID_Set dataset.

Splits are governed by the shard-level manifest in `trustfake.data.manifest`:
fit comes from train shards, calib and test from disjoint validation shards,
and the model-selection slice (what Lightning sees as `val`) is carved from
fit at row level. See the manifest module docstring for why this is structural
rather than disciplinary.

The datamodule also carries the two GEOMETRY CONTROLS, because both have to
act before the model sees anything (see `trustfake.data.baselines` for the
artifact they answer):

  * `geometry_filter` -- a row filter on the reported splits, so the
    ``width == height`` rule carries no signal in the subset that is scored.
  * `squarecrop` -- a centre crop to the short side, applied BEFORE the
    resize. Order is the whole substance of it: after the resize every image
    is already square, so a crop there is a no-op and the "control" would
    silently do nothing.
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
from datasets import Image as HFImage
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from trustfake.data.baselines import image_dims_and_format
from trustfake.data.manifest import (
    DEFAULT_MANIFEST_SEED,
    GEOMETRY_FILTERS,
    SPLIT_PROVENANCE,
    build_manifest,
    geometry_selection,
)
from trustfake.logging import get_logger

logger = get_logger("sid-set")

__all__ = ["SIDSetDataModule", "CentreSquareCrop"]

#: Roles the row-level geometry filter applies to. calib is in the list with
#: test on purpose: it fits the temperature and the moderation thresholds,
#: so it must come from the same distribution as the split those thresholds
#: are reported on. fit is NOT -- the control asks "does this model still
#: work when geometry carries no signal?", which is a question about a model
#: trained on the real distribution.
GEOMETRY_FILTERED_ROLES: tuple[str, ...] = ("calib", "test", "holdout")


def _find_column(columns: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a {kind} column in: {columns}")


class CentreSquareCrop:
    """Crop a PIL image to a centred square of its short side.

    THE PIXEL-LEVEL GEOMETRY CONTROL, and it only works in one position: in
    front of the resize. SID-Set's fake classes are essentially always
    square and most reals are not, so ``width == height -> fake`` scores
    above a CLIP probe on the raw split. Cropping every image to a square
    makes geometry carry zero label information by construction, so a model
    evaluated under this condition cannot be riding that artifact. Placed
    after `transforms.Resize`, it would crop an already-square 224x224
    tensor and change nothing at all.

    Two honest caveats. Fit AND evaluate under the same condition, or the
    control is just a covariate shift wearing a control's name. And the crop
    preserves the short side, so it does not touch the ``short side ==
    1024`` residue (see `trustfake.data.baselines`).

    A class rather than a `transforms.Lambda` so it pickles into dataloader
    workers and prints itself in a transform repr.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _original_dims(
    hf_dataset: Dataset, image_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Original (pre-resize) width and height of every row.

    Reuses `trustfake.data.baselines.image_dims_and_format`, which is also
    what computes the trivial baselines -- one reader, so the filter and the
    number it is answering can never disagree about what "square" means.

    The image column is cast to ``decode=False`` first when it is an HF
    Image feature: that hands back the raw ``{bytes, path}`` struct so only
    the file header is parsed. Decoding megapixels to measure them would
    make the filter cost more than the evaluation it enables.
    """
    dataset = hf_dataset
    features = getattr(dataset, "features", None) or {}
    if isinstance(features.get(image_column), HFImage):
        dataset = dataset.cast_column(image_column, HFImage(decode=False))

    widths = np.empty(len(dataset), dtype=np.int64)
    heights = np.empty(len(dataset), dtype=np.int64)
    for index in range(len(dataset)):
        width, height, _ = image_dims_and_format(dataset[index][image_column])
        widths[index] = width
        heights[index] = height
    return widths, heights


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
        geometry_filter: Row-level geometry control, one of
            `trustfake.data.manifest.GEOMETRY_FILTERS`: none (default,
            unchanged) | square | nonsquare | matched. Applied to
            `GEOMETRY_FILTERED_ROLES` only, and seeded by `manifest_seed`,
            so the controlled subset is a property of the protocol and not
            of the training run. Anything but 'none' changes what a reported
            number means -- and 'matched' changes the class prior too, so
            the majority-class floor moves with it. Say so in the report.
            A filter that leaves a role with fewer than two classes is
            REFUSED, not warned about: on real SID-Set 'nonsquare' does
            exactly that, and every metric downstream of it is undefined
            rather than merely wrong (see `_apply_geometry_filter`).
        squarecrop: Centre-crop every image to its short side BEFORE the
            resize (`CentreSquareCrop`). The pixel-level geometry control.
            Use it for fit and evaluation together or it is a covariate
            shift, not a control.
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
        geometry_filter: str = "none",
        squarecrop: bool = False,
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
        if geometry_filter not in GEOMETRY_FILTERS:
            msg = (
                f"Unknown geometry_filter '{geometry_filter}'. Available: "
                f"{list(GEOMETRY_FILTERS)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.geometry_filter = geometry_filter
        self.squarecrop = bool(squarecrop)

        self.num_classes: int = 3
        self.split_provenance: str = SPLIT_PROVENANCE
        self.image_column: str | None = None
        self.label_column: str | None = None

        self._train_ds: SIDSetTorchDataset | None = None
        self._val_ds: SIDSetTorchDataset | None = None
        self._calib_ds: SIDSetTorchDataset | None = None
        self._test_ds: SIDSetTorchDataset | None = None
        self._holdout_ds: SIDSetTorchDataset | None = None

        # The crop goes first or it does nothing: Resize makes every image
        # square, so a crop behind it is a no-op on an already-square tensor.
        self.transform = transforms.Compose(
            [
                *([CentreSquareCrop()] if self.squarecrop else []),
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

    def _apply_geometry_filter(self, hf_dataset: Dataset, role: str) -> Dataset:
        """Select the geometry-controlled rows of one role and log what the
        control cost and what it bought.

        The square rate per label is logged after filtering because it is
        the number that proves the control did its job: equal rates across
        labels means ``width == height`` carries no label information, which
        is the entire claim a controlled row makes.

        A role left with FEWER THAN TWO CLASSES raises rather than warns,
        and the choice is deliberate. On real SID-Set
        ``geometry_filter=nonsquare`` leaves 569 rows that are all label 0 --
        the filter strips essentially every fake, which is the artifact
        stated in its starkest form. The unequal-rate warning below cannot
        fire on that subset (a single key in `rates` is trivially equal to
        itself), so the loudest thing in the log would be an INFO line, and
        what comes out the far end is not a wrong number but an undefined
        one: macro accuracy is capped at 1/3 by the two classes that have no
        rows, detection AUROC is NaN for want of a positive, and on `calib`
        the moderation thresholds are fitted with zero fakes and then
        applied to `test`. None of those has a runtime symptom. The module
        already refuses the neighbouring failure -- `geometry_selection`
        raises when a filter selects no rows at all, "there is no controlled
        subset to report on" -- and a single-class subset is the same
        sentence with one more row in it. Refusing here costs a run that was
        going to produce numbers nobody could use; warning costs a table
        that looks finished.
        """
        widths, heights = _original_dims(hf_dataset, self.image_column)
        labels = np.asarray(hf_dataset[self.label_column], dtype=np.int64)
        keep = geometry_selection(
            widths, heights, labels, self.geometry_filter, self.manifest_seed
        )

        square = (widths == heights)[keep]
        kept_labels = labels[keep]
        rates = {
            int(label): round(float(square[kept_labels == label].mean()), 4)
            for label in sorted(set(kept_labels.tolist()))
        }
        logger.info(
            f"geometry_filter '{self.geometry_filter}' on '{role}': "
            f"{len(keep)}/{len(hf_dataset)} rows kept; square rate by label "
            f"{rates}. Report this protocol beside any number from it."
        )
        if len(rates) < 2:
            present = sorted(rates)
            missing = sorted(set(range(self.num_classes)) - set(present))
            msg = (
                f"geometry_filter '{self.geometry_filter}' left role '{role}' "
                f"with a SINGLE class: {len(keep)}/{len(hf_dataset)} rows kept, "
                f"all of label {present} (missing {missing}). Nothing "
                "downstream can be read from that subset -- macro accuracy is "
                f"capped at 1/{self.num_classes}, detection AUROC is undefined "
                "for want of both classes, and on 'calib' the moderation "
                "thresholds would be fitted with none. Use a filter that "
                "leaves both classes ('matched' equalises the geometry rate "
                "without emptying a class), or report the raw split and state "
                "the artifact instead."
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(set(rates.values())) > 1:
            logger.warning(
                f"geometry_filter '{self.geometry_filter}' left unequal square "
                f"rates across labels on '{role}' ({rates}) -- geometry still "
                "carries label information in this subset."
            )
        return hf_dataset.select(keep)

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

        # The geometry control runs after the uid tripwire, so the tripwire
        # still sees every row it is meant to police.
        if self.geometry_filter != "none":
            for role in GEOMETRY_FILTERED_ROLES:
                if role in dataset:
                    dataset[role] = self._apply_geometry_filter(dataset[role], role)

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
        logger.info(
            "Geometry protocol -- "
            f"filter: {self.geometry_filter} (roles "
            f"{list(GEOMETRY_FILTERED_ROLES)}), squarecrop: {self.squarecrop}"
        )

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
