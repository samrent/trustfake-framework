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

The third knob is about EVIDENCE rather than geometry: `input_mode="crop"`
replaces the resize with fixed-size crops at native resolution, because the
resize low-passes away exactly the high-frequency residue a tampered-image
detector has to read (see the `input_mode` arg on `SIDSetDataModule`).

The fourth, `depth_targets_dir`, is Track C's opt-in: with it set, the fit
and val items carry a third element -- the precomputed teacher depth map of
the SAME image, looked up by the manifest's uid (see
`trustfake.data.depth_targets`). calib and test items never do; every
consumer of those loaders unpacks exactly two values.
"""

from __future__ import annotations

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

from trustfake.data._images import decode_image_cell
from trustfake.data.baselines import image_dims_and_format
from trustfake.data.depth_targets import (
    DEPTH_FRAME,
    ROLE_SOURCE_SPLIT,
    DepthTargetStore,
    check_store_manifest,
    read_store_manifest,
)
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
    """Torch dataset wrapper for a Hugging Face split from SID_Set.

    Items are ``(image, label)``. With a `depth_targets` store they are
    ``(image, label, depth)``, the depth looked up by
    ``f"{uid_prefix}:{row[id_column]}"`` -- the manifest's uid, never the
    row position, which the seeded carve and the filters both move.
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        image_column: str,
        label_column: str,
        transform: transforms.Compose,
        id_column: str | None = None,
        depth_targets: DepthTargetStore | None = None,
        uid_prefix: str = "train",
    ) -> None:
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform
        if depth_targets is not None and id_column is None:
            msg = "depth targets need an id column to key them by uid"
            raise ValueError(msg)
        self.id_column = id_column
        self.depth_targets = depth_targets
        self.uid_prefix = uid_prefix

    @property
    def has_depth_targets(self) -> bool:
        return self.depth_targets is not None

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict[str, Any] = self.hf_dataset[index]
        image = sample[self.image_column]
        label = int(sample[self.label_column])

        if isinstance(image, torch.Tensor):
            pil_image = transforms.ToPILImage()(image)
        else:
            # PIL (HF-decoded), the raw HF struct {bytes, path}, or an array:
            # one decoder, EXIF-corrected like HF's, shared with the depth
            # precompute so a target always belongs to the pixels it is
            # paired with.
            pil_image = decode_image_cell(image)

        image_tensor = self.transform(pil_image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.depth_targets is not None:
            uid = f"{self.uid_prefix}:{sample[self.id_column]}"
            return image_tensor, label_tensor, self.depth_targets[uid]
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
        input_mode: How pixels reach the model. ``"resize"`` (default, the
            historical behaviour): every image is resampled to
            ``image_size x image_size``. ``"crop"``: no resampling at all --
            training takes a random ``image_size`` crop, evaluation takes
            the centre crop (deterministic; both zero-pad the rare image
            smaller than ``image_size``). The point is the tampered class:
            the manipulation evidence is high-frequency and local (seam
            residue, re-decoded texture), and a bilinear resize of a 1024px
            image to 224 is a 4.6x low-pass that erases it BEFORE the model
            trains -- the model then learns whatever survives, which is
            semantics, and tampered images are semantically real. Crop mode
            hands the model native pixel statistics instead. Two costs,
            stated rather than hidden: a single crop sees a fraction of a
            large image, so an edit outside the crop is invisible and
            tampered recall is *understated* for off-crop edits (the honest
            completion is dense/multi-crop scoring with top-k pooling --
            future work, the shards carry the masks for it); and crop mode
            is a different protocol, so its numbers must never sit in a
            table beside resize-mode numbers without saying so. Fit and
            evaluate under the same mode, like every control here.
            Incompatible with ``squarecrop`` -- the square crop exists to
            feed the resize, and composing it with crop mode would read as
            two controls while one of them changes nothing it claims to.
        depth_targets_dir: Track C's opt-in. A depth-target store written by
            `src/precompute_depth.py`; when set, fit and val items become
            ``(image, label, depth)`` with the teacher's map of the same
            image (float32, (1, S, S)); calib, test and holdout stay
            ``(image, label)``. The store's recorded pre-transform
            (`image_size`, `squarecrop`, resize mode) must match this
            datamodule's or setup() refuses it, and every fit row must be
            covered or setup() refuses that too -- a target computed on a
            different view of the pixels, or silently absent for some rows,
            is a wrong number without a symptom. Requires ``input_mode=
            "resize"``: a random crop has no fixed view to precompute.
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
        input_mode: str = "resize",
        limit_test: int | None = None,
        depth_targets_dir: str | Path | None = None,
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
        self.limit_test = limit_test
        self.squarecrop = bool(squarecrop)
        if input_mode not in ("resize", "crop"):
            msg = f"Unknown input_mode '{input_mode}'. Available: resize | crop"
            logger.error(msg)
            raise ValueError(msg)
        if input_mode == "crop" and self.squarecrop:
            # Refused rather than composed: the square crop exists to feed
            # the resize, and in crop mode it would only restrict where the
            # crops come from -- an effect, but not the one its name claims.
            msg = (
                "squarecrop=True with input_mode='crop': the squarecrop "
                "control feeds the resize that crop mode removes. Use one."
            )
            logger.error(msg)
            raise ValueError(msg)
        self.input_mode = input_mode
        if depth_targets_dir is not None and input_mode == "crop":
            # A target is a fixed view of the image; a random crop is not.
            msg = (
                "depth_targets_dir with input_mode='crop': the depth targets are "
                "precomputed on the resized full frame, and a random crop cannot "
                "be aligned with them. Use input_mode='resize'."
            )
            logger.error(msg)
            raise ValueError(msg)
        self.depth_targets_dir = (
            Path(depth_targets_dir) if depth_targets_dir is not None else None
        )
        self._depth_store: DepthTargetStore | None = None

        self.num_classes: int = 3
        self.split_provenance: str = SPLIT_PROVENANCE
        self.image_column: str | None = None
        self.label_column: str | None = None

        self._train_ds: SIDSetTorchDataset | None = None
        self._val_ds: SIDSetTorchDataset | None = None
        self._calib_ds: SIDSetTorchDataset | None = None
        self._test_ds: SIDSetTorchDataset | None = None
        self._holdout_ds: SIDSetTorchDataset | None = None

        if self.input_mode == "resize":
            # The crop goes first or it does nothing: Resize makes every image
            # square, so a crop behind it is a no-op on an already-square
            # tensor. One transform for every role, as it always was.
            self.transform = transforms.Compose(
                [
                    *([CentreSquareCrop()] if self.squarecrop else []),
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ]
            )
            self.train_transform = self.transform
        else:
            # Crop mode: native pixels, no resampling. Train crops randomly
            # (each epoch sees a different window, which is the cheap
            # substitute for dense coverage); val/calib/test crop the centre,
            # deterministically -- the calibration temperature and the
            # moderation thresholds must be fitted on a reproducible view.
            # `self.transform` stays the EVAL transform: every existing
            # consumer of the attribute reads the deterministic one.
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(self.image_size, pad_if_needed=True),
                    transforms.ToTensor(),
                ]
            )
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(self.image_size),
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

    @property
    def has_depth_targets(self) -> bool:
        """Whether fit/val items carry a depth target (Track C opt-in)."""
        return self.depth_targets_dir is not None

    def _open_depth_store(
        self, fit_shards: list[Path], id_column: str | None, fit_dataset: Dataset
    ) -> DepthTargetStore:
        """Open the precomputed store and prove it fits this run: same
        pre-transform as this datamodule, and a target for every fit row.
        Both refusals happen here, at setup, rather than at the first batch
        that would have needed the missing map."""
        if id_column is None:
            msg = (
                "depth_targets_dir needs an id column to key targets by uid; "
                f"none of {list(self.ID_COLUMN_CANDIDATES)} is in the shards."
            )
            logger.error(msg)
            raise ValueError(msg)
        if "Random" in repr(self.train_transform):
            # A target is one fixed view of the image; a stochastic train
            # transform would pair each epoch's pixels with another view's
            # geometry, and the loss would keep decreasing regardless.
            msg = (
                "depth_targets_dir with a stochastic train transform "
                f"({self.train_transform!r}): precomputed targets can only "
                "be aligned with a deterministic view of the image."
            )
            logger.error(msg)
            raise ValueError(msg)
        meta = read_store_manifest(self.depth_targets_dir)
        # The grid is tied to the head: it emits image_size // 2, and the
        # loss refuses to resample, so a store at another grid is refused
        # here rather than at the first batch.
        check_store_manifest(
            meta,
            image_size=self.image_size,
            squarecrop=self.squarecrop,
            input_mode=self.input_mode,
            output_size=self.image_size // 2,
            frame=DEPTH_FRAME,
        )
        store = DepthTargetStore(
            self.depth_targets_dir, fit_shards, ROLE_SOURCE_SPLIT["fit"]
        )
        prefix = ROLE_SOURCE_SPLIT["fit"]
        store.assert_covers(f"{prefix}:{i}" for i in fit_dataset[id_column])
        logger.info(
            f"Depth targets: {len(store)} maps at {store.size}x{store.size} from "
            f"{meta.get('teacher')} ({self.depth_targets_dir}); fit and val items "
            "carry a third element."
        )
        return store

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

        if self.depth_targets_dir is not None:
            self._depth_store = self._open_depth_store(
                manifest["fit"], id_column, dataset["fit"]
            )

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

        def _wrap(
            hf_dataset: Dataset,
            transform: transforms.Compose | None = None,
            depth: bool = False,
        ) -> SIDSetTorchDataset:
            return SIDSetTorchDataset(
                hf_dataset,
                image_column=self.image_column,
                label_column=self.label_column,
                transform=transform if transform is not None else self.transform,
                id_column=id_column if depth else None,
                depth_targets=self._depth_store if depth else None,
                uid_prefix=ROLE_SOURCE_SPLIT["fit"],
            )

        # Only the fit-train slice gets the (possibly stochastic) train
        # transform; val is model selection and must see the deterministic
        # eval view, like calib and test. In resize mode the two are the
        # same object and this changes nothing. Depth targets ride with the
        # two fit-derived slices only.
        self._train_ds = _wrap(
            fit_split["train"], transform=self.train_transform, depth=True
        )
        self._val_ds = _wrap(fit_split["test"], depth=True)
        self._calib_ds = _wrap(dataset["calib"])
        test_split = dataset["test"]
        if self.limit_test is not None and self.limit_test < test_split.num_rows:
            # A PREFIX, not a random sample: the first N rows of the same
            # ordered split, so the capped set is nested inside the full one
            # and a number from it is a number from a subset of the same
            # split rather than from a different draw. That is what lets an
            # expensive attack (Square at 500 queries is ~7h on the full test
            # set, per configuration) be reported beside a cheap one without
            # the two describing different populations.
            #
            # calib is deliberately NOT capped: the thresholds and temperature
            # must be fitted on the same calibration data for every condition,
            # or a cheap and an expensive condition are being read against
            # different policies.
            test_split = test_split.select(range(self.limit_test))
        self._test_ds = _wrap(test_split)
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
            f"{list(GEOMETRY_FILTERED_ROLES)}), squarecrop: {self.squarecrop}, "
            f"input_mode: {self.input_mode}"
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
