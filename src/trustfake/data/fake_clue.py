"""FakeClue: the cross-dataset benchmark, with an identity firewall.

FakeClue (https://huggingface.co/datasets/lingcco/FakeClue) is a second image
distribution to evaluate a SID-Set-trained detector against. It is what makes
a generalisation claim possible at all: every number this project reports
today is on one dataset, and a detector that only works on SID-Set is a
detector that has learned SID-Set.

Three things about it are traps, and all three are handled here rather than
left to the caller.

**The label convention is INVERTED relative to this project.** FakeClue's
own json uses ``0 = fake, 1 = real``; SID-Set (and every metric here, via
``real_class=0`` and ``p_fake = 1 - P(real)``) uses ``0 = real``. Loading the
raw labels would silently invert every detection AUROC, moderation decision
and per-class row -- numbers that look plausible and are exactly backwards.
Labels are therefore remapped to the project convention at read time, and the
raw ids are kept in module constants so the mapping is auditable.

**The split leaks identities.** FakeClue's 1168 deepfake rows are drawn from
only 666 FaceForensics++ identities: a mean of 3.15 rows per identity, and
194 identities appear under BOTH labels -- the same face present as real and
as fake. A row-level calib/test split therefore puts the same person on both
sides, often on both sides of the label, and what gets measured is identity
memorisation rather than detection. This module splits at the level of
*identity groups* (union-find over the FF++ identities encoded in the frame
directory, parent directory elsewhere), so no identity crosses the calib/test
boundary. ``group_split=False`` restores the naive row split for comparison;
it is not a default and a report that uses it must say so.

**The class prior is not 0.5 and a metadata rule beats chance.** The test
split is 63.8% fake, and "square image => fake" -- which reads no pixels --
scores 0.690 on it. Read any accuracy against
:func:`trustfake.data.baselines.compute_trivial_baselines`, never against 0.5.
``width``/``height`` are carried through for exactly that reason.

Batches are ``(image, label)`` 2-tuples, matching
:class:`trustfake.data.sid_set.SIDSetTorchDataset`. FakeClue ships no
manipulation masks, so there is nothing to put in a third element and none is
invented.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as L  # noqa
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from trustfake.data.manifest import DEFAULT_MANIFEST_SEED
from trustfake.logging import get_logger

__all__ = [
    "FAKE_CLUE_FAKE_ID",
    "FAKE_CLUE_REAL_ID",
    "FAKE_CLUE_PROVENANCE",
    "FakeClueDataModule",
    "FakeClueTorchDataset",
    "group_key",
    "assign_groups",
]

logger = get_logger("fake-clue")

#: FakeClue's own ids, as written in ``data_json/{split}.json``.
FAKE_CLUE_FAKE_ID = 0
FAKE_CLUE_REAL_ID = 1

#: This project's convention, shared with SID-Set: real is class 0, and every
#: other class is a kind of fake. FakeClue is binary, so fake collapses to 1.
PROJECT_REAL_ID = 0
PROJECT_FAKE_ID = 1

FAKE_CLUE_PROVENANCE = (
    "FakeClue test split, identity-grouped calib/test partition "
    "(labels remapped from FakeClue's 0=fake convention to 0=real)"
)


def group_key(image_path: str) -> tuple[str, ...]:
    """The identities a row belongs to, for the split firewall.

    FaceForensics++ frame directories encode identities in their name: a real
    clip is ``.../frames/<id>`` and a manipulated one is
    ``.../frames/<src>_<tgt>``, so a single fake row belongs to *two*
    identities and can collide with real rows of either. Returning a tuple of
    identities (rather than the directory) is what lets
    :func:`assign_groups` union those collisions together.

    Everything outside FF++ has no identity structure -- generated images are
    independent samples -- so the parent directory is the group, which keeps
    a source folder from being split across roles.
    """
    parent = str(Path(image_path).parent)
    if not parent.startswith("ff++"):
        return (parent,)
    stem = Path(parent).name
    ids = tuple(part for part in stem.split("_") if part.isdigit())
    return ids or (parent,)


def assign_groups(
    records: list[dict[str, Any]], calib_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Partition record indices into (calib, test) with no shared identity.

    Union-find over :func:`group_key`: rows sharing any identity end up in one
    component, and components -- not rows -- are dealt to the two roles. The
    deal is seeded by ``manifest_seed`` (a project constant, not the
    experiment seed) so the partition is a property of the protocol and does
    not move when a training run does.

    Components are shuffled and then taken in order until the calib quota is
    met, so the split is deterministic and the roles are disjoint by
    construction. Sizes are approximate: a component is indivisible, so the
    realised calib fraction lands near the requested one rather than on it.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    keys = [group_key(r["image"]) for r in records]
    for k in keys:
        for other in k[1:]:
            union(k[0], other)

    members: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        members[find(k[0])].append(i)

    roots = sorted(members)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(roots), generator=generator).tolist()

    # Deal per label, not over all components at once. FakeClue's components
    # are large and label-pure (`genimage/fake` alone is 916 rows), so a
    # single deal lands calib at 87.5% fake against a 53.7% test -- and the
    # temperature and moderation thresholds fitted there do not transfer to a
    # split with a different prior. Stratifying keeps both roles near the
    # dataset prior while components stay whole, so the firewall is intact.
    by_label: dict[int, list[int]] = defaultdict(list)
    for position in order:
        root = roots[position]
        labels = {int(records[i].get("label", -1)) for i in members[root]}
        # A mixed component cannot be stratified; it is dealt on its majority
        # label so it still lands somewhere deterministic.
        key = labels.pop() if len(labels) == 1 else -1
        by_label[key].append(root)

    # Largest component first, each one dealt to whichever role is furthest
    # below its quota. Taking components in shuffled order and switching
    # buckets once a running total crosses the target fails whenever ONE
    # component is bigger than the quota -- `genimage/fake` is 916 rows, so
    # that component alone would swallow calib and the split would be 93/7.
    # Greedy-by-deficit degrades gracefully instead: a component too big for
    # either quota lands on the emptier side rather than the first side.
    calib: list[int] = []
    test: list[int] = []
    for key in sorted(by_label):
        rows = sum(len(members[r]) for r in by_label[key])
        calib_target = calib_fraction * rows
        test_target = rows - calib_target
        n_calib = n_test = 0
        ordered = sorted(by_label[key], key=lambda r: -len(members[r]))
        for root in ordered:
            size = len(members[root])
            if (calib_target - n_calib) >= (test_target - n_test):
                calib.extend(members[root])
                n_calib += size
            else:
                test.extend(members[root])
                n_test += size

    # An indivisible component can be larger than a role's whole quota, and
    # then no deal can match the prior -- the component has to land entirely
    # on one side. That is not fixable here, so it is reported rather than
    # hidden: a calib fitted at a different prior than test silently
    # mis-sets every threshold downstream.
    def _fake_rate(idx: list[int]) -> float:
        if not idx:
            return float("nan")
        return sum(
            1 for i in idx if int(records[i].get("label", -1)) == FAKE_CLUE_FAKE_ID
        ) / len(idx)

    calib_rate, test_rate = _fake_rate(calib), _fake_rate(test)
    if abs(calib_rate - test_rate) > 0.10:
        logger.warning(
            f"FakeClue: calib and test priors differ "
            f"({calib_rate:.3f} vs {test_rate:.3f}). One identity group is too "
            f"large to divide, so the roles cannot be balanced without "
            f"breaking the firewall. Thresholds fitted on calib will not "
            f"transfer -- report both priors or raise calib_fraction."
        )

    logger.info(
        f"FakeClue: {len(records)} rows in {len(roots)} identity groups -> "
        f"calib {len(calib)} / test {len(test)} (requested calib "
        f"{calib_fraction:.2f}, realised {len(calib) / max(len(records), 1):.2f})"
    )
    return sorted(calib), sorted(test)


class FakeClueTorchDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Reads FakeClue images lazily out of the split's zip archive.

    The archive handle is opened per worker rather than in ``__init__``: a
    ``ZipFile`` carries a file position, so sharing one across dataloader
    workers interleaves seeks and returns corrupt bytes.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        zip_path: str | Path,
        split: str,
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.zip_path = str(zip_path)
        self.split = split
        self.transform = transform
        self._zip: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _archive(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def _read(self, archive: zipfile.ZipFile, image_path: str) -> bytes:
        # data_json stores the path without the split prefix that the archive
        # may or may not namespace entries under.
        for candidate in (image_path, f"{self.split}/{image_path}"):
            try:
                return archive.read(candidate)
            except KeyError:
                continue
        raise KeyError(
            f"{image_path!r} not found in {self.zip_path} "
            f"(tried with and without the {self.split!r} prefix)"
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = Image.open(
            io.BytesIO(self._read(self._archive(), record["image"]))
        ).convert("RGB")
        label = (
            PROJECT_REAL_ID
            if int(record["label"]) == FAKE_CLUE_REAL_ID
            else PROJECT_FAKE_ID
        )
        return self.transform(image), torch.tensor(label, dtype=torch.long)


class FakeClueDataModule(L.LightningDataModule):
    """FakeClue as a calib/test pair, identity-disjoint.

    Args:
        data_dir: Where ``test.zip`` and ``data_json/test.json`` live. Fetched
            from the Hub on first use when absent (``download=True``).
        split: FakeClue split to read ("test" or "train"). Only ``test`` is a
            benchmark; ``train`` exists to be folded into a combined training
            set and is not exposed as a train dataloader here.
        calib_fraction: Share of rows dealt to calib. Thresholds and
            temperature are fitted there and frozen before test is read, the
            same discipline SID-Set uses.
        group_split: Keep identities from crossing the calib/test boundary.
            Leave True. False reproduces the naive row split for comparison
            and must be disclosed in any report that uses it.
        manifest_seed: Seeds the group deal. A project constant, deliberately
            independent of ``experiment.seed``.
        input_mode: ``"resize"`` (resample to ``image_size``) or ``"crop"``
            (centre crop at native resolution, no resampling), matching
            :class:`~trustfake.data.sid_set.SIDSetDataModule`. Evaluate under
            the same mode the checkpoint was fitted under.
        limit_test: Cap the test role to its first N rows. A prefix, so the
            capped set is nested in the full one and an expensive attack can
            be reported beside a cheap one. calib is never capped.
    """

    HF_REPO = "lingcco/FakeClue"

    def __init__(
        self,
        data_dir: str | Path = "data/fake_clue",
        split: str = "test",
        calib_fraction: float = 0.3,
        group_split: bool = True,
        manifest_seed: int = DEFAULT_MANIFEST_SEED,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
        input_mode: str = "resize",
        normalization_layer: nn.Module | None = None,
        pin_memory: bool = True,
        limit_test: int | None = None,
        download: bool = True,
    ) -> None:
        super().__init__()
        if input_mode not in ("resize", "crop"):
            raise ValueError(
                f"input_mode must be 'resize' or 'crop', got {input_mode!r}"
            )
        if not 0.0 < calib_fraction < 1.0:
            raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.calib_fraction = calib_fraction
        self.group_split = group_split
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
        self.download = download
        self._calib_ds: FakeClueTorchDataset | None = None
        self._test_ds: FakeClueTorchDataset | None = None
        self.records: list[dict[str, Any]] = []

    # -- paths -------------------------------------------------------------

    @property
    def zip_path(self) -> Path:
        return self.data_dir / f"{self.split}.zip"

    @property
    def json_path(self) -> Path:
        return self.data_dir / "data_json" / f"{self.split}.json"

    def prepare_data(self) -> None:
        if self.zip_path.exists() and self.json_path.exists():
            return
        if not self.download:
            raise FileNotFoundError(
                f"{self.zip_path} or {self.json_path} missing and download=False"
            )
        from huggingface_hub import hf_hub_download

        for remote in (f"{self.split}.zip", f"data_json/{self.split}.json"):
            hf_hub_download(
                self.HF_REPO,
                remote,
                repo_type="dataset",
                local_dir=str(self.data_dir),
            )

    # -- transforms --------------------------------------------------------

    def _transform(self) -> transforms.Compose:
        if self.input_mode == "crop":
            steps: list[Any] = [
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        else:
            steps = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        return transforms.Compose(steps)

    # -- setup -------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        self.records = json.loads(self.json_path.read_text())
        if self.group_split:
            calib_idx, test_idx = assign_groups(
                self.records, self.calib_fraction, self.manifest_seed
            )
        else:
            generator = torch.Generator().manual_seed(self.manifest_seed)
            order = torch.randperm(len(self.records), generator=generator).tolist()
            cut = int(round(self.calib_fraction * len(self.records)))
            calib_idx, test_idx = sorted(order[:cut]), sorted(order[cut:])
            logger.warning(
                "FakeClue: group_split=False -- identities cross the calib/test "
                "boundary and the split measures memorisation as well as "
                "detection. Disclose this in any report."
            )
        if self.limit_test is not None:
            test_idx = test_idx[: self.limit_test]

        transform = self._transform()
        self._calib_ds = FakeClueTorchDataset(
            [self.records[i] for i in calib_idx], self.zip_path, self.split, transform
        )
        self._test_ds = FakeClueTorchDataset(
            [self.records[i] for i in test_idx], self.zip_path, self.split, transform
        )

    # -- loaders -----------------------------------------------------------

    def _loader(self, dataset: FakeClueTorchDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def calib_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._calib_ds is None:
            raise RuntimeError("Call setup() before requesting the calib dataloader.")
        return self._loader(self._calib_ds)

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before requesting the test dataloader.")
        return self._loader(self._test_ds)

    @property
    def calib_dataset(self) -> FakeClueTorchDataset:
        if self._calib_ds is None:
            raise RuntimeError("Call setup() first.")
        return self._calib_ds

    @property
    def test_dataset(self) -> FakeClueTorchDataset:
        if self._test_ds is None:
            raise RuntimeError("Call setup() first.")
        return self._test_ds
