"""So-Fake-OOD: the shift condition.

Two things are pinned. The label mapping, because getting it wrong silently
relabels the two forgery modalities against each other -- and unlike FakeClue
this dataset *can* express them, which is the reason to prefer it. And the
refusal to serve a calib split, because refitting a threshold on shifted data
hides the exchangeability violation the condition exists to measure.

No network, no parquet: shard selection and label mapping are pure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import transforms

from trustfake.data.so_fake_ood import (
    STRING_LABEL_TO_ID,
    SoFakeOODDataModule,
    SoFakeOODTorchDataset,
    select_shards,
)


class _FakeImage:
    def convert(self, _mode):
        from PIL import Image

        return Image.new("RGB", (16, 16), (10, 200, 10))


class _FakeHFDataset:
    def __init__(self, labels):
        self.rows = [{"image": _FakeImage(), "label": lab} for lab in labels]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# --------------------------------------------------------------------------
# Label mapping -- the reason this dataset beats FakeClue
# --------------------------------------------------------------------------


def test_string_labels_map_onto_sid_set_ids():
    """REAL/FULL_SYNTHETIC/TAMPERED -> 0/1/2, SID-Set's own convention, so a
    SID-Set checkpoint is scored without remapping AND the per-modality
    breakout survives. FakeClue cannot do this: it is binary."""
    assert STRING_LABEL_TO_ID == {"REAL": 0, "FULL_SYNTHETIC": 1, "TAMPERED": 2}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("REAL", 0), ("FULL_SYNTHETIC", 1), ("TAMPERED", 2), ("tampered", 2)],
)
def test_labels_are_mapped_case_insensitively(raw, expected):
    ds = SoFakeOODTorchDataset(_FakeHFDataset([raw]), transforms.ToTensor())
    _, label = ds[0]
    assert int(label) == expected


def test_unknown_label_raises_rather_than_guessing():
    """A label this mapping has never seen must stop the run. Silently
    folding it into a known class would relabel the dataset."""
    ds = SoFakeOODTorchDataset(
        _FakeHFDataset(["PARTIALLY_SYNTHETIC"]), transforms.ToTensor()
    )
    with pytest.raises(KeyError, match="unknown So-Fake-OOD label"):
        ds[0]


def test_integer_labels_pass_through():
    ds = SoFakeOODTorchDataset(_FakeHFDataset([2]), transforms.ToTensor())
    _, label = ds[0]
    assert int(label) == 2


def test_returns_two_tuples():
    ds = SoFakeOODTorchDataset(_FakeHFDataset(["REAL"]), transforms.ToTensor())
    item = ds[0]
    assert len(item) == 2
    assert isinstance(item[0], torch.Tensor) and item[0].shape[0] == 3


# --------------------------------------------------------------------------
# The refusal to calibrate on shifted data
# --------------------------------------------------------------------------


def test_calib_dataloader_is_refused():
    """Not an oversight. A threshold refitted after the shift papers over the
    exchangeability break that makes this condition worth running."""
    dm = SoFakeOODDataModule()
    with pytest.raises(NotImplementedError, match="(?i)in-domain calib"):
        dm.calib_dataloader()


# --------------------------------------------------------------------------
# Shard selection
# --------------------------------------------------------------------------


def test_shard_choice_is_deterministic_and_seeded():
    shards = [Path(f"test_image-{i:05d}-of-00046.parquet") for i in range(46)]
    a = select_shards(shards, 4, seed=0)
    b = select_shards(shards, 4, seed=0)
    c = select_shards(shards, 4, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 4


def test_selection_is_independent_of_filesystem_order():
    """Sorted first, so the sample depends on shard identity rather than on
    whatever order rglob happened to return."""
    shards = [Path(f"s-{i}.parquet") for i in range(20)]
    assert select_shards(shards, 5, 0) == select_shards(list(reversed(shards)), 5, 0)


def test_none_takes_every_shard_present():
    shards = [Path(f"s-{i}.parquet") for i in range(7)]
    assert select_shards(shards, None, 0) == sorted(shards)
    assert select_shards(shards, 99, 0) == sorted(shards)


def test_no_shards_is_an_error():
    with pytest.raises(FileNotFoundError, match="no So-Fake-OOD"):
        select_shards([], None, 0)


def test_zero_shards_is_refused():
    with pytest.raises(ValueError, match="n_shards"):
        select_shards([Path("a.parquet"), Path("b.parquet")], 0, 0)


def test_rejects_unknown_input_mode():
    with pytest.raises(ValueError, match="input_mode"):
        SoFakeOODDataModule(input_mode="squarecrop")


def test_dataloader_before_setup_is_an_error():
    with pytest.raises(RuntimeError, match="setup"):
        SoFakeOODDataModule().test_dataloader()
