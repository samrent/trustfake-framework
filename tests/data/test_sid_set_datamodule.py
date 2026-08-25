"""End-to-end datamodule test on synthetic parquet shards.

Builds a miniature SID_Set on disk -- same shard naming, same columns,
including the img_id-collides-across-source-splits property -- and runs the
datamodule's whole setup/dataloader path against it. No network, no GPU.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from trustfake.data import SIDSetDataModule

ROWS_PER_SHARD = 10
IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def _write_shard(path, img_ids, labels):
    table = pa.table(
        {
            "img_id": pa.array(img_ids, pa.string()),
            "label": pa.array(labels, pa.int64()),
            "image": pa.array([IMG.tolist()] * len(img_ids)),
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def sid_set_dir(tmp_path):
    """6 train + 4 validation shards. img_ids deliberately repeat across the
    two source splits (the real dataset's collision property) but are unique
    within each split."""
    for i in range(6):
        _write_shard(
            tmp_path / f"train-{i:05d}-of-00006.parquet",
            [
                f"full_synthetic_{i * ROWS_PER_SHARD + j:06d}"
                for j in range(ROWS_PER_SHARD)
            ],
            [j % 3 for j in range(ROWS_PER_SHARD)],
        )
    for i in range(4):
        _write_shard(
            tmp_path / f"validation-{i:05d}-of-00004.parquet",
            # restart the counter: these img_ids collide with train's
            [
                f"full_synthetic_{i * ROWS_PER_SHARD + j:06d}"
                for j in range(ROWS_PER_SHARD)
            ],
            [j % 3 for j in range(ROWS_PER_SHARD)],
        )
    return tmp_path


def _datamodule(data_dir, **kwargs):
    defaults = {
        "data_dir": data_dir,
        "profile": "smoke",
        "val_fraction": 0.2,
        "batch_size": 4,
        "num_workers": 0,
        "image_size": 16,
        "seed": 1,
        "pin_memory": False,
    }
    defaults.update(kwargs)
    return SIDSetDataModule(**defaults)


def test_setup_and_loaders(sid_set_dir):
    dm = _datamodule(sid_set_dir)
    dm.setup()

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 3, 16, 16)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert y.dtype == torch.long

    # smoke profile: fit=2, calib=1, test=2 shards
    n_fit = len(dm.train_dataset) + len(dm.val_dataset)
    assert n_fit == 2 * ROWS_PER_SHARD
    assert len(dm.calib_dataset) == 1 * ROWS_PER_SHARD
    assert len(dm.test_dataset) == 2 * ROWS_PER_SHARD
    assert len(dm.val_dataset) == int(round(n_fit * 0.2))


def test_selection_slice_comes_from_fit_not_from_test(sid_set_dir):
    """Defect 1's regression test: the rows Lightning selects on (val) and the
    rows reported on (test) must be disjoint by construction. img_ids repeat
    across source splits, so compare uids."""
    dm = _datamodule(sid_set_dir)
    dm.setup()

    def uids(wrapped, source):
        return {f"{source}:{r}" for r in wrapped.hf_dataset["img_id"]}

    val_uids = uids(dm.val_dataset, "train")
    test_uids = uids(dm.test_dataset, "validation")
    calib_uids = uids(dm.calib_dataset, "validation")
    assert not (val_uids & test_uids)
    assert not (calib_uids & test_uids)
    # and val really is a slice of fit
    assert val_uids <= uids(dm.train_dataset, "train") | val_uids


def test_val_carve_is_seeded(sid_set_dir):
    a = _datamodule(sid_set_dir, seed=1)
    b = _datamodule(sid_set_dir, seed=1)
    c = _datamodule(sid_set_dir, seed=2)
    for dm in (a, b, c):
        dm.setup()
    ids = lambda dm: list(dm.val_dataset.hf_dataset["img_id"])  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_duplicate_img_id_across_calib_and_test_is_fatal(tmp_path):
    """The uid tripwire: a duplicated validation img_id spanning calib and
    test shards must abort setup, not silently merge rows."""
    for i in range(2):
        _write_shard(
            tmp_path / f"train-{i:05d}-of-00002.parquet",
            [f"real_{i}_{j}" for j in range(ROWS_PER_SHARD)],
            [0] * ROWS_PER_SHARD,
        )
    # Learn which shard the manifest sends to calib, then plant one img_id
    # shared between the calib shard and a test shard -- and nowhere else, so
    # only the cross-role check can catch it.
    from trustfake.data.manifest import PROFILES, assign_shards

    val_names = [f"validation-{i:05d}-of-00003.parquet" for i in range(3)]
    roles = assign_shards(
        [f"train-{i:05d}-of-00002.parquet" for i in range(2)],
        val_names,
        PROFILES["smoke"],
        seed=0,
    )
    calib_shard, leak_test_shard = roles["calib"][0], roles["test"][0]
    for i, name in enumerate(val_names):
        ids = [f"val_{i}_{j}" for j in range(ROWS_PER_SHARD)]
        if name in (calib_shard, leak_test_shard):
            ids[0] = "leaked_across_roles"
        _write_shard(tmp_path / name, ids, [1] * ROWS_PER_SHARD)
    dm = _datamodule(tmp_path)
    with pytest.raises(ValueError, match="calib and test"):
        dm.setup()


def test_missing_shards_raise(tmp_path):
    dm = _datamodule(tmp_path)
    with pytest.raises(FileNotFoundError, match="No SID_Set parquet shards"):
        dm.setup()
