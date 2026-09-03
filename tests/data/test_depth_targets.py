"""Depth targets: the store, the datamodule's opt-in, and the precompute.

The properties that matter: with the flag off nothing changes (2-tuples
everywhere); with it on, fit and val items carry the map of THEIR image,
resolved by the manifest's uid and not by row position; calib and test stay
2-tuples; and every mismatch -- a store computed on a different view of the
pixels, a fit row with no target, a crop-mode run -- is refused at setup.
No network, no GPU; the stub teacher stands in for Depth Anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from trustfake.data import SIDSetDataModule
from trustfake.data.depth_targets import (
    STORE_MANIFEST,
    DepthTargetStore,
    check_store_manifest,
    depth_target_paths,
    precompute_depth_targets,
    read_store_manifest,
    write_store_manifest,
)
from trustfake.depth import FakeDepthTeacher

ROWS = 10
SIZE = 8  # depth grid
IMAGE_SIZE = 16
REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "src" / "precompute_depth.py"


def _write_shard(path, img_ids, labels):
    rng = np.random.default_rng(len(img_ids))
    images = [rng.integers(0, 255, (8, 8, 3), dtype=np.uint8).tolist() for _ in img_ids]
    table = pa.table(
        {
            "img_id": pa.array(img_ids, pa.string()),
            "label": pa.array(labels, pa.int64()),
            "image": pa.array(images),
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def sid_set_dir(tmp_path):
    data = tmp_path / "sid_set"
    data.mkdir()
    for i in range(3):
        _write_shard(
            data / f"train-{i:05d}-of-00003.parquet",
            [f"img_{i * ROWS + j:04d}" for j in range(ROWS)],
            [j % 3 for j in range(ROWS)],
        )
    for i in range(3):
        _write_shard(
            data / f"validation-{i:05d}-of-00003.parquet",
            [f"img_{i * ROWS + j:04d}" for j in range(ROWS)],
            [j % 3 for j in range(ROWS)],
        )
    return data


def _meta(**over):
    meta = {
        "teacher": "fake",
        "revision": None,
        "input_size": IMAGE_SIZE,
        "multiple": 1,
        "keep_aspect_ratio": True,
        "output_size": SIZE,
        "frame": "median_mad",
        "image_size": IMAGE_SIZE,
        "squarecrop": False,
        "input_mode": "resize",
        "dtype": "float16",
    }
    meta.update(over)
    return meta


def _write_store(data_dir, depth_dir, shards=("train-00000", "train-00001"), **over):
    """Hand-written store whose map for a row is a constant equal to that
    row's number, so a lookup error is visible in one pixel."""
    write_store_manifest(depth_dir, _meta(**over))
    for stem in shards:
        shard = data_dir / f"{stem}-of-00003.parquet"
        ids = pq.read_table(shard, columns=["img_id"]).column("img_id").to_pylist()
        npy, ids_path = depth_target_paths(depth_dir, shard)
        maps = np.stack(
            [np.full((SIZE, SIZE), float(i.split("_")[1]), np.float16) for i in ids]
        )
        np.save(npy, maps)
        ids_path.write_text(json.dumps(ids))
    return depth_dir


def _dm(data_dir, **kw):
    defaults = {
        "data_dir": data_dir,
        "profile": "smoke",
        "val_fraction": 0.2,
        "batch_size": 4,
        "num_workers": 0,
        "image_size": IMAGE_SIZE,
        "seed": 1,
        "pin_memory": False,
    }
    defaults.update(kw)
    return SIDSetDataModule(**defaults)


# --------------------------------------------------------------------------
# The datamodule opt-in
# --------------------------------------------------------------------------


def test_default_leaves_every_item_a_pair(sid_set_dir):
    dm = _dm(sid_set_dir)
    dm.setup()
    assert not dm.has_depth_targets
    for ds in (dm.train_dataset, dm.val_dataset, dm.calib_dataset, dm.test_dataset):
        assert len(ds[0]) == 2
    assert len(next(iter(dm.train_dataloader()))) == 2


def test_fit_items_carry_their_own_map_and_eval_items_do_not(sid_set_dir, tmp_path):
    store = _write_store(sid_set_dir, tmp_path / "depth")
    dm = _dm(sid_set_dir, depth_targets_dir=store)
    dm.setup()
    assert dm.has_depth_targets

    for ds in (dm.train_dataset, dm.val_dataset):
        assert ds.has_depth_targets
        for index in range(len(ds)):
            image, label, depth = ds[index]
            assert depth.shape == (1, SIZE, SIZE) and depth.dtype == torch.float32
            img_id = ds.hf_dataset[index]["img_id"]
            # the map is the one for THIS row's identity, not its position
            assert float(depth[0, 0, 0]) == float(img_id.split("_")[1])
    for ds in (dm.calib_dataset, dm.test_dataset):
        assert not ds.has_depth_targets
        assert len(ds[0]) == 2

    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == 3
    assert batch[2].shape == (4, 1, SIZE, SIZE)
    assert len(next(iter(dm.calib_dataloader()))) == 2
    assert len(next(iter(dm.test_dataloader()))) == 2


def test_crop_mode_is_refused_with_depth_targets(sid_set_dir, tmp_path):
    with pytest.raises(ValueError, match="input_mode='crop'"):
        _dm(sid_set_dir, depth_targets_dir=tmp_path / "depth", input_mode="crop")


def test_store_computed_under_other_settings_is_refused(sid_set_dir, tmp_path):
    store = _write_store(sid_set_dir, tmp_path / "depth", image_size=32)
    dm = _dm(sid_set_dir, depth_targets_dir=store)
    with pytest.raises(ValueError, match="different settings"):
        dm.setup()

    store = _write_store(sid_set_dir, tmp_path / "depth_sq", squarecrop=True)
    with pytest.raises(ValueError, match="squarecrop"):
        _dm(sid_set_dir, depth_targets_dir=store).setup()


def test_missing_shard_and_missing_row_are_refused_at_setup(sid_set_dir, tmp_path):
    store = _write_store(sid_set_dir, tmp_path / "one_shard", shards=("train-00000",))
    with pytest.raises(FileNotFoundError, match="train-00001"):
        _dm(sid_set_dir, depth_targets_dir=store).setup()

    store = _write_store(sid_set_dir, tmp_path / "one_row_short")
    _, ids_path = depth_target_paths(
        store, sid_set_dir / "train-00001-of-00003.parquet"
    )
    ids = json.loads(ids_path.read_text())
    ids_path.write_text(json.dumps(ids[:-1]))
    with pytest.raises(ValueError, match="no depth target"):
        _dm(sid_set_dir, depth_targets_dir=store).setup()


def test_store_at_another_grid_is_refused(sid_set_dir, tmp_path):
    """The head emits image_size // 2 and the loss does not resample."""
    store = _write_store(sid_set_dir, tmp_path / "depth", output_size=SIZE + 2)
    with pytest.raises(ValueError, match="output_size"):
        _dm(sid_set_dir, depth_targets_dir=store).setup()


def test_a_stochastic_train_transform_is_refused(sid_set_dir, tmp_path):
    """No augmentation exists today, which is exactly what makes a per-image
    precomputed target valid; this pins that a future Random* stage cannot
    silently misalign the targets."""
    from torchvision import transforms

    store = _write_store(sid_set_dir, tmp_path / "depth")
    dm = _dm(sid_set_dir, depth_targets_dir=store)
    dm.train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), *dm.train_transform.transforms]
    )
    with pytest.raises(ValueError, match="stochastic train transform"):
        dm.setup()


def test_limit_shards_caps_a_run_and_the_rest_is_resumable(sid_set_dir, tmp_path):
    depth = tmp_path / "depth"
    kw = {"profile": "smoke", "image_size": IMAGE_SIZE, "batch_size": 4}
    first = precompute_depth_targets(
        sid_set_dir, depth, _teacher(), limit_shards=1, **kw
    )
    assert first["written"] == 1
    rest = precompute_depth_targets(sid_set_dir, depth, _teacher(), **kw)
    assert rest["written"] == 1 and rest["skipped"] == 1


def test_non_finite_teacher_output_is_refused(sid_set_dir, tmp_path):
    class _NaNTeacher(FakeDepthTeacher):
        def forward(self, x):
            out = super().forward(x)
            out[0, 0, 0, 0] = float("nan")
            return out

    teacher = _NaNTeacher(output_size=SIZE, input_size=IMAGE_SIZE, multiple=1)
    with pytest.raises(ValueError, match="non-finite"):
        precompute_depth_targets(
            sid_set_dir,
            tmp_path / "depth",
            teacher,
            profile="smoke",
            image_size=IMAGE_SIZE,
            batch_size=4,
        )


def _exif_rotated_jpeg(width=12, height=8) -> bytes:
    """A JPEG whose EXIF says 'rotate 90': HF datasets decodes it transposed."""
    import io

    from PIL import Image

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    exif = image.getexif()
    exif[Image.ExifTags.Base.Orientation] = 6
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_precompute_decodes_exactly_like_the_datamodule_including_exif(tmp_path):
    """HF datasets applies exif_transpose on decode; a precompute that reads
    the same bytes with a bare Image.open would pair an EXIF-rotated photo's
    target with a differently oriented image, and the loss would keep
    decreasing. Both sides decode through one helper; this pins it on an
    HF-written shard with an orientation tag."""
    from datasets import Dataset, Features, Value
    from datasets import Image as HFImage

    from trustfake.data._images import decode_image_cell
    from trustfake.data.depth_targets import _pre_transform

    data = tmp_path / "sid_set"
    data.mkdir()
    features = Features(
        {"img_id": Value("string"), "label": Value("int64"), "image": HFImage()}
    )

    def rows(shard_index):
        return {
            "img_id": [f"img_{shard_index * ROWS + j:04d}" for j in range(ROWS)],
            "label": [j % 3 for j in range(ROWS)],
            "image": [
                {"bytes": _exif_rotated_jpeg(), "path": None} for _ in range(ROWS)
            ],
        }

    for i in range(3):
        Dataset.from_dict(rows(i), features=features).to_parquet(
            str(data / f"train-{i:05d}-of-00003.parquet")
        )
        Dataset.from_dict(rows(i), features=features).to_parquet(
            str(data / f"validation-{i:05d}-of-00003.parquet")
        )

    depth = tmp_path / "depth"
    precompute_depth_targets(
        data, depth, _teacher(), profile="smoke", image_size=IMAGE_SIZE, batch_size=4
    )
    dm = _dm(data, depth_targets_dir=depth)
    dm.setup()
    image, _, target = dm.train_dataset[0]
    # the datamodule saw the EXIF-transposed image (portrait after rotation)
    raw = pq.read_table(data / "train-00000-of-00003.parquet", columns=["image"])
    cell = raw.column("image").to_pylist()[0]
    pil = decode_image_cell(cell)
    assert pil.size == (8, 12), "EXIF orientation was not applied"
    assert torch.equal(image, _pre_transform(IMAGE_SIZE, False)(pil))
    # ... and its target is the teacher's map of THAT image
    assert torch.allclose(target, _teacher()(image.unsqueeze(0))[0], atol=2e-2)


def test_a_store_without_manifest_is_not_a_store(sid_set_dir, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=STORE_MANIFEST):
        _dm(sid_set_dir, depth_targets_dir=tmp_path / "empty").setup()


def test_shards_without_an_id_column_are_refused(tmp_path):
    data = tmp_path / "noid"
    data.mkdir()
    table = pa.table(
        {
            "label": pa.array([j % 3 for j in range(ROWS)], pa.int64()),
            "image": pa.array([np.zeros((8, 8, 3), np.uint8).tolist()] * ROWS),
        }
    )
    for i in range(3):
        pq.write_table(table, data / f"train-{i:05d}-of-00003.parquet")
        pq.write_table(table, data / f"validation-{i:05d}-of-00003.parquet")
    depth = tmp_path / "depth"
    write_store_manifest(depth, _meta())
    with pytest.raises(ValueError, match="id column"):
        _dm(data, depth_targets_dir=depth).setup()


def test_store_survives_a_pickle_round_trip(sid_set_dir, tmp_path):
    """Dataloader workers receive the dataset by pickle; memmap handles must
    be reopened, not carried."""
    import pickle

    store = DepthTargetStore(
        _write_store(sid_set_dir, tmp_path / "depth"),
        [sid_set_dir / "train-00000-of-00003.parquet"],
        "train",
    )
    _ = store["train:img_0003"]
    clone = pickle.loads(pickle.dumps(store))
    assert torch.equal(clone["train:img_0003"], store["train:img_0003"])
    with pytest.raises(KeyError, match="train:nope"):
        _ = store["train:nope"]


def test_manifest_check_names_the_mismatch():
    with pytest.raises(ValueError, match="image_size: store has 16, run needs 32"):
        check_store_manifest(_meta(), image_size=32)
    check_store_manifest(_meta(), image_size=IMAGE_SIZE, squarecrop=False)


# --------------------------------------------------------------------------
# The precompute
# --------------------------------------------------------------------------


def _teacher():
    return FakeDepthTeacher(output_size=SIZE, input_size=IMAGE_SIZE, multiple=1)


def test_precompute_writes_a_store_the_datamodule_accepts(sid_set_dir, tmp_path):
    depth = tmp_path / "depth"
    summary = precompute_depth_targets(
        sid_set_dir,
        depth,
        _teacher(),
        profile="smoke",
        image_size=IMAGE_SIZE,
        batch_size=4,
    )
    assert summary["written"] == 2 and summary["rows"] == 2 * ROWS
    meta = read_store_manifest(depth)
    assert meta["teacher"] == "fake_luminance_blur" and meta["frame"] == "median_mad"
    assert meta["image_size"] == IMAGE_SIZE and meta["input_mode"] == "resize"

    npy, ids_path = depth_target_paths(
        depth, sid_set_dir / "train-00000-of-00003.parquet"
    )
    array = np.load(npy)
    assert array.shape == (ROWS, SIZE, SIZE) and array.dtype == np.float16
    assert json.loads(ids_path.read_text()) == [f"img_{j:04d}" for j in range(ROWS)]
    # stored in the shared frame: LOWER median 0, mean absolute deviation 1
    flat = array.astype(np.float32).reshape(ROWS, -1)
    lower_median = np.sort(flat, axis=1)[:, (flat.shape[1] - 1) // 2]
    assert np.allclose(lower_median, 0.0, atol=1e-2)
    assert np.allclose(np.abs(flat).mean(axis=1), 1.0, atol=1e-2)
    assert np.isfinite(flat).all()

    dm = _dm(sid_set_dir, depth_targets_dir=depth)
    dm.setup()
    image, label, d = dm.train_dataset[0]
    assert d.shape == (1, SIZE, SIZE)
    # the map is the teacher's map of the same pixels
    expected = _teacher()(image.unsqueeze(0))[0]
    assert torch.allclose(d, expected, atol=2e-2)


def test_precompute_is_resumable_and_refuses_a_mismatched_store(sid_set_dir, tmp_path):
    depth = tmp_path / "depth"
    kw = {"profile": "smoke", "image_size": IMAGE_SIZE, "batch_size": 4}
    precompute_depth_targets(sid_set_dir, depth, _teacher(), **kw)
    again = precompute_depth_targets(sid_set_dir, depth, _teacher(), **kw)
    assert again["written"] == 0 and again["skipped"] == 2
    with pytest.raises(ValueError, match="different settings"):
        precompute_depth_targets(
            sid_set_dir, depth, _teacher(), profile="smoke", image_size=32
        )
    with pytest.raises(ValueError, match="not in profile"):
        precompute_depth_targets(
            sid_set_dir, depth, _teacher(), roles=("holdout",), **kw
        )


def test_cli_smoke(sid_set_dir, tmp_path):
    out = tmp_path / "cli_depth"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--data-dir",
            str(sid_set_dir),
            "--out-dir",
            str(out),
            "--profile",
            "smoke",
            "--stub-teacher",
            "--image-size",
            str(IMAGE_SIZE),
            "--size",
            str(SIZE),
            "--teacher-input-size",
            str(IMAGE_SIZE),
            "--batch",
            "4",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    marker = "SUMMARY_JSON\n"
    summary = json.loads(proc.stdout[proc.stdout.rindex(marker) + len(marker) :])
    assert summary["written"] == 2
    assert (out / STORE_MANIFEST).exists()

    # --size must be the head's grid; a limit-shards run is resumable
    bad = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--data-dir",
            str(sid_set_dir),
            "--out-dir",
            str(out),
            "--profile",
            "smoke",
            "--stub-teacher",
            "--image-size",
            str(IMAGE_SIZE),
            "--size",
            "4",
            "--teacher-input-size",
            str(IMAGE_SIZE),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert bad.returncode != 0 and "image-size // 2" in bad.stderr
