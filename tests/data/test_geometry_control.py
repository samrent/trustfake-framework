"""Tests for the two geometry controls: the row filter and the squarecrop
pre-transform.

Both answer the same artifact -- SID-Set's fake classes are square and most
reals are not, so ``width == height -> fake`` scores above a CLIP probe -- so
both are tested on the property that makes them a control rather than a
relabelling: after the control, geometry carries no label information. A
version that merely reduced the correlation would pass a "did it change
anything" test and fail the science.
"""

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from trustfake.data import SIDSetDataModule
from trustfake.data.baselines import image_dims_and_format
from trustfake.data.manifest import GEOMETRY_FILTERS, geometry_selection
from trustfake.data.sid_set import CentreSquareCrop, _original_dims

ROWS_PER_SHARD = 12


def _png_bytes(width: int, height: int, value: int = 0) -> bytes:
    buffer = io.BytesIO()
    array = np.full((height, width, 3), value, np.uint8)
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_shard(path, rows) -> None:
    """rows: list of (img_id, label, image_bytes)."""
    pq.write_table(
        pa.table(
            {
                "img_id": pa.array([r[0] for r in rows], pa.string()),
                "label": pa.array([r[1] for r in rows], pa.int64()),
                "image": pa.array([{"bytes": r[2], "path": None} for r in rows]),
            }
        ),
        path,
    )


@pytest.fixture
def sid_set_dir(tmp_path):
    """The artifact in miniature, but not absolutely: every class holds both
    geometries, so `matched` has something to match rather than degenerating
    immediately. Label 0 (real) is mostly non-square, labels 1 and 2 mostly
    square -- the SID-Set correlation, softened."""

    def rows(prefix):
        out = []
        for j in range(ROWS_PER_SHARD):
            label = j % 3
            # reals square 1 time in 4; fakes non-square 1 time in 4.
            square = (j % 4 == 0) if label == 0 else (j % 4 != 0)
            size = (64, 64) if square else (64, 48)
            out.append((f"{prefix}_{j}", label, _png_bytes(*size)))
        return out

    for i in range(2):
        _write_shard(tmp_path / f"train-{i:05d}-of-00002.parquet", rows(f"t{i}"))
    for i in range(3):
        _write_shard(tmp_path / f"validation-{i:05d}-of-00003.parquet", rows(f"v{i}"))
    return tmp_path


def _datamodule(data_dir, **kwargs):
    defaults = {
        "data_dir": data_dir,
        "profile": "smoke",
        "val_fraction": 0.25,
        "batch_size": 4,
        "num_workers": 0,
        "image_size": 16,
        "seed": 1,
        "pin_memory": False,
    }
    defaults.update(kwargs)
    return SIDSetDataModule(**defaults)


def _square_rate_by_label(dataset) -> dict[int, float]:
    """Square rate per label of a wrapped HF dataset, read from the ORIGINAL
    bytes -- the same place the artifact lives."""
    rates: dict[int, list[int]] = {}
    hf = dataset.hf_dataset
    for index in range(len(hf)):
        row = hf[index]
        width, height, _ = image_dims_and_format(row["image"])
        rates.setdefault(int(row["label"]), []).append(int(width == height))
    return {label: float(np.mean(flags)) for label, flags in rates.items()}


# --------------------------------------------------------------------------
# (a) the row filter
# --------------------------------------------------------------------------


def test_selection_modes_pick_the_right_rows():
    widths = np.array([10, 10, 20, 30])
    heights = np.array([10, 20, 20, 40])
    labels = np.array([0, 0, 1, 1])

    assert geometry_selection(widths, heights, labels, "none") == [0, 1, 2, 3]
    assert geometry_selection(widths, heights, labels, "square") == [0, 2]
    assert geometry_selection(widths, heights, labels, "nonsquare") == [1, 3]


def test_matched_equalises_the_square_rate_across_classes():
    """THE property. Labels start with wildly different square rates (1.0 vs
    0.1); after `matched` they must be identical, so `width == height`
    carries exactly zero information about the label."""
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 100)
    square = np.concatenate([rng.random(100) < 0.1, rng.random(100) < 0.9])
    widths = np.full(200, 64)
    heights = np.where(square, 64, 48)
    assert square[labels == 0].mean() != square[labels == 1].mean()

    keep = geometry_selection(widths, heights, labels, "matched", seed=0)
    kept_square = (widths == heights)[keep]
    kept_labels = labels[keep]
    rates = {
        label: kept_square[kept_labels == label].mean()
        for label in np.unique(kept_labels)
    }

    assert len(set(rates.values())) == 1
    assert set(rates) == {0, 1}
    # Both geometries survive, so this is a genuine match rather than the
    # degenerate all-square case tested below.
    assert 0.0 < next(iter(rates.values())) < 1.0
    # ... and it matched by subsampling, not by emptying a class.
    assert all((kept_labels == label).sum() > 0 for label in (0, 1))


def test_matched_also_equalises_the_class_prior():
    """A consequence worth asserting because it must be reported: the
    majority-class floor moves with it."""
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], [300, 100, 100])
    square = np.concatenate(
        [rng.random(300) < 0.2, rng.random(100) < 0.9, rng.random(100) < 0.8]
    )
    widths = np.full(500, 64)
    heights = np.where(square, 64, 48)

    keep = geometry_selection(widths, heights, labels, "matched", seed=0)
    counts = np.bincount(labels[keep])

    assert len(set(counts.tolist())) == 1


def test_matched_degenerates_to_square_only_when_a_class_is_all_square():
    """SID-Set's actual shape: a fake class with no non-square row at all.
    Any equal-rate subset containing it must then be square-only, and the
    sampler must say so by construction rather than quietly emitting an
    unequal subset."""
    labels = np.repeat([0, 1], 100)
    square = np.concatenate([np.arange(100) < 20, np.ones(100, bool)])
    widths = np.full(200, 64)
    heights = np.where(square, 64, 48)

    keep = geometry_selection(widths, heights, labels, "matched", seed=0)

    assert (widths == heights)[keep].all()
    assert np.bincount(labels[keep]).tolist() == [20, 20]


def test_matched_selection_is_seeded_by_the_manifest_seed():
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 200)
    square = np.concatenate([rng.random(200) < 0.5, rng.random(200) < 0.9])
    widths = np.full(400, 64)
    heights = np.where(square, 64, 48)

    args = (widths, heights, labels, "matched")
    assert geometry_selection(*args, seed=0) == geometry_selection(*args, seed=0)
    assert geometry_selection(*args, seed=0) != geometry_selection(*args, seed=1)


def test_selection_rejects_an_unknown_mode_and_an_empty_result():
    widths = np.array([10, 10])
    heights = np.array([10, 10])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="Unknown geometry_filter"):
        geometry_selection(widths, heights, labels, "squareish")
    # no non-square row exists -> there is no controlled subset to report on
    with pytest.raises(ValueError, match="selected 0 of 2 rows"):
        geometry_selection(widths, heights, labels, "nonsquare")


def test_selection_rejects_ragged_inputs():
    with pytest.raises(ValueError, match="parallel"):
        geometry_selection(np.zeros(3), np.zeros(3), np.zeros(2), "square")


def test_original_dims_reads_an_hf_image_column_without_decoding_it():
    """The real dataset carries an HF Image feature, so the filter would
    otherwise decode every megapixel just to measure it. `_original_dims`
    casts to decode=False and reads the header -- and must still return the
    ORIGINAL dimensions, not the ones the datamodule will resize to."""
    from datasets import Dataset
    from datasets import Image as HFImage

    sizes = [(64, 48), (64, 64), (32, 96)]
    hf = Dataset.from_dict(
        {"image": [_png_bytes(w, h) for w, h in sizes], "label": [0, 1, 2]}
    ).cast_column("image", HFImage())
    assert isinstance(hf.features["image"], HFImage)

    widths, heights = _original_dims(hf, "image")

    assert widths.tolist() == [w for w, _ in sizes]
    assert heights.tolist() == [h for _, h in sizes]


def test_datamodule_filter_is_off_by_default(sid_set_dir):
    dm = _datamodule(sid_set_dir)
    dm.setup()
    assert dm.geometry_filter == "none"
    assert len(dm.test_dataset) == 2 * ROWS_PER_SHARD


@pytest.mark.parametrize("mode", ["square", "nonsquare", "matched"])
def test_datamodule_filter_controls_the_reported_splits(sid_set_dir, mode):
    """calib and test are filtered together -- thresholds fitted on calib are
    reported on test, so they must describe the same distribution -- while
    fit is left alone: the control asks what a model trained on the real
    distribution does when geometry stops carrying signal."""
    baseline = _datamodule(sid_set_dir)
    baseline.setup()
    dm = _datamodule(sid_set_dir, geometry_filter=mode)
    dm.setup()

    assert len(dm.test_dataset) < len(baseline.test_dataset)
    assert len(dm.calib_dataset) < len(baseline.calib_dataset)
    n_fit = len(dm.train_dataset) + len(dm.val_dataset)
    n_fit_baseline = len(baseline.train_dataset) + len(baseline.val_dataset)
    assert n_fit == n_fit_baseline


def test_datamodule_matched_filter_kills_the_signal_in_the_test_split(sid_set_dir):
    """End to end, on parquet: the artifact is present in the raw test split
    and absent after the filter."""
    raw = _datamodule(sid_set_dir)
    raw.setup()
    raw_rates = _square_rate_by_label(raw.test_dataset)
    assert len(set(raw_rates.values())) > 1  # the artifact is there to remove

    dm = _datamodule(sid_set_dir, geometry_filter="matched")
    dm.setup()
    rates = _square_rate_by_label(dm.test_dataset)

    assert len(set(rates.values())) == 1


def test_datamodule_rejects_an_unknown_filter(sid_set_dir):
    with pytest.raises(ValueError, match="Unknown geometry_filter"):
        _datamodule(sid_set_dir, geometry_filter="squarish")


def test_every_documented_filter_mode_is_accepted(sid_set_dir):
    for mode in GEOMETRY_FILTERS:
        assert _datamodule(sid_set_dir, geometry_filter=mode).geometry_filter == mode


# --------------------------------------------------------------------------
# (b) the squarecrop pre-transform
# --------------------------------------------------------------------------


def test_squarecrop_destroys_the_width_equals_height_signal():
    """WP1's control test (`wp1/tests/test_attacks.py:201`): whatever goes in,
    a square of the short side comes out -- so `width == height` is constant
    and carries zero label information by construction."""
    for width, height in ((1024, 1024), (1024, 681), (640, 960), (7, 3)):
        cropped = CentreSquareCrop()(Image.new("RGB", (width, height)))
        assert cropped.size == (min(width, height), min(width, height))


def test_squarecrop_takes_the_centre_not_a_corner():
    array = np.zeros((16, 64, 3), np.uint8)
    array[:, :8] = 255  # a bright band on the far left only
    cropped = np.asarray(CentreSquareCrop()(Image.fromarray(array)))

    assert cropped.shape == (16, 16, 3)
    assert cropped.max() == 0


def test_squarecrop_runs_before_the_resize(sid_set_dir):
    """Load-bearing order. Behind the resize every image is already 224x224,
    so the crop would be a silent no-op and the 'control' would control
    nothing."""
    dm = _datamodule(sid_set_dir, squarecrop=True)
    stages = dm.transform.transforms

    assert isinstance(stages[0], CentreSquareCrop)
    assert isinstance(stages[1], torch.nn.Module) or "Resize" in repr(stages[1])
    assert "Resize" in repr(stages[1])
    assert isinstance(_datamodule(sid_set_dir).transform.transforms[0], type(stages[1]))


def test_squarecrop_changes_what_the_model_sees(sid_set_dir):
    """The functional half of the order test: with the crop in front, the
    content outside the centre square never reaches the tensor."""
    array = np.zeros((16, 64, 3), np.uint8)
    array[:, :8] = 255
    image = Image.fromarray(array)

    plain = _datamodule(sid_set_dir).transform(image)
    cropped = _datamodule(sid_set_dir, squarecrop=True).transform(image)

    assert plain.max().item() > 0.5  # the band survives a bare resize
    assert cropped.max().item() == 0.0  # and is gone after the centre crop
    assert plain.shape == cropped.shape == (3, 16, 16)


def test_squarecrop_and_row_filter_compose(sid_set_dir):
    """The two controls are independent knobs and must not fight: rows are
    chosen from original geometry, pixels are cropped afterwards."""
    dm = _datamodule(sid_set_dir, geometry_filter="matched", squarecrop=True)
    dm.setup()

    images, labels = next(iter(dm.test_dataloader()))
    assert images.shape[-2:] == (16, 16)
    assert labels.dtype == torch.long
    assert len(set(_square_rate_by_label(dm.test_dataset).values())) == 1
