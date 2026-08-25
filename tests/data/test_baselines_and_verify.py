"""Tests for the trivial baselines and the provenance checks. Synthetic
parquet with a controllable geometry artifact. No network, no GPU."""

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from trustfake.data.baselines import compute_trivial_baselines, headline
from trustfake.data.manifest import PROFILES
from trustfake.data.verify import (
    verify_manifest_reproducible,
    verify_run_outputs,
)


def _png_bytes(w, h):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(w, h):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_shard(path, rows):
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
def sid_set_with_artifact(tmp_path):
    """Reals are non-square JPEGs, fakes (labels 1,2) are square PNGs -- the
    SID-Set shortcut, in miniature. smoke profile: fit=2, calib=1, test=2."""

    def rows(prefix, n):
        out = []
        for j in range(n):
            if j % 2 == 0:  # real: non-square jpeg
                out.append((f"{prefix}_{j}", 0, _jpeg_bytes(64, 48)))
            else:  # fake: square png
                out.append((f"{prefix}_{j}", 1 + (j % 2), _png_bytes(64, 64)))
        return out

    for i in range(2):
        _write_shard(tmp_path / f"train-{i:05d}-of-00002.parquet", rows(f"t{i}", 10))
    for i in range(3):
        _write_shard(
            tmp_path / f"validation-{i:05d}-of-00003.parquet", rows(f"v{i}", 10)
        )
    return tmp_path


def test_square_is_fake_is_perfect_on_the_artifact(sid_set_with_artifact):
    b = compute_trivial_baselines(
        sid_set_with_artifact, profile="smoke", split_role="test"
    )
    # every fake is square, every real is not -> the rule is exact here
    assert b["square_is_fake"] == 1.0
    assert b["png_is_fake"] == 1.0
    assert b["majority_class"] == 0.5
    assert b["square_rate_by_label"]["0"] == 0.0  # reals never square
    assert "width==height" in headline(b)


def test_baselines_reads_original_dims_not_resized(sid_set_with_artifact):
    """The shortcut is measured on original bytes; a 64x48 real must read as
    non-square even though the datamodule would resize it to a square."""
    b = compute_trivial_baselines(
        sid_set_with_artifact, profile="smoke", split_role="calib"
    )
    assert 0.0 <= b["square_is_fake"] <= 1.0
    assert b["n"] == 1 * 10  # smoke calib = 1 shard


@pytest.fixture
def sid_set_with_generation_scale(tmp_path):
    """The decode-scale residue in miniature: fakes are generated at
    1024x1024, reals are 640x480. Note the reals here are also square-free,
    so both rules are exact and the two can be told apart by which control
    removes them."""

    def rows(prefix, n):
        out = []
        for j in range(n):
            if j % 2 == 0:
                out.append((f"{prefix}_{j}", 0, _jpeg_bytes(640, 480)))
            else:
                out.append((f"{prefix}_{j}", 1, _png_bytes(1024, 1024)))
        return out

    for i in range(2):
        _write_shard(tmp_path / f"train-{i:05d}-of-00002.parquet", rows(f"t{i}", 6))
    for i in range(3):
        _write_shard(
            tmp_path / f"validation-{i:05d}-of-00003.parquet", rows(f"v{i}", 6)
        )
    return tmp_path


def test_shortside_residue_is_reported(sid_set_with_generation_scale):
    """The baseline that a centre crop does NOT remove: cropping preserves
    the short side exactly, so 'short side == 1024 -> fake' survives the
    geometry control that kills 'width == height'. Without it in the table a
    controlled row would look cleaner than it is."""
    b = compute_trivial_baselines(
        sid_set_with_generation_scale, profile="smoke", split_role="test"
    )

    assert b["shortside1024_is_fake"] == 1.0
    assert b["shortside1024_rate_by_label"]["0"] == 0.0
    assert b["shortside1024_rate_by_label"]["1"] == 1.0


def test_headline_is_protocol_aware(sid_set_with_artifact):
    """Printing the raw 0.98 next to a geometry-controlled row would misstate
    the protocol -- and in the direction that makes an honest result look
    like a failure to clear a bar it was never read against."""
    b = compute_trivial_baselines(
        sid_set_with_artifact, profile="smoke", split_role="test"
    )

    raw = headline(b)
    assert "TRIVIAL BASELINE" in raw
    assert "width==height" in raw

    for condition in ("squarecrop", "square", "nonsquare", "matched"):
        controlled = headline(b, condition=condition)
        assert "GEOMETRY-CONTROLLED" in controlled, condition
        assert f"'{condition}'" in controlled
        assert f"{b['majority_class']:.4f}" in controlled
        # the raw number is named as the RAW number, never as this row's
        assert "On the RAW" in controlled

    # An uncontrolled condition keeps the plain wording.
    assert "TRIVIAL BASELINE" in headline(b, condition="jpeg_q40")
    assert "TRIVIAL BASELINE" in headline(b, condition=None)


def test_headline_always_names_the_surviving_residue(sid_set_with_artifact):
    """Both branches must carry it: the geometry control is evidence about
    'width == height' and about nothing else."""
    b = compute_trivial_baselines(
        sid_set_with_artifact, profile="smoke", split_role="test"
    )

    for condition in (None, "squarecrop"):
        assert "short side == 1024" in headline(b, condition=condition)


def test_verify_manifest_reproducible_passes(sid_set_with_artifact):
    fails = verify_manifest_reproducible(sid_set_with_artifact, PROFILES["smoke"])
    assert fails == []


def test_verify_flags_missing_data(tmp_path):
    fails = verify_manifest_reproducible(tmp_path, PROFILES["smoke"])
    assert any("No SID_Set parquet shards" in f for f in fails)


def test_verify_run_outputs(tmp_path):
    fails, warns = verify_run_outputs(tmp_path / "nope")
    assert fails and "does not exist" in fails[0]

    run = tmp_path / "run"
    run.mkdir()
    fails, warns = verify_run_outputs(run)
    assert any("no checkpoints" in f for f in fails)

    (run / "model-epoch01.ckpt").write_bytes(b"x")
    (run / "config.yaml").write_text("a: 1\n")
    fails, warns = verify_run_outputs(run)
    assert fails == []
