"""Tests for the trivial baselines and the provenance checks. Synthetic
parquet with a controllable geometry artifact. No network, no GPU."""

import io
import json
import subprocess
import sys
from pathlib import Path

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


@pytest.fixture
def sid_set_mixed_geometry(tmp_path):
    """Both geometries in both classes, so every row filter leaves a
    non-degenerate subset -- and, deliberately, a subset whose class prior
    differs from the raw split's. That gap is what a test of the controlled
    headline needs: with equal priors, quoting the raw floor beside a
    filtered accuracy would pass unnoticed.

    Per shard: 8 reals (2 square) and 4 fakes (3 square). Raw majority class
    2/3; square 0.6, nonsquare 6/7, matched 0.5.
    """

    def rows(prefix):
        out = []
        for j in range(12):
            label = 0 if j < 8 else 1
            square = j < 2 if label == 0 else j < 11
            size = (64, 64) if square else (64, 48)
            out.append((f"{prefix}_{j}", label, _png_bytes(*size)))
        return out

    for i in range(2):
        _write_shard(tmp_path / f"train-{i:05d}-of-00002.parquet", rows(f"t{i}"))
    for i in range(3):
        _write_shard(tmp_path / f"validation-{i:05d}-of-00003.parquet", rows(f"v{i}"))
    return tmp_path


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

    # squarecrop is a PRE-TRANSFORM, not a row filter: the reported rows are
    # unchanged, so the raw split's floor is the right one to quote.
    controlled = headline(b, condition="squarecrop")
    assert "GEOMETRY-CONTROLLED" in controlled
    assert "'squarecrop'" in controlled
    assert f"{b['majority_class']:.4f}" in controlled
    # the raw number is named as the RAW number, never as this row's
    assert "On the RAW" in controlled

    # An uncontrolled condition keeps the plain wording.
    assert "TRIVIAL BASELINE" in headline(b, condition="jpeg_q40")
    assert "TRIVIAL BASELINE" in headline(b, condition=None)


@pytest.mark.parametrize("mode", ["square", "nonsquare", "matched"])
def test_controlled_headline_quotes_the_filtered_floor(sid_set_mixed_geometry, mode):
    """A row filter changes WHICH rows are reported, so it changes the class
    prior and every floor computed from it. The headline beside a filtered
    accuracy must therefore quote the FILTERED subset's floor; quoting the
    raw split's is how a model that beat nothing comes to look like it
    cleared a bar."""
    raw = compute_trivial_baselines(
        sid_set_mixed_geometry, profile="smoke", split_role="test"
    )
    filtered = compute_trivial_baselines(
        sid_set_mixed_geometry, profile="smoke", split_role="test", geometry_filter=mode
    )
    assert filtered["n"] < raw["n"]

    controlled = headline(filtered, condition=mode)

    assert "GEOMETRY-CONTROLLED" in controlled
    assert f"'{mode}'" in controlled
    assert f"{filtered['majority_class']:.4f}" in controlled
    assert f"n={filtered['n']}" in controlled
    # The raw split is quoted, but named as the raw split.
    assert "On the RAW" in controlled
    assert f"n={raw['n']}" in controlled
    if filtered["majority_class"] != raw["majority_class"]:
        assert f"{raw['majority_class']:.4f}" not in controlled


@pytest.mark.parametrize("mode", ["square", "nonsquare", "matched"])
def test_controlled_headline_refuses_baselines_from_another_protocol(
    sid_set_mixed_geometry, mode
):
    """The failure this guards is silent by construction: both numbers are
    real, both are printed with four decimals, and only the protocol tells
    them apart. Refusing beats printing a plausible wrong floor."""
    raw = compute_trivial_baselines(
        sid_set_mixed_geometry, profile="smoke", split_role="test"
    )

    with pytest.raises(ValueError, match="same row filter"):
        headline(raw, condition=mode)


def test_headline_always_names_the_surviving_residue(sid_set_with_artifact):
    """Both branches must carry it: the geometry control is evidence about
    'width == height' and about nothing else."""
    b = compute_trivial_baselines(
        sid_set_with_artifact, profile="smoke", split_role="test"
    )

    for condition in (None, "squarecrop"):
        assert "short side == 1024" in headline(b, condition=condition)


# --------------------------------------------------------------------------
# the CLI: the only surface most of these numbers are ever read through
# --------------------------------------------------------------------------

BASELINES_CLI = Path(__file__).resolve().parents[2] / "src" / "baselines.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BASELINES_CLI), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _cli_json(stdout: str) -> dict:
    """The JSON document the CLI prints, extracted from a stream it shares
    with the logger -- `build_manifest` logs the split provenance to stdout,
    so the payload is never the whole of it."""
    start = stdout.index("\n{\n") + 1 if "\n{\n" in stdout else stdout.index("{")
    document, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return document


def test_cli_computes_the_baselines_on_the_filtered_subset(sid_set_mixed_geometry):
    """The flag has to reach `compute_trivial_baselines`, not merely exist:
    a CLI that accepts --geometry-filter and computes the raw numbers anyway
    prints the wrong floor with the right label on it."""
    expected = compute_trivial_baselines(
        sid_set_mixed_geometry,
        profile="smoke",
        split_role="test",
        geometry_filter="nonsquare",
    )

    result = _run_cli(
        "--data-dir",
        str(sid_set_mixed_geometry),
        "--profile",
        "smoke",
        "--geometry-filter",
        "nonsquare",
    )

    assert result.returncode == 0, result.stderr
    payload = _cli_json(result.stdout)
    assert payload["geometry_filter"] == "nonsquare"
    assert payload["n"] == expected["n"]
    assert payload["majority_class"] == expected["majority_class"]
    # The unfiltered numbers survive for contrast, under their own key.
    assert payload["raw"]["n"] > payload["n"]


def test_cli_condition_defaults_to_the_filter(sid_set_mixed_geometry):
    """--geometry-filter and --condition describe the same evaluation from
    two ends. Leaving --condition unset must not silently print the
    uncontrolled wording over controlled numbers."""
    result = _run_cli(
        "--data-dir",
        str(sid_set_mixed_geometry),
        "--profile",
        "smoke",
        "--geometry-filter",
        "matched",
    )

    assert result.returncode == 0, result.stderr
    assert "GEOMETRY-CONTROLLED protocol ('matched')" in result.stdout
    assert "TRIVIAL BASELINE" not in result.stdout


def test_cli_refuses_a_condition_that_contradicts_the_filter(sid_set_mixed_geometry):
    """Caught at the flag, before any shard is read: `headline` refuses the
    same mismatch, but only after the work, and its message names an API
    call rather than the flag to change."""
    result = _run_cli(
        "--data-dir",
        str(sid_set_mixed_geometry),
        "--profile",
        "smoke",
        "--geometry-filter",
        "matched",
        "--condition",
        "square",
    )

    assert result.returncode == 2
    assert "--geometry-filter square" in result.stderr


def test_cli_still_prints_the_uncontrolled_headline_by_default(sid_set_mixed_geometry):
    """No filter, no condition: the raw protocol, worded as such."""
    result = _run_cli("--data-dir", str(sid_set_mixed_geometry), "--profile", "smoke")

    assert result.returncode == 0, result.stderr
    assert "TRIVIAL BASELINE" in result.stdout
    assert _cli_json(result.stdout)["geometry_filter"] == "none"


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
