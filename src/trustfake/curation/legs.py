"""Evaluation legs L1-L4, frozen before curation starts (spec table).

L1  SID-Set test role -- the same 26 validation shards the TB-E2 grid
    reported on (profile 'train', manifest_seed 0, via
    `trustfake.data.manifest.assign_shards`), so a TB-E3 L1 number and a
    TB-E2 number describe the same rows. The calib role rides along for
    any post-hoc quantity that needs fitting.
L2  So-Fake-OOD, every cached row. Untouched: no So-Fake-Set sibling data
    anywhere in training (policy).
L3  >= 2 entire held-out generators, reserved at ingestion, preferably
    newer than any training generator.
L4  >= 1 entire held-out manipulation tool, reserved at ingestion.

`freeze_legs` writes ``frozen_legs.json`` (choices + counts) plus one uid
parquet per leg, and REFUSES to overwrite an existing manifest: frozen
means frozen -- delete the file by hand if the freeze itself was wrong,
and say so in the docs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trustfake.data.manifest import DEFAULT_MANIFEST_SEED, PROFILES, assign_shards
from trustfake.logging import get_logger

logger = get_logger("curation.legs")

__all__ = ["sid_roles", "freeze_legs", "load_legs"]

SID_PROFILE = "train"  # what the TB-E2 grid trained and reported under


def sid_roles(sid_index: pd.DataFrame) -> pd.Series:
    """Role per cached SID-Set row (fit / calib / test / unused).

    Reconstructed from the shard names through `assign_shards`, the same
    function the datamodule uses -- one implementation, zero drift.
    """
    shards = sid_index["shard"].unique().tolist()
    train_shards = sorted(s for s in shards if s.startswith("train-"))
    val_shards = sorted(s for s in shards if s.startswith("validation-"))
    assignment = assign_shards(
        train_shards, val_shards, PROFILES[SID_PROFILE], DEFAULT_MANIFEST_SEED
    )
    role_of_shard = {
        shard: role for role, names in assignment.items() for shard in names
    }
    return sid_index["shard"].map(lambda s: role_of_shard.get(s, "unused"))


def freeze_legs(
    out_dir: str | Path,
    sid_index: pd.DataFrame,
    sfo_index: pd.DataFrame,
    l3_uids: pd.DataFrame,
    l4_uids: pd.DataFrame,
    l3_choice: dict,
    l4_choice: dict,
) -> Path:
    """Write the frozen-legs manifest. l3/l4 frames need uid+label3 columns."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "frozen_legs.json"
    if manifest_file.exists():
        msg = (
            f"{manifest_file} already exists -- the legs are frozen. Delete it "
            "by hand only to redo a wrong freeze, and record why in the docs."
        )
        logger.error(msg)
        raise FileExistsError(msg)

    roles = sid_roles(sid_index)
    legs = {
        "L1": sid_index.loc[roles == "test", ["uid", "label3"]],
        "L2": sfo_index[["uid", "label3"]],
        "L3": l3_uids[["uid", "label3"]],
        "L4": l4_uids[["uid", "label3"]],
    }
    calib = sid_index.loc[roles == "calib", ["uid", "label3"]]
    for name, frame in {**legs, "calib": calib}.items():
        pq.write_table(
            pa.Table.from_pandas(frame.reset_index(drop=True)),
            out / f"{name}.parquet",
        )

    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "L1": {
            "definition": (
                f"SID-Set test role, profile '{SID_PROFILE}', "
                f"manifest_seed {DEFAULT_MANIFEST_SEED}"
            ),
            "n": int(len(legs["L1"])),
        },
        "L2": {
            "definition": "So-Fake-OOD, all cached shards",
            "n": int(len(legs["L2"])),
        },
        "L3": {**l3_choice, "n": int(len(legs["L3"]))},
        "L4": {**l4_choice, "n": int(len(legs["L4"]))},
        "calib": {
            "definition": "SID-Set calib role (rides along, never reported)",
            "n": int(len(calib)),
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))
    counts = {k: v["n"] for k, v in manifest.items() if isinstance(v, dict)}
    logger.info(f"legs frozen: {counts}")
    return manifest_file


def load_legs(out_dir: str | Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    out = Path(out_dir)
    manifest = json.loads((out / "frozen_legs.json").read_text())
    frames = {
        name: pq.read_table(out / f"{name}.parquet").to_pandas()
        for name in ("L1", "L2", "L3", "L4", "calib")
    }
    return manifest, frames
