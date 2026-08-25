"""Shard-level split manifest for SID_Set: a structural leakage firewall.

The framework previously reported metrics on the same rows that early stopping
and checkpointing selected on (`_resolve_splits` returned the validation split
as both `val_ds` and `test_ds`). This module makes that impossible without
editing this file, by assigning whole parquet shards to roles:

    fit      train-* shards only. The model's training data. The
             model-selection slice (early stopping, checkpointing) is carved
             from fit at row level, so calib and test are never seen by
             selection.
    calib    validation-* shards only. Reserved for fitting post-hoc
             quantities (temperature, moderation thresholds); never reported.
    test     validation-* shards only, never a shard used by calib. The
             reported split.
    holdout  optional. train-* shards disjoint from fit: a sealed tranche for
             one final confirmation, untouched by any decision made earlier.

Two properties are deliberate and load-bearing:

  * The validation-shard permutation depends only on `seed`, not on how many
    train shards a profile uses -- so enlarging `fit` (e.g. `full` -> `train`)
    leaves calib and test identical, and every earlier number stays comparable.
  * `seed` here is the MANIFEST seed, a constant, and must never be derived
    from the experiment seed: the reported split must not move when the
    training seed does.

Naming discipline: the official SID-Set test split is withheld by the authors
(SIDA repository, to prevent leakage). Everything called "test" here is carved
from the VALIDATION split; `SPLIT_PROVENANCE` is what reports must print.
Never write "SID-Set test set".

The row key, where one is needed, is ``uid = "<source_split>:<img_id>"``.
`img_id` alone is NOT unique across source splits -- the synthetic and
tampered classes are numbered sequentially and the counter restarts per split,
so thousands of img_ids exist in both train and validation with different
image bytes. Nothing errors when this is keyed wrong; rows are silently
merged or overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trustfake.logging import get_logger

logger = get_logger("manifest")

__all__ = [
    "PROFILES",
    "DEFAULT_MANIFEST_SEED",
    "SPLIT_PROVENANCE",
    "assign_shards",
    "discover_shards",
    "build_manifest",
]

SPLIT_PROVENANCE = (
    "SID-Set validation split, held-out slice "
    "(official test split withheld by the dataset authors)"
)

# The manifest seed is a project constant, independent of experiment.seed.
DEFAULT_MANIFEST_SEED = 0

# Shard counts per role. SID_Set ships 249 train / 34 validation shards.
PROFILES: dict[str, dict[str, int]] = {
    # Offline pre-flight: enough shards to exercise every code path.
    "smoke": {"fit": 2, "calib": 1, "test": 2},
    # Linear-probe scale evaluation.
    "full": {"fit": 12, "calib": 8, "test": 26},
    # Weight training needs more fit data than a probe does. calib and test
    # stay identical to "full" (same seed, same validation permutation, which
    # does not depend on the train-shard count).
    "train": {"fit": 30, "calib": 8, "test": 26},
    # As "train", plus a sealed holdout drawn from train shards disjoint from
    # fit. All 34 validation shards are consumed by calib+test, so unused
    # train shards are the only genuinely-unseen pool.
    "train_holdout": {"fit": 30, "calib": 8, "test": 26, "holdout": 6},
}


def assign_shards(
    train_shards: list[str],
    val_shards: list[str],
    counts: dict[str, int],
    seed: int = DEFAULT_MANIFEST_SEED,
) -> dict[str, list[str]]:
    """
    Assign shard names to roles. Pure: a function of the sorted name lists,
    the per-role counts and the seed, with the firewall enforced by assertion.

    Args:
        train_shards: Names of the train-* shards available.
        val_shards: Names of the validation-* shards available.
        counts: Per-role shard counts, e.g. an entry of `PROFILES`.
        seed: Manifest seed. A project constant -- never the experiment seed.

    Returns:
        Mapping role -> list of shard names.
    """
    train_shards = sorted(train_shards)
    val_shards = sorted(val_shards)

    need_train = counts["fit"] + counts.get("holdout", 0)
    need_val = counts["calib"] + counts["test"]
    if len(train_shards) < need_train or len(val_shards) < need_val:
        msg = (
            f"Not enough shards: have {len(train_shards)} train / "
            f"{len(val_shards)} validation, need {need_train} / {need_val}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # The permutation covers validation shards only, so it is independent of
    # the train-shard counts by construction.
    val_order = np.random.default_rng(seed).permutation(len(val_shards))

    assign = {
        "fit": train_shards[: counts["fit"]],
        "calib": [val_shards[i] for i in val_order[: counts["calib"]]],
        "test": [
            val_shards[i]
            for i in val_order[counts["calib"] : counts["calib"] + counts["test"]]
        ],
    }
    if counts.get("holdout"):
        assign["holdout"] = train_shards[
            counts["fit"] : counts["fit"] + counts["holdout"]
        ]

    # Structural assertions: these ARE the firewall, not comments about it.
    assert all(Path(s).name.startswith("train-") for s in assign["fit"]), (
        "fit must come from train shards only"
    )
    assert all(
        Path(s).name.startswith("validation-") for s in assign["calib"] + assign["test"]
    ), "calib and test must come from validation shards only"
    assert not (set(assign["calib"]) & set(assign["test"])), (
        "calib and test share a shard -- post-hoc quantities would be fitted "
        "on reported data"
    )
    if "holdout" in assign:
        assert not (set(assign["fit"]) & set(assign["holdout"])), (
            "holdout shares a shard with fit -- it would not be unseen"
        )
        assert all(Path(s).name.startswith("train-") for s in assign["holdout"]), (
            "holdout must come from train shards only"
        )
    return assign


def discover_shards(data_dir: str | Path) -> tuple[list[Path], list[Path]]:
    """
    Find the SID_Set parquet shards under `data_dir` (recursively).

    Returns:
        (train_shards, validation_shards), each sorted by name.
    """
    data_dir = Path(data_dir)
    train = sorted(data_dir.rglob("train-*.parquet"))
    val = sorted(data_dir.rglob("validation-*.parquet"))
    if not train or not val:
        msg = (
            f"No SID_Set parquet shards under {data_dir} "
            f"(found {len(train)} train-*.parquet, {len(val)} "
            "validation-*.parquet). Fetch them first -- see "
            "jobs/download_sidset.sh."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
    return train, val


def build_manifest(
    data_dir: str | Path,
    profile: str,
    seed: int = DEFAULT_MANIFEST_SEED,
) -> dict[str, list[Path]]:
    """
    Discover the shards on disk and assign them to roles for `profile`.

    Returns:
        Mapping role -> list of shard paths.
    """
    if profile not in PROFILES:
        msg = f"Unknown profile '{profile}'. Available profiles: {list(PROFILES)}"
        logger.error(msg)
        raise ValueError(msg)

    train, val = discover_shards(data_dir)
    by_name = {p.name: p for p in [*train, *val]}
    assign = assign_shards(
        [p.name for p in train], [p.name for p in val], PROFILES[profile], seed
    )
    manifest = {role: [by_name[n] for n in names] for role, names in assign.items()}

    for role, paths in manifest.items():
        logger.debug(f"manifest[{role}]: {len(paths)} shards")
    logger.info(f"Split provenance: {SPLIT_PROVENANCE}")
    return manifest
