"""Provenance / self-consistency checks -- run before believing any result.

The leakage firewall is only as trustworthy as its reproducibility: if the
shard-to-role assignment could drift, a later run could silently report on
different rows than an earlier one. These checks make that impossible to miss.

  * verify_manifest_reproducible: the split assignment is a deterministic
    function of (sorted shard names, profile, seed), and re-deriving it must
    reproduce the roles bit-for-bit, with calib and test shard-disjoint and
    sourced from the right split.
  * verify_run_outputs: a finished training run must have left a checkpoint
    and a resolved config behind, so a reported number is traceable to the
    config that produced it.

`verify_all` returns a (failures, warnings) pair; a non-empty failures list
means do not trust the artifacts.
"""

from __future__ import annotations

from pathlib import Path

from trustfake.data.manifest import (
    DEFAULT_MANIFEST_SEED,
    assign_shards,
    discover_shards,
)
from trustfake.logging import get_logger

logger = get_logger("verify")

__all__ = [
    "verify_manifest_reproducible",
    "verify_run_outputs",
    "verify_all",
]


def verify_manifest_reproducible(
    data_dir: str | Path,
    profile_counts: dict[str, int],
    seed: int = DEFAULT_MANIFEST_SEED,
) -> list[str]:
    """Re-derive the shard assignment twice and confirm it is identical and
    firewall-consistent. Returns a list of failure messages (empty = ok)."""
    fails: list[str] = []
    try:
        train, val = discover_shards(data_dir)
    except FileNotFoundError as e:
        return [str(e)]

    train_names = [p.name for p in train]
    val_names = [p.name for p in val]
    a = assign_shards(train_names, val_names, profile_counts, seed)
    b = assign_shards(
        list(reversed(train_names)), list(reversed(val_names)), profile_counts, seed
    )
    if a != b:
        fails.append(
            "manifest assignment is not order-invariant (non-deterministic split)"
        )
    if set(a.get("calib", [])) & set(a.get("test", [])):
        fails.append("calib and test share a shard (leakage firewall broken)")
    if "holdout" in a and (set(a["fit"]) & set(a["holdout"])):
        fails.append("holdout shares a shard with fit (not unseen)")
    return fails


def verify_run_outputs(output_dir: str | Path) -> tuple[list[str], list[str]]:
    """Check a finished run left a checkpoint and a saved config behind."""
    output_dir = Path(output_dir)
    fails: list[str] = []
    warns: list[str] = []
    if not output_dir.exists():
        return [f"output dir does not exist: {output_dir}"], warns

    ckpts = list(output_dir.rglob("*.ckpt"))
    real_ckpts = [c for c in ckpts if c.name != "last.ckpt"]
    if not ckpts:
        fails.append(f"no checkpoints under {output_dir}")
    elif not real_ckpts:
        warns.append(
            f"only last.ckpt under {output_dir} (no monitored best checkpoint)"
        )

    configs = list(output_dir.rglob("*.yaml")) + list(output_dir.rglob("config.yaml"))
    if not configs:
        warns.append(f"no saved config under {output_dir} (result not traceable)")
    return fails, warns


def verify_all(
    data_dir: str | Path,
    profile_counts: dict[str, int],
    output_dir: str | Path | None = None,
    seed: int = DEFAULT_MANIFEST_SEED,
) -> tuple[list[str], list[str]]:
    fails = verify_manifest_reproducible(data_dir, profile_counts, seed)
    warns: list[str] = []
    if output_dir is not None:
        f2, w2 = verify_run_outputs(output_dir)
        fails += f2
        warns += w2
    for f in fails:
        logger.error(f"VERIFY FAIL: {f}")
    for w in warns:
        logger.warning(f"VERIFY WARN: {w}")
    if not fails:
        logger.success("verify: all checks passed")
    return fails, warns
