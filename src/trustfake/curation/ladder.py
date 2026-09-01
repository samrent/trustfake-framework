"""TB-E3 driver: pool -> frozen legs -> gates -> arms -> 48 fits -> table.

Everything lands under ``$OUTPUT_PATH/tb_e3/``: the pool index, the frozen
legs, gate reports, arm uid-lists and one JSON per completed fit (which is
also the resume marker -- a fit whose JSON exists is skipped). Protocol
constants live at module top, chosen and recorded before any fit ran.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trustfake.curation.cacheview import load_cache
from trustfake.curation.legs import sid_roles
from trustfake.logging import get_logger

logger = get_logger("curation.ladder")

__all__ = ["build_pool", "freeze_all_legs", "run_gates", "run_fits", "collate"]

#: Training environments and which cached rows of each may enter the pool.
#: (dataset -> filter name; the filters live in `_POOL_FILTERS`.)
POOL_DATASETS = (
    "sid_set",
    "community_forensics_small",
    "audits",
    "synthbuster",
    "vision",
    "imd2020",
    "tgif",
    "sagi_d",
)

#: L3, frozen at ingestion: entire commercial generators, newest available
#: in any training environment, reserved across ALL environments (name
#: patterns, case-insensitive). So-Fake-OOD (L2) covers the 2025-era
#: generators; L3 covers held-out closed-source mechanisms.
L3_GENERATOR_PATTERNS = ("dalle3", "dall-e-3", "midjourney", "firefly")

#: Reals reserved (seeded) from CF-Small as L3 negatives, excluded from
#: every arm. Drawn stratified by real_source.
L3_REAL_RESERVE = 6000
L3_RESERVE_SEED = 20260901

#: Selection seeds per arm (spec: 3), and the two size protocols.
SEEDS = (1, 2, 3)
SIZES = ("natural", "matched")


def _out(root: str | Path) -> Path:
    return Path(root) / "tb_e3"


def _pool_filter(dataset: str, index: pd.DataFrame) -> pd.Series:
    if dataset == "sid_set":
        return sid_roles(index) == "fit"
    if dataset == "audits":
        return index["source_split"].isin(["train", "val"])
    if dataset == "tgif":
        keep = index["source_split"].astype(str).str.contains("training")
        return keep & (index.get("tool", pd.Series("", index=index.index)) != "ps-sp")
    if dataset == "sagi_d":
        # Frozen L4 rule: PowerPaint has zero training presence, so any row
        # whose (possibly multi-tool) generator mentions it stays out.
        keep = index["source_split"].isin(["train", "val"])
        powerpaint = (
            index["generator"].fillna("").astype(str).str.contains("powerpaint")
        )
        return keep & ~powerpaint
    return pd.Series(True, index=index.index)


def _l3_generator_mask(index: pd.DataFrame) -> pd.Series:
    generator = index["generator"].fillna("").astype(str).str.lower()
    pattern = "|".join(re.escape(p) for p in L3_GENERATOR_PATTERNS)
    return generator.str.contains(pattern, regex=True)


def _l3_real_reserve_uids(cf_index: pd.DataFrame) -> pd.Index:
    reals = cf_index.loc[cf_index["label3"] == 0]
    rng = np.random.default_rng(L3_RESERVE_SEED)
    per_source = max(1, L3_REAL_RESERVE // max(1, reals["real_source"].nunique()))
    chosen: list[np.ndarray] = []
    for _, group in reals.groupby("real_source"):
        take = min(per_source, len(group))
        chosen.append(rng.choice(group["uid"].to_numpy(), size=take, replace=False))
    return pd.Index(np.concatenate(chosen))


def build_pool(
    features_root: str | Path, out_root: str | Path
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble the training-candidate pool across every available cache.

    Returns (pool index with a global ``pool_row``, features aligned to it).
    Rows excluded here, in order: non-pool roles/splits, L3 generators
    (entire generators, every environment), the reserved L3-negative reals,
    the L4 held-out tools (handled by the audits/tgif filters upstream).
    """
    frames, blocks = [], []
    for dataset in POOL_DATASETS:
        try:
            index, features = load_cache(features_root, dataset)
        except FileNotFoundError:
            logger.warning(f"pool: no cache for {dataset} -- skipped")
            continue
        keep = _pool_filter(dataset, index)
        l3 = _l3_generator_mask(index)
        if l3.any():
            logger.info(f"pool: {dataset}: {int(l3.sum())} rows to L3 generators")
        keep &= ~l3
        if dataset == "community_forensics_small":
            reserve = _l3_real_reserve_uids(index)
            keep &= ~index["uid"].isin(reserve)
            logger.info(f"pool: reserved {len(reserve)} CF reals as L3 negatives")
        kept = index.loc[keep].copy()
        frames.append(kept)
        blocks.append(features[kept["cache_row"].to_numpy()])
        logger.info(f"pool: {dataset}: {len(kept)} rows in")
    pool = pd.concat(frames, ignore_index=True)
    features = np.concatenate(blocks, axis=0)
    pool["pool_row"] = np.arange(len(pool))

    out = _out(out_root) / "pool"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(pool.drop(columns=["cache_row"])), out / "pool.parquet"
    )
    counts = pool.groupby(["dataset", "label3"]).size()
    logger.info(f"pool: {len(pool)} rows total\n{counts}")
    return pool, features


def freeze_all_legs(features_root: str | Path, out_root: str | Path) -> None:
    """Runbook step 4. Refuses to re-freeze (see legs.freeze_legs)."""
    from trustfake.curation.legs import freeze_legs

    sid_index, _ = load_cache(features_root, "sid_set")
    sfo_index, _ = load_cache(features_root, "so_fake_ood")
    synthbuster_index, _ = load_cache(features_root, "synthbuster")
    cf_index, _ = load_cache(features_root, "community_forensics_small")
    audits_index, _ = load_cache(features_root, "audits")

    l3_fakes = synthbuster_index.loc[_l3_generator_mask(synthbuster_index)]
    l3_reals = cf_index.loc[cf_index["uid"].isin(_l3_real_reserve_uids(cf_index))]
    l3 = pd.concat([l3_fakes, l3_reals], ignore_index=True)

    l4_parts = [audits_index.loc[audits_index["source_split"] == "test"]]
    try:
        tgif_index, _ = load_cache(features_root, "tgif")
        # ps-sp (every split -- the tool is held out entirely) plus the
        # ORIGINALS' testing rows as negatives. The random-mask tools are
        # training tools; their testing rows must NOT enter L4, which
        # measures unseen mechanisms only.
        l4_parts.append(
            tgif_index.loc[
                (tgif_index["tool"] == "ps-sp")
                | (
                    (tgif_index["tool"] == "orig")
                    & (tgif_index["source_split"] == "testing")
                )
            ]
        )
    except FileNotFoundError:
        logger.warning("legs: no tgif cache yet -- L4 is audits-only at freeze time")
    l4 = pd.concat(l4_parts, ignore_index=True)

    freeze_legs(
        _out(out_root) / "legs",
        sid_index=sid_index,
        sfo_index=sfo_index,
        l3_uids=l3,
        l4_uids=l4,
        l3_choice={
            "definition": (
                "entire held-out generators, all environments: DALL-E 3, "
                "Midjourney v5, Adobe Firefly (Synthbuster folders; commercial, "
                "closed, newest in any training env) + seeded CF-Small real "
                f"reserve as negatives (seed {L3_RESERVE_SEED})"
            ),
            "patterns": list(L3_GENERATOR_PATTERNS),
        },
        l4_choice={
            "definition": (
                "held-out manipulation tools: AUDITS test split (PowerPaint "
                "fakes -- zero training presence by the dataset's own design -- "
                "plus test Authentic as negatives) and TGIF ps-sp "
                "(Photoshop/Firefly generative fill) with TGIF originals as "
                "negatives when cached"
            ),
        },
    )


def _leg_eval_sets(
    features_root: str | Path, out_root: str | Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(features, labels) per leg, loaded via each dataset's cache."""
    from trustfake.curation.cacheview import gather_features
    from trustfake.curation.legs import load_legs

    _, leg_frames = load_legs(_out(out_root) / "legs")
    caches: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    def dataset_of(uid: str) -> str:
        head = uid.split(":", 1)[0]
        return {
            "community_forensics": "community_forensics_small",
        }.get(head, head)

    evaluation = {}
    for leg, frame in leg_frames.items():
        if leg == "calib" or frame.empty:
            continue
        parts_features, parts_labels = [], []
        for dataset, rows in frame.groupby(frame["uid"].map(dataset_of)):
            if dataset not in caches:
                caches[dataset] = load_cache(features_root, dataset)
            index, features = caches[dataset]
            parts_features.append(gather_features(index, features, rows["uid"]))
            parts_labels.append(rows["label3"].to_numpy())
        evaluation[leg] = (
            np.concatenate(parts_features),
            np.concatenate(parts_labels),
        )
        logger.info(f"leg {leg}: {evaluation[leg][1].size} rows")
    return evaluation


def run_gates(pool: pd.DataFrame, out_root: str | Path, seed: int = 0) -> None:
    """G1 + G3 on the raw pool (per-arm gate reports ride with each fit)."""
    from trustfake.curation.gates import g1_metadata_auroc, g3_cell_counts

    out = _out(out_root) / "gates"
    out.mkdir(parents=True, exist_ok=True)
    g1 = g1_metadata_auroc(pool, seed=seed)
    (out / "g1_pool.json").write_text(json.dumps(g1, indent=2))
    g3 = g3_cell_counts(pool)
    g3.to_csv(out / "g3_pool.csv", index=False)
    logger.info(f"gates: G1 per env {g1}")


def run_fits(
    pool: pd.DataFrame,
    features: np.ndarray,
    features_root: str | Path,
    out_root: str | Path,
    device: str = "cuda",
) -> None:
    """All 8 arms x 2 sizes x 3 seeds, resumable per fit."""
    from trustfake.curation.arms import ARM_NAMES, build_arm, prepare_base
    from trustfake.curation.fit import evaluate_probe, fit_probe
    from trustfake.curation.gates import g1_metadata_auroc, g3_cell_counts

    out = _out(out_root)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    (out / "arms").mkdir(parents=True, exist_ok=True)

    evaluation = _leg_eval_sets(features_root, out_root)
    leg_features = np.concatenate([f for f, _ in evaluation.values()])

    prepared = prepare_base(pool, features, leg_features, device=device)
    # Pass 1: natural size. Track each arm's n to derive the matched budget.
    natural_sizes: dict[tuple[str, int], int] = {}
    arm_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for seed in SEEDS:
        for arm in ARM_NAMES:
            rows = build_arm(arm, prepared, features, seed, device=device)
            arm_cache[(arm, seed)] = rows
            natural_sizes[(arm, seed)] = len(rows)
    smallest = min(natural_sizes.values())
    logger.info(f"matched-size budget (smallest arm) = {smallest}")

    for size in SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed + 1000)
            for arm in ARM_NAMES:
                marker = out / "fits" / f"{arm}_{size}_s{seed}.json"
                if marker.exists():
                    logger.info(f"SKIP fit {arm} {size} s{seed}")
                    continue
                rows = arm_cache[(arm, seed)]
                if size == "matched" and len(rows) > smallest:
                    keep = np.sort(
                        rng.choice(len(rows), size=smallest, replace=False)
                    )
                    rows = rows.iloc[keep]
                arm_file = out / "arms" / f"{arm}_{size}_s{seed}.parquet"
                pq.write_table(
                    pa.Table.from_pandas(
                        rows[["uid", "dataset", "label3", "binary_only"]].reset_index(
                            drop=True
                        )
                    ),
                    arm_file,
                )
                gate_g1 = g1_metadata_auroc(rows, seed=seed)
                gate_g3 = g3_cell_counts(rows)
                arm_features = features[rows["pool_row"].to_numpy()]
                binary_mask = rows["binary_only"].to_numpy()
                result = fit_probe(
                    arm_features,
                    rows["label3"].to_numpy(),
                    seed=seed,
                    device=device,
                    binary_mask=binary_mask if binary_mask.any() else None,
                )
                metrics = {
                    leg: evaluate_probe(result["state"], f, y, device=device)
                    for leg, (f, y) in evaluation.items()
                }
                # Weights land BEFORE the JSON marker: a marker must imply
                # the head exists on disk (1,539 params; the fits are cheap
                # but the ladder's heads are the deliverable models).
                import torch

                weights_file = out / "fits" / f"{arm}_{size}_s{seed}.pt"
                torch.save(result["state"], weights_file)
                record = {
                    "arm": arm,
                    "size": size,
                    "seed": seed,
                    "n_rows": int(len(rows)),
                    "weights": weights_file.name,
                    "fitted_at": datetime.now(UTC).isoformat(
                        timespec="seconds"
                    ),
                    "val_f1": result["val_f1"],
                    "best_epoch": result["best_epoch"],
                    "g1": gate_g1,
                    "g3_undersized": int(gate_g3["undersized"].sum()),
                    "legs": metrics,
                }
                marker.write_text(json.dumps(record, indent=2))
                logger.info(
                    f"fit {arm} {size} s{seed}: n={len(rows)} "
                    + " ".join(
                        f"{leg}:auroc={m['detection_auroc']:.4f}"
                        for leg, m in metrics.items()
                    )
                )


def collate(out_root: str | Path) -> pd.DataFrame:
    """One flat results table from every completed fit."""
    records = []
    for fit_file in sorted((_out(out_root) / "fits").glob("*.json")):
        fit = json.loads(fit_file.read_text())
        for leg, metrics in fit["legs"].items():
            records.append(
                {
                    "arm": fit["arm"],
                    "size": fit["size"],
                    "seed": fit["seed"],
                    "n_rows": fit["n_rows"],
                    "leg": leg,
                    **metrics,
                }
            )
    table = pd.DataFrame(records)
    pq.write_table(pa.Table.from_pandas(table), _out(out_root) / "results.parquet")
    return table
