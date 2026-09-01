"""TB-E3 arm generation: C0-C5b as index lists (spec section "Arms").

Every arm is (uids, selection seed), reproducible from the cache manifests
plus this module -- never a file copy. Operationalizations that the spec
left to the executor, decided BEFORE any fit and recorded here:

  * G2 leg-decontamination applies to EVERY arm, C0 included: the gate
    protects the measurement, not the arm. C0's "naive pool, as shipped"
    is therefore "as shipped, minus rows that would leak into L1-L4".
  * C0 maps unmappable binary fakes to SYNTHETIC -- the naive default the
    colleague repo's combined.py uses, which is exactly the practice the
    ladder exists to interrogate. C1/C2 inherit it (their deltas must be
    hygiene and marginals, nothing else).
  * C1 hygiene = within-pool near-duplicate removal (keep-first at the G2
    threshold) + decode-clean rows only. Label-map corrections would land
    here too; G4 found none for the ingested shortlist.
  * C2 matching bins: format x JPEG-quality band x min-side band x
    squareness x resample direction (the route-to-224 signature). Within
    each environment, every class present is subsampled to the per-bin
    minimum across classes, seeded. Single-class environments pass through
    untouched -- there is no within-environment shortcut to remove when
    the environment has one class, and dropping them would change the
    environment mix, which is C4's variable, not C2's.
  * C4/C5 build on the STRICT-DROP mapping (C3a), the pre-registered tie
    default for H2. With zero unmappable rows in the ingested shortlist
    C3a == C3b == C2 structurally; the machinery stays for reuse.
  * C4 equalizes every (class x environment) cell to the size of the
    smallest cell that meets the G3 minimum; cells below the minimum pass
    through flagged (G3 reports them) rather than silently inflating the
    equalized budget.
  * C5's budget B = |C4| by construction, so C4 vs C5a vs C5b compare
    ALLOCATION at a fixed total: equalized cells vs random vs coverage
    coreset. C5b is greedy k-center on the cached embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trustfake.curation.gates import G2_COSINE_THRESHOLD, G3_MIN_CELL
from trustfake.logging import get_logger

logger = get_logger("curation.arms")

__all__ = ["ARM_NAMES", "nuisance_bins", "build_arm"]

ARM_NAMES = ("C0", "C1", "C2", "C3a", "C3b", "C4", "C5a", "C5b")


def nuisance_bins(index: pd.DataFrame) -> pd.Series:
    """The C2 matching key: one string bin per row.

    format x JPEG-quality band x min-side band x squareness x resample
    direction. Coarse on purpose: bins must stay populated enough to match
    within, and every axis here is one a headers-only classifier (G1) or
    the resize path could read.
    """
    q = index["jpeg_q"]
    q_band = pd.cut(
        q, bins=[0, 69, 79, 89, 94, 99, 100], labels=["<70", "70s", "80s", "90-94", "95-99", "100"]
    ).astype("string")
    q_band = q_band.fillna("noq")
    min_side = index[["width", "height"]].min(axis=1)
    side_band = pd.cut(
        min_side,
        bins=[0, 223, 512, 1024, 10**9],
        labels=["<224", "224-512", "513-1024", ">1024"],
    ).astype("string")
    resample = pd.Series(
        np.where(min_side > 224, "down", np.where(min_side < 224, "up", "none")),
        index=index.index,
    )
    square = (index["width"] == index["height"]).map({True: "sq", False: "rect"})
    return (
        index["format"].fillna("unknown").astype(str)
        + "|" + q_band.astype(str)
        + "|" + side_band.astype(str)
        + "|" + square.astype(str)
        + "|" + resample.astype(str)
    )


def _drop_leg_leakage(
    pool: pd.DataFrame,
    pool_features: np.ndarray,
    leg_features: np.ndarray,
    device: str,
) -> pd.DataFrame:
    from trustfake.curation.gates import g2_max_cosine

    best = g2_max_cosine(pool_features, leg_features, device=device)
    keep = best < G2_COSINE_THRESHOLD
    logger.info(
        f"G2 leg-decontamination: dropped {int((~keep).sum())} of {len(pool)} rows "
        f"(max-cos >= {G2_COSINE_THRESHOLD})"
    )
    return pool.loc[keep]


def _dedup_within(
    pool: pd.DataFrame, pool_features: np.ndarray, device: str, chunk: int = 2048
) -> pd.DataFrame:
    """Keep-first near-duplicate removal within the pool (C1 hygiene).

    Row order is the cache order (deterministic), so "first" is stable.
    Chunked lower-triangular max-cosine on the GPU.
    """
    torch_device = torch.device(device)
    rows = pool["cache_row"].to_numpy()
    features = torch.from_numpy(pool_features).to(torch_device, torch.float16)
    n = features.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=torch_device)
    for start in range(chunk, n, chunk):
        block = features[start : start + chunk]
        # against every EARLIER row that itself survived
        earlier = features[:start]
        alive = keep[:start]
        sims = (block @ earlier.T)
        sims[:, ~alive] = 0
        dup = sims.max(dim=1).values >= G2_COSINE_THRESHOLD
        keep[start : start + chunk] &= ~dup
    kept = keep.cpu().numpy()
    logger.info(f"C1 within-pool dedup: dropped {int((~kept).sum())} of {n} rows")
    del features
    torch.cuda.empty_cache()
    return pool.iloc[np.flatnonzero(kept)]


def _match_marginals(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """C2: equalize nuisance-bin marginals across classes, per environment."""
    rng = np.random.default_rng(seed)
    bins = nuisance_bins(pool)
    kept_positions: list[np.ndarray] = []
    for env, env_rows in pool.groupby("dataset"):
        classes = env_rows["label3"].unique()
        if len(classes) < 2:
            kept_positions.append(env_rows.index.to_numpy())
            continue
        env_bins = bins.loc[env_rows.index]
        for bin_key, bin_rows in env_rows.groupby(env_bins):
            counts = bin_rows["label3"].value_counts()
            quota = int(counts.reindex(classes).fillna(0).min())
            if quota == 0:
                continue
            for _, class_rows in bin_rows.groupby("label3"):
                positions = class_rows.index.to_numpy()
                if positions.size > quota:
                    positions = rng.choice(positions, size=quota, replace=False)
                kept_positions.append(np.sort(positions))
    keep = np.sort(np.concatenate(kept_positions))
    logger.info(f"C2 matching: kept {keep.size} of {len(pool)} rows")
    return pool.loc[keep]


def _equalize_cells(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """C4: every (class x environment) cell at the smallest gate-passing cell's n."""
    rng = np.random.default_rng(seed)
    sizes = pool.groupby(["dataset", "label3"]).size()
    eligible = sizes[sizes >= G3_MIN_CELL]
    if eligible.empty:
        raise ValueError("C4: no (class x environment) cell meets the G3 minimum")
    budget = int(eligible.min())
    kept_positions = []
    for _, cell_rows in pool.groupby(["dataset", "label3"]):
        positions = cell_rows.index.to_numpy()
        if positions.size > budget:
            positions = rng.choice(positions, size=budget, replace=False)
        kept_positions.append(np.sort(positions))
    keep = np.sort(np.concatenate(kept_positions))
    logger.info(
        f"C4 equalize: budget {budget}/cell over {len(sizes)} cells -> {keep.size} rows"
    )
    return pool.loc[keep]


def _k_center(
    pool: pd.DataFrame,
    pool_features: np.ndarray,
    budget: int,
    seed: int,
    device: str,
    batch: int = 64,
) -> pd.DataFrame:
    """C5b: greedy k-center (farthest-point) coverage coreset at `budget`.

    Batched greedy: each step adds the `batch` currently-farthest points,
    then refreshes the min-distance array -- the standard practical
    relaxation that keeps the step count tractable at coreset scale.
    """
    torch_device = torch.device(device)
    features = torch.from_numpy(pool_features).to(torch_device, torch.float16)
    n = features.shape[0]
    if budget >= n:
        return pool
    generator = np.random.default_rng(seed)
    start = int(generator.integers(0, n))
    min_distance = 1.0 - (features @ features[start]).float()
    chosen = [start]
    while len(chosen) < budget:
        take = min(batch, budget - len(chosen))
        farthest = torch.topk(min_distance, take).indices
        chosen.extend(farthest.tolist())
        sims = (features @ features[farthest].T).float()
        distance = 1.0 - sims.max(dim=1).values
        min_distance = torch.minimum(min_distance, distance)
        min_distance[farthest] = -1.0
    del features
    torch.cuda.empty_cache()
    positions = pool.index.to_numpy()[np.array(sorted(set(chosen)))]
    logger.info(f"C5b k-center: {len(positions)} of {n} rows")
    return pool.loc[positions]


def build_arm(
    arm: str,
    pool: pd.DataFrame,
    features: np.ndarray,
    leg_features: np.ndarray,
    seed: int,
    device: str = "cuda",
) -> pd.DataFrame:
    """The arm's index rows (a subset of `pool`), by the ladder above.

    `pool` must be the FULL training-candidate pool (leg-reserved uids and
    eval-only datasets already excluded upstream), row-aligned with
    `features`; `leg_features` is the concatenated L1-L4 feature matrix.
    """
    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown arm '{arm}'. Arms: {ARM_NAMES}")

    pool = pool.reset_index(drop=True)
    base = _drop_leg_leakage(pool, features, leg_features, device)
    base = base.copy()
    # Naive mapping: unmappable binary fakes fold to synthetic (C0-C2).
    naive = base.assign(
        label3=np.where(base["label3"] == -1, 1, base["label3"]),
        binary_only=False,
    )
    if arm == "C0":
        return naive

    deduped_rows = _dedup_within(
        base, features[base.index.to_numpy()], device
    ).index.to_numpy()
    c1 = naive.loc[deduped_rows]
    if arm == "C1":
        return c1

    c2 = _match_marginals(c1, seed)
    if arm == "C2":
        return c2

    if arm == "C3b":
        rows = base.loc[c2.index]
        return rows.assign(binary_only=rows["label3"] == -1).assign(
            label3=lambda d: d["label3"].where(d["label3"] != -1, 1)
        )
    # C3a and everything above it: strict drop.
    c3a = base.loc[c2.index]
    c3a = c3a.loc[c3a["label3"] != -1].assign(binary_only=False)
    if arm == "C3a":
        return c3a

    c4 = _equalize_cells(c3a, seed)
    if arm == "C4":
        return c4

    budget = len(c4)
    if arm == "C5a":
        rng = np.random.default_rng(seed + 7)
        positions = np.sort(
            rng.choice(c3a.index.to_numpy(), size=min(budget, len(c3a)), replace=False)
        )
        logger.info(f"C5a random: {positions.size} of {len(c3a)} rows")
        return c3a.loc[positions]
    return _k_center(
        c3a, features[c3a.index.to_numpy()], budget, seed, device
    )
