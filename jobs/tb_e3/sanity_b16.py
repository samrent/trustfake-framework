"""Sanity gate for the TB-E3 pipeline: reproduce TB-E2's B/16 in-domain cell.

The whole ladder stands on (embed once -> fit on features) being the same
measurement as TB-E2's (train probe on images). Before any arm is fit, this
script fits the probe on the cached SID-Set fit-role features and evaluates
the SAME 1000-row test prefix the grid used (test shards in manifest
permutation order, first 1000 rows), then prints both columns side by side.

Reference numbers (snapshots/2026-09-01-backbone-grid.md, clip_vit_b16_probe,
SID-Set clean, limit_test=1000):

    accuracy 0.9093 | fd_auroc 0.8219 | aurc 0.0245 | det_auroc 0.9619 |
    det_syn 0.9977 | det_tam 0.9209 | rec_syn 0.9972 | rec_tam 0.8766

Exact equality is not expected (different val carve draw, different head
init); agreement within ~0.01-0.02 on accuracy/AUROCs validates the path.
A larger gap means the cache or the fit recipe is NOT the TB-E2 instrument
-- stop and find out why before running the ladder.
"""

from __future__ import annotations

import os

import pandas as pd

from trustfake.curation.cacheview import load_cache
from trustfake.curation.fit import evaluate_probe, fit_probe
from trustfake.curation.legs import sid_roles
from trustfake.data.manifest import DEFAULT_MANIFEST_SEED, PROFILES, assign_shards

REFERENCE = {
    "accuracy": 0.9093,
    "fd_auroc": 0.8219,
    "aurc": 0.0245,
    "detection_auroc": 0.9619,
    "detection_auroc_synthetic": 0.9977,
    "detection_auroc_tampered": 0.9209,
    "recall_synthetic": 0.9972,
    "recall_tampered": 0.8766,
}


def main() -> None:
    features_root = os.path.join(os.environ["OUTPUT_PATH"], "tb_e3", "features")
    index, features = load_cache(features_root, "sid_set")
    roles = sid_roles(index)

    fit_rows = index.loc[roles == "fit"]
    print(f"fit rows: {len(fit_rows)}")

    # The grid's limit_test=1000 prefix: test shards in the order assign_shards
    # returns them (validation permutation), first 1000 rows in shard row order.
    shards = index["shard"].unique().tolist()
    assignment = assign_shards(
        sorted(s for s in shards if s.startswith("train-")),
        sorted(s for s in shards if s.startswith("validation-")),
        PROFILES["train"],
        DEFAULT_MANIFEST_SEED,
    )
    test_frames = [
        index.loc[index["shard"] == shard].sort_values("row")
        for shard in assignment["test"]
    ]
    test_prefix = pd.concat(test_frames).head(1000)
    n_shards = test_prefix["shard"].nunique()
    print(f"test prefix: {len(test_prefix)} rows from {n_shards} shard(s)")

    result = fit_probe(
        features[fit_rows["cache_row"].to_numpy()],
        fit_rows["label3"].to_numpy(),
        seed=1,  # the grid ran experiment.seed default 1
    )
    metrics = evaluate_probe(
        result["state"],
        features[test_prefix["cache_row"].to_numpy()],
        test_prefix["label3"].to_numpy(),
    )

    print(f"\n{'metric':28s} {'tb_e3':>8s} {'tb_e2':>8s} {'delta':>8s}")
    for key, reference in REFERENCE.items():
        value = metrics[key]
        print(f"{key:28s} {value:8.4f} {reference:8.4f} {value - reference:+8.4f}")
    print(f"\nbest epoch {result['best_epoch']}, val f1 {result['val_f1']:.4f}")
    worst = max(abs(metrics[k] - v) for k, v in REFERENCE.items())
    verdict = "OK (within 0.02)" if worst <= 0.02 else "INVESTIGATE"
    print(f"max |delta| = {worst:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
