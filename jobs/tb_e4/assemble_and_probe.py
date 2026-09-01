"""TB-E4 stages 3-4: assemble C2-prime, freeze dev legs, gates, Arm A.

Composition = C3a's synthetic side (verbatim from the TB-E3 arm parquets,
minus dev-L3 holdouts) + the QF-85 tampered pairs (dev-L4 tool excluded at
enumeration). Gates re-run on the assembled arm; an environment failing G1
after the re-encode is EXCLUDED and recorded (spec: fixed-not-trained).
Arm A = 3 probe fits, evaluated once on the frozen legs; H5 applied against
the stored C3a results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from trustfake.curation.cacheview import load_cache
from trustfake.curation.fit import evaluate_probe, fit_probe
from trustfake.curation.gates import (
    G2_COSINE_THRESHOLD,
    g1_metadata_auroc,
    g3_cell_counts,
)
from trustfake.curation.ladder import SEEDS, _leg_eval_sets

OUT = os.environ["OUTPUT_PATH"]
E3 = Path(OUT) / "tb_e3"
E4 = Path(OUT) / "tb_e4"
FEATURES_E3 = str(E3 / "features")
DEV_SEED = 20260902
SYNTH_ENVS = ("community_forensics_small", "synthbuster", "vision", "sid_set")


def freeze_dev_legs(cf_index, tgif_index):
    legs_dir = E4 / "legs_dev"
    legs_dir.mkdir(parents=True, exist_ok=True)
    manifest = legs_dir / "dev_legs.json"
    if manifest.exists():
        return
    giga = cf_index.loc[
        cf_index["generator"].fillna("").str.contains("GigaGAN", case=False)
    ]
    reals = cf_index.loc[cf_index["label3"] == 0]
    rng = np.random.default_rng(DEV_SEED)
    reserve = reals.iloc[
        np.sort(rng.choice(len(reals), size=2000, replace=False))
    ]
    dev_l3 = pd.concat([giga, reserve], ignore_index=True)
    fill = tgif_index.loc[
        tgif_index["tool"].astype(str).str.startswith("flux1filldev")
    ]
    negs = tgif_index.loc[
        (tgif_index["tool"] == "orig") & (tgif_index["source_split"] == "validation")
    ]
    dev_l4 = pd.concat([fill, negs], ignore_index=True)
    for name, frame in (("dev_L3", dev_l3), ("dev_L4", dev_l4)):
        pq.write_table(
            pa.Table.from_pandas(frame[["uid", "label3"]], preserve_index=False),
            legs_dir / f"{name}.parquet",
        )
    manifest.write_text(json.dumps({
        "dev_L3": {"n": len(dev_l3),
                   "definition": "CF GigaGAN + 2k seeded CF reals"},
        "dev_L4": {"n": len(dev_l4),
                   "definition": "TGIF flux1filldev + orig validation"},
        "seed": DEV_SEED,
    }, indent=2))
    print("dev legs frozen:", len(dev_l3), len(dev_l4))


def main():
    E4.mkdir(parents=True, exist_ok=True)
    caches = {env: load_cache(FEATURES_E3, env) for env in SYNTH_ENVS + ("tgif",)}
    pairs_index, pairs_features = load_cache(str(E4 / "features_qf85"), "tb_e4_pairs")
    pairs_index["dataset"] = pairs_index["env"]

    freeze_dev_legs(caches["community_forensics_small"][0], caches["tgif"][0])
    dev = {
        name: pq.read_table(E4 / "legs_dev" / f"{name}.parquet").to_pandas()
        for name in ("dev_L3", "dev_L4")
    }
    held_uids = set(dev["dev_L3"]["uid"]) | set(dev["dev_L4"]["uid"])

    # --- synthetic side from C3a, minus dev holdouts; features from e3 cache
    synth_frames, synth_feats = [], []
    arm3a = pq.read_table(E3 / "arms" / "C3a_natural_s1.parquet").to_pandas()
    synth_rows = arm3a[arm3a["dataset"].isin(SYNTH_ENVS)]
    synth_rows = synth_rows[~synth_rows["uid"].isin(held_uids)]
    for env, rows in synth_rows.groupby("dataset"):
        index, feats = caches[env]
        joined = index.set_index("uid").loc[rows["uid"]]
        synth_frames.append(joined.reset_index())
        synth_feats.append(feats[joined["cache_row"].to_numpy()])
    synth_index = pd.concat(synth_frames, ignore_index=True)
    synth_features = np.concatenate(synth_feats)

    arm_index = pd.concat(
        [synth_index, pairs_index.drop(columns=["cache_row"]).assign(
            cache_row=pairs_index["cache_row"])],
        ignore_index=True,
    )
    arm_features = np.concatenate([synth_features, pairs_features])
    print("assembled:", arm_index.groupby(["dataset", "label3"]).size())

    # --- G2 vs frozen legs (content level, both sides standard-B/16 space)
    evaluation = _leg_eval_sets(FEATURES_E3, OUT)
    leg_features = np.concatenate([f for f, _ in evaluation.values()])
    from trustfake.curation.gates import g2_max_cosine

    best = g2_max_cosine(arm_features.astype(np.float16), leg_features)
    keep = best < G2_COSINE_THRESHOLD
    print(f"G2: dropped {int((~keep).sum())} of {len(arm_index)}")
    arm_index = arm_index.loc[keep].reset_index(drop=True)
    arm_features = arm_features[keep]

    # --- G1 per environment; failures EXCLUDE the environment (recorded)
    g1 = g1_metadata_auroc(arm_index)
    print("G1:", {k: round(v, 3) if v == v else "nan" for k, v in g1.items()})
    excluded = [env for env, v in g1.items() if v == v and v >= 0.55]
    if excluded:
        print("G1 EXCLUSIONS:", excluded)
        keep_env = ~arm_index["dataset"].isin(excluded)
        arm_index = arm_index.loc[keep_env].reset_index(drop=True)
        arm_features = arm_features[keep_env.to_numpy()]
    g3 = g3_cell_counts(arm_index)

    arm_dir = E4 / "arm"
    arm_dir.mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(
            arm_index.drop(columns=["cache_row"]), preserve_index=False
        ),
        arm_dir / "c2prime.parquet",
    )
    (arm_dir / "gates.json").write_text(json.dumps({
        "g1": g1, "g1_excluded": excluded,
        "g3_undersized": int(g3["undersized"].sum()),
        "n": len(arm_index),
    }, indent=2, default=str))

    # --- Arm A: 3 fits, frozen-leg eval, H5
    fits_dir = E4 / "fits"
    fits_dir.mkdir(exist_ok=True)
    labels = arm_index["label3"].to_numpy()
    for seed in SEEDS:
        marker = fits_dir / f"armA_s{seed}.json"
        if marker.exists():
            continue
        result = fit_probe(arm_features, labels, seed=seed)
        torch.save(result["state"], fits_dir / f"armA_s{seed}.pt")
        metrics = {
            leg: evaluate_probe(result["state"], f, y)
            for leg, (f, y) in evaluation.items()
        }
        marker.write_text(json.dumps(
            {"seed": seed, "n_rows": len(arm_index), "val_f1": result["val_f1"],
             "legs": metrics}, indent=2))
        print(f"armA s{seed}: " + " ".join(
            f"{leg}={m['detection_auroc']:.4f}" for leg, m in metrics.items()))

    # --- H5 against stored C3a
    def leg_stats(records, leg):
        vals = np.array([r["legs"][leg]["detection_auroc"] for r in records])
        return vals.mean(), vals.std(ddof=1)

    arm_a = [json.loads((fits_dir / f"armA_s{s}.json").read_text()) for s in SEEDS]
    c3a = [json.loads((E3 / "fits" / f"C3a_natural_s{s}.json").read_text())
           for s in SEEDS]
    verdict = {}
    for leg in ("L1", "L2", "L3", "L4"):
        ma, sa = leg_stats(arm_a, leg)
        mc, sc = leg_stats(c3a, leg)
        verdict[leg] = {"armA": round(ma, 4), "c3a": round(mc, 4),
                        "delta": round(ma - mc, 4),
                        "two_sd": round(2 * float(np.sqrt((sa**2 + sc**2) / 2)), 4)}
    wins_l4 = verdict["L4"]["delta"] > verdict["L4"]["two_sd"]
    holds = all(verdict[leg]["delta"] > -0.01 for leg in ("L2", "L3"))
    verdict["H5"] = (
        "ADOPT pair-matching (wins L4 beyond noise, L2/L3 held)"
        if wins_l4 and holds else
        f"not adopted (wins_L4={wins_l4}, L2/L3_held={holds})"
    )
    (E4 / "h5_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    print("ASSEMBLE_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
