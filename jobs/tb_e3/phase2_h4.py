"""TB-E3 phase 2 (H4) runner: GroupDRO + V-REx on the winning arm (C3a).

Rebuilds the C3a natural arm per seed from the saved arm parquets, fits the
two invariance objectives, evaluates on the frozen legs, and prints the H4
comparison against the ladder's own pooled-ERM (C3a) fits. Results land in
$OUTPUT_PATH/tb_e3/phase2/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from trustfake.curation.cacheview import gather_features, load_cache
from trustfake.curation.fit import evaluate_probe
from trustfake.curation.ladder import SEEDS, _leg_eval_sets, _out
from trustfake.curation.phase2 import fit_probe_grouped

OUT = os.environ["OUTPUT_PATH"]
FEATURES_ROOT = os.path.join(OUT, "tb_e3", "features")


def main() -> None:
    out = _out(OUT) / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    evaluation = _leg_eval_sets(FEATURES_ROOT, OUT)
    caches: dict[str, tuple] = {}

    for seed in SEEDS:
        arm_file = _out(OUT) / "arms" / f"C3a_natural_s{seed}.parquet"
        arm = pq.read_table(arm_file).to_pandas()
        parts_f, parts_y, parts_g = [], [], []
        for dataset, rows in arm.groupby("dataset"):
            if dataset not in caches:
                caches[dataset] = load_cache(FEATURES_ROOT, dataset)
            index, features = caches[dataset]
            parts_f.append(gather_features(index, features, rows["uid"]))
            parts_y.append(rows["label3"].to_numpy())
            parts_g.append(rows["dataset"].to_numpy())
        x = np.concatenate(parts_f)
        y = np.concatenate(parts_y)
        g = np.concatenate(parts_g)

        for objective in ("groupdro", "vrex"):
            marker = out / f"{objective}_s{seed}.json"
            if marker.exists():
                print(f"SKIP {objective} s{seed}")
                continue
            result = fit_probe_grouped(x, y, g, seed=seed, objective=objective)
            import torch

            torch.save(result["state"], out / f"{objective}_s{seed}.pt")
            metrics = {
                leg: evaluate_probe(result["state"], f, labels)
                for leg, (f, labels) in evaluation.items()
            }
            marker.write_text(
                json.dumps(
                    {
                        "objective": objective,
                        "seed": seed,
                        "n_rows": int(len(arm)),
                        "val_f1": result["val_f1"],
                        "legs": metrics,
                    },
                    indent=2,
                )
            )
            print(
                f"{objective} s{seed}: "
                + " ".join(
                    f"{leg}={m['detection_auroc']:.4f}" for leg, m in metrics.items()
                )
            )

    # H4 comparison: ERM = the ladder's own C3a natural fits.
    erm = [
        json.loads((_out(OUT) / "fits" / f"C3a_natural_s{s}.json").read_text())
        for s in SEEDS
    ]
    rows = []
    for name, records in [
        ("pooled_erm", erm),
        *[
            (
                obj,
                [
                    json.loads((out / f"{obj}_s{s}.json").read_text())
                    for s in SEEDS
                ],
            )
            for obj in ("groupdro", "vrex")
        ],
    ]:
        means = {
            leg: float(np.mean([r["legs"][leg]["detection_auroc"] for r in records]))
            for leg in ("L1", "L2", "L3", "L4")
        }
        rows.append((name, means))
        print(f"{name:12s} " + " ".join(f"{leg}={v:.4f}" for leg, v in means.items()))

    erm_means = rows[0][1]
    for name, means in rows[1:]:
        beats = means["L3"] > erm_means["L3"] and means["L4"] > erm_means["L4"]
        l1_ok = means["L1"] >= erm_means["L1"] - 0.01
        print(
            f"H4 {name}: beats ERM on L3+L4 = {beats}, L1 within 0.01 = {l1_ok} "
            f"-> {'ADOPT' if beats and l1_ok else 'not adopted'}"
        )
    Path(out / "h4_summary.json").write_text(
        json.dumps({name: means for name, means in rows}, indent=2)
    )


if __name__ == "__main__":
    main()
