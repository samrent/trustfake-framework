"""Hardening column: FARE4-ViT-B/16 re-embed of the winning arms + legs, refit.

Re-embeds ONLY the uids the comparison needs (the C3a arms' union + every
frozen leg) under the adversarially fine-tuned encoder
(hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K, eps=4/255 -- the strongest
published robust CLIP at this size; the project's 8/255 budget exceeds it,
label any attacked numbers as beyond-training-budget). Then refits the
probe per seed and evaluates the same frozen legs: the curation x
robustification cell of the follow-on grid.

Output: $OUTPUT_PATH/tb_e3/features_fare/ + tb_e3/hardening/fare_*.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pyarrow.parquet as pq

from trustfake.curation.cacheview import gather_features, load_cache
from trustfake.curation.embed import embed_dataset
from trustfake.curation.fit import evaluate_probe, fit_probe
from trustfake.curation.ladder import SEEDS, _out
from trustfake.curation.legs import load_legs

OUT = os.environ["OUTPUT_PATH"]
DATA = os.environ["DATA_PATH"]
FARE_ROOT = os.path.join(OUT, "tb_e3", "features_fare")
FARE_TAG = "hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K"


def main() -> None:
    out = _out(OUT) / "hardening"
    out.mkdir(parents=True, exist_ok=True)

    _, leg_frames = load_legs(_out(OUT) / "legs")
    arms = {
        seed: pq.read_table(
            _out(OUT) / "arms" / f"C3a_natural_s{seed}.parquet"
        ).to_pandas()
        for seed in SEEDS
    }
    wanted: set[str] = set()
    for frame in leg_frames.values():
        wanted.update(frame["uid"])
    for arm in arms.values():
        wanted.update(arm["uid"])
    print(f"uids to re-embed under FARE: {len(wanted)}")

    def dataset_of(uid: str) -> str:
        head = uid.split(":", 1)[0]
        return {"community_forensics": "community_forensics_small"}.get(head, head)

    datasets = sorted({dataset_of(u) for u in wanted})
    for dataset in datasets:
        embed_dataset(
            dataset,
            data_dir=os.path.join(DATA, dataset),
            out_dir=FARE_ROOT,
            model_name=FARE_TAG,
            pretrained="",
            uid_filter=wanted,
        )

    caches = {d: load_cache(FARE_ROOT, d) for d in datasets}

    def gather(uids, labels_frame):
        parts_f, parts_y = [], []
        for dataset, rows in labels_frame.groupby(labels_frame["uid"].map(dataset_of)):
            index, features = caches[dataset]
            parts_f.append(gather_features(index, features, rows["uid"]))
            parts_y.append(rows["label3"].to_numpy())
        return np.concatenate(parts_f), np.concatenate(parts_y)

    evaluation = {
        leg: gather(frame["uid"], frame)
        for leg, frame in leg_frames.items()
        if leg != "calib" and not frame.empty
    }

    for seed in SEEDS:
        marker = out / f"fare_c3a_s{seed}.json"
        if marker.exists():
            print(f"SKIP fare s{seed}")
            continue
        x, y = gather(arms[seed]["uid"], arms[seed])
        result = fit_probe(x, y, seed=seed)
        import torch

        torch.save(result["state"], out / f"fare_c3a_s{seed}.pt")
        metrics = {
            leg: evaluate_probe(result["state"], f, labels)
            for leg, (f, labels) in evaluation.items()
        }
        marker.write_text(
            json.dumps(
                {"encoder": FARE_TAG, "seed": seed, "n_rows": int(len(arms[seed])),
                 "val_f1": result["val_f1"], "legs": metrics},
                indent=2,
            )
        )
        summary = " ".join(
            f"{leg}={m['detection_auroc']:.4f}" for leg, m in metrics.items()
        )
        print(f"fare_c3a s{seed}: {summary}")
    print("HARDEN_FARE_COMPLETE")


if __name__ == "__main__":
    main()
