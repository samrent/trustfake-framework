"""TB-E3 curation CLI -- thin argparse front over trustfake.curation.

Deliberately not Hydra: the curation pipeline is embed-once + index-list
arms, a different workflow from the train/test configs, and its protocol
constants live in the spec (documentation/specs/tb-e3-curation-ladder.md)
rather than in a config tree.

    .venv/bin/python src/curate.py embed --dataset sid_set
"""

from __future__ import annotations

import argparse
import os

from trustfake.curation.embed import embed_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="embed one dataset into the feature cache")
    embed.add_argument("--dataset", required=True)
    embed.add_argument("--data-dir", default=None, help="default: $DATA_PATH/<dataset>")
    embed.add_argument("--out", default=None, help="default: $OUTPUT_PATH/tb_e3/features")
    embed.add_argument("--batch-size", type=int, default=128)
    embed.add_argument("--num-workers", type=int, default=6)
    embed.add_argument("--limit-shards", type=int, default=None)
    embed.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.command == "embed":
        data_dir = args.data_dir or os.path.join(os.environ["DATA_PATH"], args.dataset)
        out = args.out or os.path.join(os.environ["OUTPUT_PATH"], "tb_e3", "features")
        embed_dataset(
            args.dataset,
            data_dir=data_dir,
            out_dir=out,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit_shards=args.limit_shards,
            device=args.device,
        )


if __name__ == "__main__":
    main()
