"""CLI: compute the trivial metadata baselines for a SID-Set split.

python src/baselines.py            # test split, 'full' profile
python src/baselines.py --role calib --profile smoke
python src/baselines.py --condition squarecrop   # geometry-controlled wording
"""

import argparse
import json
import os

from trustfake.data.baselines import compute_trivial_baselines, headline
from trustfake.logging import get_logger

logger = get_logger("baselines-cli")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="defaults to $DATA_PATH/sid_set")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--role", default="test")
    ap.add_argument("--real-class", type=int, default=0)
    ap.add_argument(
        "--condition",
        default=None,
        help=(
            "evaluation protocol these baselines will be printed beside "
            "(e.g. 'squarecrop', or a geometry_filter mode). Under a geometry "
            "control the raw 'width==height' accuracy misstates the protocol, "
            "so the headline says so instead of printing it bare."
        ),
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data_dir = a.data_dir or os.path.join(os.environ["DATA_PATH"], "sid_set")
    baselines = compute_trivial_baselines(
        data_dir, profile=a.profile, split_role=a.role, real_class=a.real_class
    )
    print(json.dumps(baselines, indent=2))
    print("\n" + headline(baselines, condition=a.condition))
    if a.out:
        with open(a.out, "w") as f:
            f.write(json.dumps(baselines, indent=2) + "\n")
        logger.info(f"written: {a.out}")


if __name__ == "__main__":
    main()
