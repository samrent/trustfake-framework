"""CLI: compute the trivial metadata baselines for a SID-Set split.

python src/baselines.py                             # test split, 'full' profile
python src/baselines.py --role calib --profile smoke
python src/baselines.py --condition squarecrop      # geometry-controlled wording
python src/baselines.py --geometry-filter matched   # baselines ON that subset

`--geometry-filter` and `--condition` describe the same evaluation, from the
two ends: the filter says which rows these numbers were computed on, the
condition says which protocol they will be printed beside. They must agree.
A row filter changes the class prior and therefore every floor here -- the
majority-class floor on SID-Set's `nonsquare` subset is 1.0000, not the raw
split's 0.6657 -- so printing a raw floor next to a filtered accuracy is how
a model that beat nothing comes to look like it cleared a bar. The two flags
are reconciled here rather than left to the caller: `--condition` defaults to
the filter that was applied, and a pair that disagrees is refused up front
instead of surfacing as a traceback out of `headline`.
"""

import argparse
import json
import os

from trustfake.data.baselines import compute_trivial_baselines, headline
from trustfake.data.manifest import GEOMETRY_FILTERS
from trustfake.logging import get_logger

logger = get_logger("baselines-cli")

#: The geometry filters that change WHICH rows are reported, i.e. everything
#: but 'none'. Derived from the manifest's list rather than restated, so a
#: new mode cannot be added there and silently miss the CLI.
ROW_FILTERS: tuple[str, ...] = tuple(m for m in GEOMETRY_FILTERS if m != "none")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help="defaults to $DATA_PATH/sid_set")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--role", default="test")
    ap.add_argument("--real-class", type=int, default=0)
    ap.add_argument(
        "--geometry-filter",
        default="none",
        choices=list(GEOMETRY_FILTERS),
        help=(
            "row-level geometry control the baselines are computed ON. MUST "
            "match the SIDSetDataModule(geometry_filter=...) the model was "
            "evaluated under: a row filter changes the class prior, so it "
            "changes every floor printed here. The unfiltered numbers stay "
            "available under the 'raw' key."
        ),
    )
    ap.add_argument(
        "--condition",
        default=None,
        help=(
            "evaluation protocol these baselines will be printed beside "
            "(e.g. 'squarecrop', or a geometry_filter mode). Under a geometry "
            "control the raw 'width==height' accuracy misstates the protocol, "
            "so the headline says so instead of printing it bare. Defaults to "
            "--geometry-filter when one is set."
        ),
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # A row-filter condition is only meaningful against baselines computed
    # under that same filter, so reconcile the two flags before any work is
    # done. `headline` refuses the mismatch too, but after the shards have
    # been read -- and an argparse error names the flag to change.
    condition = a.condition
    if condition is None and a.geometry_filter != "none":
        condition = a.geometry_filter
    elif (
        condition is not None
        and condition.lower() in ROW_FILTERS
        and condition.lower() != a.geometry_filter
    ):
        ap.error(
            f"--condition {condition!r} is a row filter, so the baselines must "
            f"be computed on that same subset, but --geometry-filter is "
            f"{a.geometry_filter!r}. Pass --geometry-filter "
            f"{condition.lower()} (or drop --condition, which then defaults "
            "to the filter)."
        )

    data_dir = a.data_dir or os.path.join(os.environ["DATA_PATH"], "sid_set")
    baselines = compute_trivial_baselines(
        data_dir,
        profile=a.profile,
        split_role=a.role,
        real_class=a.real_class,
        geometry_filter=a.geometry_filter,
    )
    print(json.dumps(baselines, indent=2))
    print("\n" + headline(baselines, condition=condition))
    if a.out:
        with open(a.out, "w") as f:
            f.write(json.dumps(baselines, indent=2) + "\n")
        logger.info(f"written: {a.out}")


if __name__ == "__main__":
    main()
