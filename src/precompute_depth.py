"""Precompute Track C depth targets with the frozen teacher, keyed to the
manifest.

Runs Depth Anything V2 (small) once over the fit shards of a profile and
stores per-image relative depth at the head's grid (112x112, float16), in
the shared zero-median / unit-MAD frame, under a store the datamodule reads
back by uid. Resumable per shard; refuses to extend a store computed under
different preprocessing.

Usage (on the 3090 box, from the repo root)::

    .venv/bin/python src/precompute_depth.py --profile train
    .venv/bin/python src/precompute_depth.py --profile all --batch 64
    .venv/bin/python src/precompute_depth.py --stub-teacher --out-dir /tmp/d  # smoke

The store directory defaults to `$DATA_PATH/sid_set_depth`; pass the same
path to training as `datamodule.datamodule.depth_targets_dir=...`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from trustfake.data.depth_targets import precompute_depth_targets
from trustfake.data.manifest import DEFAULT_MANIFEST_SEED, PROFILES
from trustfake.depth import (
    DEFAULT_DEPTH_SIZE,
    DEFAULT_TEACHER_INPUT_SIZE,
    DEPTH_ANYTHING_V2_SMALL,
    DEPTH_ANYTHING_V2_SMALL_REVISION,
    FakeDepthTeacher,
    load_depth_teacher,
)
from trustfake.logging import get_logger
from trustfake.utils import resolve_device

logger = get_logger("precompute-depth")


def _default(env: str, suffix: str) -> str | None:
    root = os.environ.get(env)
    return str(Path(root) / suffix) if root else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=_default("DATA_PATH", "sid_set"))
    ap.add_argument("--out-dir", default=_default("DATA_PATH", "sid_set_depth"))
    ap.add_argument("--profile", default="train", choices=sorted(PROFILES))
    ap.add_argument(
        "--roles",
        default="fit",
        help="comma-separated manifest roles to cover (default: fit; val is "
        "carved from fit at row level, calib/test never need targets)",
    )
    ap.add_argument("--manifest-seed", type=int, default=DEFAULT_MANIFEST_SEED)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--squarecrop", action="store_true")
    ap.add_argument(
        "--size",
        type=int,
        default=None,
        help="target grid; default image_size // 2, the head's own grid "
        f"({DEFAULT_DEPTH_SIZE} at 224)",
    )
    ap.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="write at most N shards this call (time a first run; resumable)",
    )
    ap.add_argument(
        "--teacher-input-size",
        type=int,
        default=DEFAULT_TEACHER_INPUT_SIZE,
        help="target of the teacher's resize rule (518 = the checkpoint's own)",
    )
    ap.add_argument("--hub-id", default=DEPTH_ANYTHING_V2_SMALL)
    ap.add_argument("--revision", default=DEPTH_ANYTHING_V2_SMALL_REVISION)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="default: the best available")
    ap.add_argument(
        "--stub-teacher",
        action="store_true",
        help="use the parameter-free stub teacher (smoke tests only; NEVER for "
        "a reported run)",
    )
    args = ap.parse_args(argv)

    if args.data_dir is None or args.out_dir is None:
        ap.error("--data-dir/--out-dir are required when DATA_PATH is not set")
    if args.size is None:
        args.size = args.image_size // 2
    if args.size <= 0 or args.batch <= 0 or args.image_size <= 0:
        ap.error("--size, --batch and --image-size must be positive")
    if args.size != args.image_size // 2:
        ap.error(
            f"--size {args.size} is not --image-size // 2 = {args.image_size // 2}; "
            "the head emits half the input grid and the loss does not resample"
        )

    device = args.device or resolve_device()
    if args.stub_teacher:
        logger.warning(
            "Using the STUB teacher: the targets are blurred luminance, not "
            "depth. Fine for a smoke test; a training run on them is not Track C."
        )
        teacher = FakeDepthTeacher(
            output_size=args.size, input_size=args.teacher_input_size, autocast=True
        )
    else:
        teacher = load_depth_teacher(
            args.hub_id,
            revision=args.revision,
            device=device,
            output_size=args.size,
            input_size=args.teacher_input_size,
            autocast=True,  # no-grad offline pass: fp16 is fine here
        )

    summary = precompute_depth_targets(
        args.data_dir,
        args.out_dir,
        teacher,
        profile=args.profile,
        roles=[r.strip() for r in args.roles.split(",") if r.strip()],
        manifest_seed=args.manifest_seed,
        image_size=args.image_size,
        squarecrop=args.squarecrop,
        batch_size=args.batch,
        device=device,
        limit_shards=args.limit_shards,
    )
    summary["out_dir"] = str(args.out_dir)
    summary["teacher"] = teacher.describe()
    print("SUMMARY_JSON")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
