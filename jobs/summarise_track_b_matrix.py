"""Collate the Track B matrix into one readable two-axis table.

Runs ON THE BOX, so the results survive the laptop going away: the matrix is
detached (reparented to init) and this collator is too, which means the whole
experiment reaches a readable conclusion with nothing connected to it.

Each cell's log names its own `test_lightning_logs/version_N`, so the mapping
from cell to metrics is exact rather than inferred from run order or mtimes.

Usage:  python3 jobs/summarise_track_b_matrix.py [<matrix-log-dir>] > RESULTS.md
"""

from __future__ import annotations

import csv
import os
import re
import sys

# accuracy is the PREDICTION axis; fd_auroc is the CONFIDENCE axis. An attack
# that leaves the first bit-identical while moving the second is the failure
# this project exists to measure, and an accuracy-only table cannot see it.
COLUMNS = [
    ("accuracy", "acc"),
    ("fd_auroc", "Phi"),
    ("aurc", "aurc"),
    ("detection_auroc", "det"),
    ("detection_auroc_synthetic", "det_syn"),
    ("detection_auroc_tampered", "det_tam"),
    ("recall_synthetic", "rec_syn"),
    ("recall_tampered", "rec_tam"),
]
DATASETS = ["sid_set", "fake_clue", "so_fake_ood"]
DATASET_LABEL = {
    "sid_set": "SID-Set (in-domain)",
    "fake_clue": "FakeClue (cross-dataset, binary fold)",
    "so_fake_ood": "So-Fake-OOD (shift, in-domain thresholds)",
}
CONDITIONS = ["clean", "pgd", "query_overconf", "query_underconf"]
VERSION_RE = re.compile(r"(/[^\s]*test_lightning_logs/version_\d+)")


def version_dir(cell_log: str) -> str | None:
    """The version directory this cell wrote, taken from its own log."""
    found = None
    with open(cell_log, errors="replace") as fh:
        for line in fh:
            m = VERSION_RE.search(line)
            if m:
                found = m.group(1)
    return found


def metrics_for(version: str, condition: str) -> dict[str, float]:
    path = os.path.join(version, "metrics.csv")
    if not os.path.exists(path):
        return {}
    merged: dict[str, str] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                if v not in ("", None):
                    merged[k] = v
    # A clean cell logs its metrics under the `nat_` prefix; an attacked cell
    # logs BOTH `nat_` and `<attack>_`, and we want the attacked column.
    prefix = "nat" if condition == "clean" else condition
    out = {}
    for key, _ in COLUMNS:
        raw = merged.get(f"{prefix}_{key}")
        if raw not in (None, ""):
            try:
                out[key] = float(raw)
            except ValueError:
                pass
    return out


def main() -> None:
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    cells: dict[tuple[str, str, str], dict[str, float]] = {}
    missing: list[str] = []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or "__" not in fname:
            continue
        stem = fname[: -len(".log")]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        model, dataset, cond = parts
        if not os.path.exists(os.path.join(log_dir, f"{stem}.done")):
            missing.append(stem)
            continue
        version = version_dir(os.path.join(log_dir, fname))
        if version is None:
            missing.append(stem + " (no version dir in log)")
            continue
        cells[(model, dataset, cond)] = metrics_for(version, cond)

    print("# Track B matrix — the two axes, three datasets\n")
    print(
        "`acc` is the PREDICTION axis, `Phi` (fd_auroc) is the CONFIDENCE axis. "
        "A confidence attack that leaves `acc` unchanged while `Phi` falls is "
        "the point; an accuracy-only table is blind to it.\n"
    )
    print(f"Cells collated: {len(cells)}/24. Missing: {len(missing)}\n")

    for dataset in DATASETS:
        print(f"\n## {DATASET_LABEL[dataset]}\n")
        head = "| model | condition | " + " | ".join(lbl for _, lbl in COLUMNS) + " |"
        print(head)
        print("|" + "---|" * (len(COLUMNS) + 2))
        for model in ("clip_probe", "resnet"):
            for cond in CONDITIONS:
                m = cells.get((model, dataset, cond))
                if m is None:
                    continue
                vals = " | ".join(f"{m[k]:.4f}" if k in m else "—" for k, _ in COLUMNS)
                print(f"| {model} | {cond} | {vals} |")

    if missing:
        print("\n## Not collated\n")
        for m in sorted(missing):
            print(f"- {m}")

    # The single comparison the array exists to make.
    print("\n## Confidence axis: does accuracy hold while Phi moves?\n")
    print("| model | dataset | acc clean -> attacked | Phi clean -> attacked |")
    print("|---|---|---|---|")
    for model in ("clip_probe", "resnet"):
        for dataset in DATASETS:
            c = cells.get((model, dataset, "clean"))
            a = cells.get((model, dataset, "query_overconf"))
            if not c or not a:
                continue

            # c/a passed explicitly rather than closed over: a closure over a
            # loop variable is a late-binding bug waiting for someone to move
            # the call out of the iteration.
            def fmt(key: str, clean=c, attacked=a) -> str:
                if key not in clean or key not in attacked:
                    return "—"
                return f"{clean[key]:.4f} -> {attacked[key]:.4f}"

            print(f"| {model} | {dataset} | {fmt('accuracy')} | {fmt('fd_auroc')} |")


if __name__ == "__main__":
    main()
