"""Collate the Track C chain into readable tables.

Runs ON THE BOX with the system python (stdlib only). Each cell's log names
its own `test_lightning_logs/version_N`, so the mapping from cell to metrics
is exact -- which matters more here than in Track B: the three scorings of
one checkpoint log IDENTICAL metric keys (`nat_fd_auroc`, ...) into different
version dirs, and any collator that merges by key would let the last-written
score overwrite the others. Every cell is also checked against its own
`experiment_config.yaml`: a cell whose recorded uncertainty score does not
match the score part of its key is reported as a mismatch, never as a
number.

Cell key: <arm>__<dataset>__<cond>__<score>, score in
  msp | depth | combined | depth_tr | depth_wb | combined_tr | combined_wb
(`_tr` transfer / `_wb` white-box attack scoring; see jobs/track_c_depth.sh).

Usage:  python3 jobs/summarise_track_c.py [<chain-log-dir>] > RESULTS_track_c.md
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

COLUMNS = [
    ("accuracy", "acc"),
    ("fd_auroc", "Phi"),
    ("aurc", "aurc"),
    ("n_operating_points", "n_op"),
    ("recall_tampered", "rec_tam"),
    ("moderation_residual_risk", "resid_risk"),
    ("moderation_review_rate", "review"),
]
DATASET_LABEL = {
    "sid_set": "SID-Set (in-domain)",
    "so_fake_ood": "So-Fake-OOD (shift, in-domain thresholds)",
}
SCORE_TARGET = {
    "msp": "MultiClassMaxProbability",
    "depth": "DepthConsistencyScore",
    "depth_tr": "DepthConsistencyScore",
    "depth_wb": "DepthConsistencyScore",
    "combined": "CombinedDepthScore",
    "combined_tr": "CombinedDepthScore",
    "combined_wb": "CombinedDepthScore",
}
VERSION_RE = re.compile(r"(/[^\s]*test_lightning_logs/version_\d+)")
TARGET_RE = re.compile(r"_target_:\s*trustfake\.metrics\.uncertainty\.\w+\.(\w+)")
PIPE_RE = re.compile(r"training_pipe:\s*(\w+)")


def version_dir(cell_log: str) -> str | None:
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
    if condition == "clean":
        prefix = "nat"
    elif condition.startswith("corruption_"):
        # the corruption's logged name is what keys its metrics; find it
        prefix = next(
            (
                k.rsplit("_fd_auroc", 1)[0]
                for k in merged
                if k.endswith("_fd_auroc") and not k.startswith("nat_")
            ),
            condition,
        )
    else:
        prefix = condition
    out = {}
    for key, _ in COLUMNS:
        raw = merged.get(f"{prefix}_{key}")
        if raw not in (None, ""):
            try:
                out[key] = float(raw)
            except ValueError:
                pass
    return out


def recorded_score(version: str) -> str | None:
    path = os.path.join(version, "experiment_config.yaml")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        m = TARGET_RE.search(fh.read())
    return m.group(1) if m else None


def gate_for(version: str) -> dict:
    path = os.path.join(version, "depth_calib_gate.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def fmt(value: float | None) -> str:
    if value is None:
        return "--"
    if value != value:  # NaN
        return "NaN"
    return f"{value:.4f}" if abs(value) < 100 else f"{value:.0f}"


def main() -> None:
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    cells: dict[tuple[str, str, str, str], dict[str, float]] = {}
    gates: dict[tuple[str, str], dict] = {}
    problems: list[str] = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".log") or "__" not in name:
            continue
        parts = name[:-4].split("__")
        if len(parts) != 4:
            continue
        arm, dataset, cond, score = parts
        if not os.path.exists(os.path.join(log_dir, name[:-4] + ".done")):
            problems.append(f"{name[:-4]}: no .done marker (failed or still running)")
            continue
        version = version_dir(os.path.join(log_dir, name))
        if version is None:
            problems.append(f"{name[:-4]}: no version dir in its log")
            continue
        recorded = recorded_score(version)
        expected = SCORE_TARGET.get(score)
        if expected and recorded and recorded != expected:
            problems.append(
                f"{name[:-4]}: key says {score} ({expected}) but the run "
                f"recorded {recorded} -- NOT collated"
            )
            continue
        cells[(arm, dataset, cond, score)] = metrics_for(version, cond)
        gate = gate_for(version)
        if gate:
            gates[(arm, dataset)] = gate

    arms = sorted({k[0] for k in cells})
    datasets = [d for d in DATASET_LABEL if any(k[1] == d for k in cells)]
    conds = sorted({k[2] for k in cells}, key=lambda c: (c != "clean", c))
    scores = sorted({k[3] for k in cells}, key=lambda s: (s != "msp", s))

    print("# Track C -- auxiliary depth: results\n")
    print(f"Collated from `{os.path.abspath(log_dir)}`; {len(cells)} cells.\n")
    print(
        "Track C is its own table. Phi = fd_auroc (failure detection). `_tr` = "
        "transfer attack scoring, `_wb` = white-box (the attack optimises the "
        "depth score itself). Every depth-score number carries the caveat that "
        "a gradient adaptive attack on the residual is a follow-up.\n"
    )

    if gates:
        print("## Validity gate G2 (calib): is the residual just 1 - MSP relabelled?\n")
        print(
            "| arm | dataset | abs rho vs 1-MSP | residual mean | residual std "
            "| n | verdict |"
        )
        print("|---|---|---:|---:|---:|---:|---|")
        for (arm, dataset), g in sorted(gates.items()):
            rho = g.get("spearman_abs_residual_vs_msp")
            verdict = (
                "DEGENERATE"
                if rho is not None and rho >= g.get("degeneracy_threshold", 0.98)
                else "independent"
            )
            print(
                f"| {arm} | {dataset} | {fmt(rho)} | {fmt(g.get('residual_mean'))} | "
                f"{fmt(g.get('residual_std'))} | {g.get('n_calib', '--')} | {verdict} |"
            )
        print()

    for dataset in datasets:
        print(f"## {DATASET_LABEL[dataset]}\n")
        print("### Prediction axis: accuracy per condition (msp cells)\n")
        header = "| arm | " + " | ".join(conds) + " |"
        print(header)
        print("|---|" + "---:|" * len(conds))
        for arm in arms:
            row = [
                fmt(cells.get((arm, dataset, c, "msp"), {}).get("accuracy"))
                for c in conds
            ]
            print(f"| {arm} | " + " | ".join(row) + " |")
        print()
        print("### Confidence axis: Phi (fd_auroc), three scorings side by side\n")
        header = "| arm | cond | " + " | ".join(scores) + " |"
        print(header)
        print("|---|---|" + "---:|" * len(scores))
        for arm in arms:
            for cond in conds:
                row = [
                    fmt(cells.get((arm, dataset, cond, s), {}).get("fd_auroc"))
                    for s in scores
                ]
                if all(r == "--" for r in row):
                    continue
                print(f"| {arm} | {cond} | " + " | ".join(row) + " |")
        print()
        print("### Selective / moderation columns, per cell\n")
        print(
            "| arm | cond | score | " + " | ".join(label for _, label in COLUMNS) + " |"
        )
        print("|---|---|---|" + "---:|" * len(COLUMNS))
        for arm in arms:
            for cond in conds:
                for score in scores:
                    m = cells.get((arm, dataset, cond, score))
                    if m is None:
                        continue
                    print(
                        f"| {arm} | {cond} | {score} | "
                        + " | ".join(fmt(m.get(k)) for k, _ in COLUMNS)
                        + " |"
                    )
        print()

    if problems:
        print("## Not collated\n")
        for p in problems:
            print(f"- {p}")
        print()

    print("## What is NOT readable here\n")
    print(
        "- Depth-score robustness against a GRADIENT adaptive attack on the "
        "residual (follow-up); `_wb` covers the gradient-free query attacks and "
        "the gradient confidence attacks through the teacher only.\n"
        "- Anything about Track A/B arms: different fits, separate tables.\n"
        "- A cell whose G2 verdict is DEGENERATE: its depth score is MSP "
        "relabelled, whatever its Phi."
    )


if __name__ == "__main__":
    main()
