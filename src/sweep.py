"""Robustness-training sweep, ranked on CONFIDENCE resilience.

Three grids, from `SWEEP-STRATEGIES.md`:

    A  the knobs of the methods that already exist (standard / PGD-AT /
       TRADES). No new training code. It is the honest baseline: you cannot
       claim a hybrid or a new defence beats tuning until you have tuned.
    B  stacked / hybrid defences (AT + consistency KL). That document
       recommends SKIPPING it -- carefully-combined defences frequently do
       not beat a well-tuned single method, and the famous ones dissolved
       under adaptive attack. Implemented here so the recommendation can be
       tested rather than assumed.
    C  the confidence-targeted defences (at_conf, conf_reg). The only arms
       aimed at the failure this project measures, and the novel
       contribution.

**The ranking objective is not robust accuracy.** Robust accuracy under
PGD/AutoAttack is the crowded axis this project cannot out-compete. The
confidence axis is the thesis, and it is what keeps residual risk low under
attack -- so configurations are ranked by how well failure-detection AUROC
SURVIVES the confidence attacks, subject to a clean-accuracy floor. A
configuration that keeps confidence trustworthy but cannot classify is not a
winner, which is what the floor is for.

Sweep numbers RANK; they do not report. The `sweep` profile carves its own
small calib/test, so the winner must be re-run at `train` for anything
quotable. That separation is also what stops a 20-configuration search from
quietly becoming 20 attempts at the reported test split.

Usage:
    python src/sweep.py --strategy A --epochs 8
    python src/sweep.py --strategy AC --epochs 8      # A then C
    python src/sweep.py --rank-only                    # re-rank what exists
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The cheap conditions. (Both groups are scored on the same capped
#: split while conditions are being chosen -- see SUBSAMPLE_ROWS.)
#:
#: `ace_uint8` and `overconf` are the objective. `underconf` is the mirror of
#: over-confidence and was implemented but never once run in wp1. `pgd` is the
#: prediction-axis control and is not optional: the whole claim is that a
#: confidence attack leaves accuracy bit-identical while destroying selective
#: risk, and without an adjacent attack that DOES wreck accuracy, half that
#: sentence has no evidence next to it. `jpeg` is the realistic-perturbation
#: rung the proposal names outright ("compression, resizing, re-encoding").
FULL_SPLIT_CONDITIONS = (
    "clean",
    "pgd",
    "ace_uint8",
    "overconf",
    "underconf",
    # The REALISTIC axis, one condition per perturbation class the proposal
    # names verbatim -- "compression, resizing, re-encoding". The proposal's
    # evaluation triad is clean / realistic / adversarial; with a single
    # corruption the middle axis was a token rather than an axis.
    "+corruption=jpeg",
    "+corruption=downscale",
    "+corruption=webp",
)

#: The expensive conditions -- these are what make the cap necessary.
#:
#: AutoAttack and Square are run-once-at-the-end jobs in their own right --
#: Square at 500 queries is ~7h per configuration on the full split. The four
#: minimum-norm attacks answer a different question from the fixed-budget
#: ones: not "does the model survive at eps" but "how little does it take",
#: which is the quantity a robustness curve should be read against. Capping
#: the split is what makes them affordable; nesting is what keeps them
#: comparable to the rows above.
SUBSAMPLE_CONDITIONS = (
    "autoattack",
    "square",
    "deepfool",
    "cw",
    "bb",
    "pdpgd",
)

#: Rows every condition is scored on, cheap and expensive alike.
#:
#: One cap rather than two tiers, while the conditions themselves are still
#: being chosen: a uniform n means every column of the table describes the
#: same 1000 images, so nothing needs qualifying when the columns are read
#: side by side, and the whole stack finishes fast enough to iterate on.
#:
#: The cost is precision. At n=1000 a rate carries roughly +/-1.5 points of
#: binomial standard error, so small gaps between arms are NOT readable yet
#: -- this size is for choosing conditions, not for reporting results. The
#: cap is a prefix, so raising it later yields a superset and the numbers
#: stay comparable across the change.
SUBSAMPLE_ROWS = 1000

SCORING_CONDITIONS = FULL_SPLIT_CONDITIONS + SUBSAMPLE_CONDITIONS


#: THE budget. Every experiment is pinned here -- 8/255 is Madry's setting and
#: RobustBench's headline column, so it is the number that makes a result
#: comparable to the field rather than only to itself.
#:
#: The forensic argument cuts the other way and is on the record: a ball this
#: wide erases the small-amplitude high-frequency evidence a deepfake detector
#: reads, and wp1 recorded training collapsing onto a constant output here.
#: Pinning means that argument is now something the runs TEST rather than
#: something the design assumes -- so `rank` flags collapse explicitly
#: (see `_collapse_flag`) instead of letting a degenerate model report a
#: respectable-looking accuracy at the majority class.
EPS = 8 / 255

#: Inner-loop steps for every adversarial arm. Fixed, not swept: on the
#: measured 1-4/255 ladder, 3 vs 7 steps moved confidence resilience by less
#: than one seed's noise (0.696 vs 0.699 at 1/255), so the knob does not earn
#: a dimension.
STEPS = 7


def grid_a(epochs: int) -> list[dict]:
    """The arms that already exist, one configuration each at EPS."""
    return [
        {"name": "e8_standard", "pipe": "standard", "adv_eps": EPS},
        {"name": "e8_pgd_at", "pipe": "pgd_at", "adv_eps": EPS, "adv_steps": STEPS},
        {
            "name": "e8_trades",
            "pipe": "trades",
            "adv_eps": EPS,
            "adv_steps": STEPS,
            "trades_beta": 6.0,
        },
    ]


def grid_b(epochs: int) -> list[dict]:
    """Strategy B: the AT + consistency-KL hybrid. Recommended to skip."""
    return [
        {
            "name": "e8_at_kl",
            "pipe": "at_kl",
            "adv_eps": EPS,
            "adv_steps": STEPS,
            "at_kl_beta": 6.0,
        },
        {
            "name": "e8_mart",
            "pipe": "mart",
            "adv_eps": EPS,
            "adv_steps": STEPS,
            "mart_beta": 6.0,
        },
    ]


def grid_c(epochs: int) -> list[dict]:
    """Strategy C: the confidence-targeted defences."""
    return [
        {"name": "e8_at_conf", "pipe": "at_conf", "adv_eps": EPS, "adv_steps": STEPS},
        {
            "name": "e8_conf_reg",
            "pipe": "conf_reg",
            "adv_eps": EPS,
            "lambda_reg": 1.0,
        },
    ]


#: Config group needed by any arm with an evidential head.
_EVIDENTIAL = {
    "wrapper": "evidential",
    "loss": "evidential",
    "uncertainty_score": "evidential_predictive_entropy",
}


def grid_e(epochs: int) -> list[dict]:
    """EV-AT and its ablation ladder -- the PI's own method, decomposed.

    The headline row is `ev_at`. The rest exist because "does EV-AT help" is
    a far less useful answer than "which part of it does the work", and that
    second question is the one an independent harness can ask that a method's
    own authors cannot easily ask of their own paper.

    The ladder isolates one component per rung:

        ev_only        evidential head, NO adversary  -- separates "evidential"
                       from "adversarially trained", and is the rung that
                       silently produced a NaN loss until the loss_input fix.
        ev_at_b0       adversary, REA switched off (beta=0)
        ev_at          full: adversary + REA
        ev_at_awp      + weight-space perturbation, which the paper reports as
                       an additional, non-substitutable gain
        ev_at_kl / _l2 the same beta with a different discrepancy. Comparable
                       only because every mode's self-discrepancy floor is now
                       subtracted; before that, beta=1 meant ~1.10 of constant
                       under ikl and ~0.0001 of signal under kl.

    `pgd_at` from grid A is the fourth corner: adversarial but not evidential.
    """
    base = {**_EVIDENTIAL, "adv_eps": EPS, "adv_steps": STEPS}
    return [
        {"name": "e8_ev_only", "pipe": "standard", **_EVIDENTIAL},
        {"name": "e8_ev_at_b0", "pipe": "evidential_adversarial", **base, "beta": 0.0},
        {"name": "e8_ev_at", "pipe": "evidential_adversarial", **base, "beta": 1.0},
        {
            "name": "e8_ev_at_awp",
            "pipe": "evidential_adversarial",
            **base,
            "beta": 1.0,
            "awp_gamma": 0.01,
        },
        {
            "name": "e8_ev_at_kl",
            "pipe": "evidential_adversarial",
            **base,
            "beta": 1.0,
            "rea_mode": "kl",
        },
        {
            "name": "e8_ev_at_l2",
            "pipe": "evidential_adversarial",
            **base,
            "beta": 1.0,
            "rea_mode": "l2",
        },
    ]


GRIDS = {"A": grid_a, "B": grid_b, "C": grid_c, "E": grid_e}


def build_grid(strategy: str, epochs: int) -> list[dict]:
    cfgs: list[dict] = []
    for letter in strategy.upper():
        if letter not in GRIDS:
            raise SystemExit(f"unknown strategy {letter!r}; pick from {list(GRIDS)}")
        cfgs.extend(GRIDS[letter](epochs))
    return cfgs


def _overrides(cfg: dict, profile: str, epochs: int, batch: int, workers: int) -> list:
    keys = (
        "adv_eps",
        "adv_steps",
        "trades_beta",
        "at_kl_beta",
        "lambda_reg",
        "beta",
        "rea_mode",
        "ikl_ema",
        "awp_gamma",
    )
    # Config GROUPS, not values -- hydra selects these with `key=value` at the
    # top level, same syntax here but a different mechanism.
    groups = ("wrapper", "loss", "uncertainty_score")
    over = [
        f"experiment.name={cfg['name']}",
        f"experiment.training_pipe={cfg['pipe']}",
        f"datamodule.datamodule.profile={profile}",
        f"trainer.trainer.max_epochs={epochs}",
        f"dataloaders.batch_size={batch}",
        f"dataloaders.num_workers={workers}",
        "adv_warmup_epochs=2",
    ]
    over += [f"{k}={cfg[k]}" for k in keys if k in cfg]
    over += [f"{g}={cfg[g]}" for g in groups if g in cfg]
    return over


def _run(script: str, overrides: list, log: pathlib.Path) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            [sys.executable, str(REPO / "src" / script), *overrides],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode == 0


def _already_trained(name: str) -> bool:
    """Does a checkpoint for this configuration exist?

    Training is the expensive half and is deterministic given the seed, so a
    re-run that only adds evaluation conditions should not pay for it twice.
    """
    root = pathlib.Path(os.environ["OUTPUT_PATH"]) / name
    return any(root.rglob("*.ckpt"))


def _latest_test_metrics(name: str) -> dict[str, float]:
    """Every test-time metric for an experiment, merged across conditions.

    Each `src/test.py` invocation writes its OWN `version_N/metrics.csv`, and
    a configuration is evaluated once per condition -- so the clean, ACE and
    over-confidence numbers live in three different files. Reading only the
    newest returns whichever condition happened to run last, which silently
    reduced `confidence_resilience` to a single attack and would have ranked
    the whole grid on half its objective without erroring.

    Files are merged oldest-first so a re-run of a condition wins over the
    run it replaced.
    """
    root = pathlib.Path(os.environ["OUTPUT_PATH"]) / name
    candidates = sorted(
        root.rglob("test_lightning_logs/version_*/metrics.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    merged: dict[str, float] = {}
    for path in candidates:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if value not in ("", None):
                        try:
                            merged[key] = float(value)
                        except ValueError:
                            pass
    return merged


def confidence_resilience(metrics: dict[str, float]) -> float | None:
    """Mean failure-detection AUROC under the confidence attacks.

    This is the objective. Failure AUROC is what a moderation layer's
    abstention rule reads: above 0.5 confidence ranks mistakes below correct
    answers, below 0.5 it ranks them above and abstention actively selects
    for errors. Averaging the realisable (uint8-quantised) ACE with the
    label-free over-confidence attack scores a configuration against the two
    threat models an attacker can actually mount.
    """
    scores = [
        metrics[f"{cond}_fd_auroc"]
        for cond in ("ace_uint8", "overconf")
        if f"{cond}_fd_auroc" in metrics
    ]
    if len(scores) < 2:
        # Both conditions or nothing: averaging one attack and calling it
        # resilience against two would rank the grid on half its objective.
        return None
    return sum(scores) / len(scores)


def _collapse_flag(metrics: dict[str, float]) -> str | None:
    """Has training degenerated to a near-constant output?

    The failure mode wp1 recorded at this budget. It does not look like a
    failure: a model that always answers the majority class posts a
    respectable accuracy and a plausible loss curve. What gives it away is the
    uncertainty score going nearly constant, which collapses the number of
    distinct operating points on the risk-coverage curve -- a healthy run here
    produced 3295 of them, a degenerate one produces a handful. Selective
    metrics computed over a handful of operating points are not wrong so much
    as vacuous, and they must not be ranked as if they were results.
    """
    n_op = metrics.get("nat_n_operating_points")
    if n_op is not None and n_op < 32:
        return f"near-constant uncertainty ({int(n_op)} operating points)"
    return None


def rank(names: list[str], clean_floor: float, out_dir: pathlib.Path) -> list[dict]:
    rows = []
    for name in names:
        metrics = _latest_test_metrics(name)
        if not metrics:
            rows.append({"name": name, "status": "no metrics"})
            continue
        clean = metrics.get("nat_accuracy_top1", metrics.get("nat_accuracy"))
        resilience = confidence_resilience(metrics)
        collapse = _collapse_flag(metrics)
        rows.append(
            {
                "name": name,
                "clean_accuracy": clean,
                "confidence_resilience": resilience,
                "clean_fd_auroc": metrics.get("nat_fd_auroc"),
                "ace_uint8_fd_auroc": metrics.get("ace_uint8_fd_auroc"),
                "overconf_fd_auroc": metrics.get("overconf_fd_auroc"),
                "nat_aurc": metrics.get("nat_aurc"),
                # WP4's own deployment indicators, at thresholds frozen on
                # clean calib. Confidence resilience is the mechanism metric;
                # THIS is the proposal's endpoint -- "coverage, residual risk,
                # and review rate" -- so the ranking carries both.
                "residual_risk_clean": metrics.get("nat_moderation_residual_risk"),
                "residual_risk_ace": metrics.get("ace_uint8_moderation_residual_risk"),
                "review_rate_ace": metrics.get("ace_uint8_moderation_review_rate"),
                "n_operating_points": metrics.get("nat_n_operating_points"),
                # A configuration below the floor is excluded from the
                # ranking rather than deleted: "kept confidence honest by
                # refusing to classify" is a real failure mode and the table
                # should show it happened.
                "meets_floor": (clean is not None and clean >= clean_floor),
                "collapsed": collapse,
                "status": "ok",
            }
        )

    eligible = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("meets_floor")
        and not r.get("collapsed")
        and r.get("confidence_resilience") is not None
    ]
    eligible.sort(key=lambda r: r["confidence_resilience"], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": "mean failure-detection AUROC under ace_uint8 + overconf",
        "clean_accuracy_floor": clean_floor,
        "wp4_note": (
            "residual_risk and review_rate are WP4's deliverable indicators "
            "(D4), measured at thresholds frozen on clean calib. The proposal "
            "names them as the deployment-oriented endpoint; confidence "
            "resilience is the mechanism that predicts them."
        ),
        "provenance": (
            "Sweep profile: small fit/calib/test. These numbers RANK "
            "configurations; they are not results. Re-run the winner at "
            "profile=train before quoting anything."
        ),
        "ranked": eligible,
        "all": rows,
    }
    (out_dir / "sweep_ranking.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Sweep ranking — confidence resilience",
        "",
        f"Objective: {payload['objective']}.",
        f"Clean-accuracy floor: {clean_floor:.2f}.",
        "",
        f"**{payload['provenance']}**",
        "",
        "| # | config | clean acc | conf. resilience | clean AUROC(fail) | "
        "ace_uint8 | overconf | resid. risk clean>ACE | review@ACE | n_op |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(eligible, 1):

        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"

        lines.append(
            f"| {i} | `{r['name']}` | {fmt(r['clean_accuracy'])} | "
            f"{fmt(r['confidence_resilience'])} | {fmt(r['clean_fd_auroc'])} | "
            f"{fmt(r['ace_uint8_fd_auroc'])} | {fmt(r['overconf_fd_auroc'])} | "
            f"{fmt(r['residual_risk_clean'])} > {fmt(r['residual_risk_ace'])} | "
            f"{fmt(r['review_rate_ace'])} | {r['n_operating_points']} |"
        )
    excluded = [r for r in rows if r not in eligible]
    if excluded:
        lines += ["", "## Excluded", ""]
        for r in excluded:
            why = r.get("status")
            if why == "ok":
                why = r.get("collapsed") or "below clean floor"
            lines.append(f"- `{r['name']}` — {why}")
    (out_dir / "sweep_ranking.md").write_text("\n".join(lines) + "\n")
    return eligible


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strategy",
        default="AC",
        help="letters from A/B/C, in order; 'AC' is the documented recommendation",
    )
    ap.add_argument("--profile", default="sweep")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--clean-floor",
        type=float,
        default=0.75,
        help="a configuration below this is ranked nowhere: keeping confidence "
        "honest by refusing to classify is not a win",
    )
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument(
        "--subsample",
        type=int,
        default=SUBSAMPLE_ROWS,
        help="rows every condition is scored on (a prefix of the test split)",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated substrings; run only configurations whose name "
        "matches one. Ranking still covers the whole grid, so a rung added "
        "later is scored beside the rungs that already ran.",
    )
    args = ap.parse_args()

    for var in ("OUTPUT_PATH", "LOGS_PATH", "DATA_PATH"):
        if var not in os.environ:
            raise SystemExit(f"{var} is not set; source the repo's .env first")

    cfgs = build_grid(args.strategy, args.epochs)
    logs = pathlib.Path(os.environ["LOGS_PATH"]) / "sweep"
    # Ranking always spans the FULL grid, even when --only narrows what runs:
    # the point of adding a rung is to see it beside the others.
    names = [c["name"] for c in cfgs]
    if args.only:
        wanted = tuple(s.strip() for s in args.only.split(",") if s.strip())
        cfgs = [c for c in cfgs if any(w in c["name"] for w in wanted)]
        if not cfgs:
            raise SystemExit(f"--only {args.only!r} matched no configuration")

    if not args.rank_only:
        print(
            f"sweep: {len(cfgs)} configurations, strategy {args.strategy}",
            flush=True,
        )
        for i, cfg in enumerate(cfgs, 1):
            t0 = time.time()
            over = _overrides(cfg, args.profile, args.epochs, args.batch, args.workers)
            if _already_trained(cfg["name"]):
                print(
                    f"[{i}/{len(cfgs)}] train {cfg['name']}: skipped (checkpoint)",
                    flush=True,
                )
                ok = True
            else:
                ok = _run("train.py", over, logs / f"train_{cfg['name']}.log")
                print(
                    f"[{i}/{len(cfgs)}] train {cfg['name']}: "
                    f"{'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if not ok:
                continue

            for cond in SCORING_CONDITIONS:
                label = cond.replace("+corruption=", "corr_")
                eval_over = [
                    f"experiment.name={cfg['name']}",
                    f"datamodule.datamodule.profile={args.profile}",
                    f"dataloaders.batch_size={args.batch}",
                    f"dataloaders.num_workers={args.workers}",
                ]
                eval_over.append(f"datamodule.datamodule.limit_test={args.subsample}")
                if cond.startswith("+"):
                    eval_over.append(cond)
                elif cond != "clean":
                    eval_over.append(f"+attack={cond}")
                e0 = time.time()
                good = _run(
                    "test.py", eval_over, logs / f"eval_{cfg['name']}_{label}.log"
                )
                print(
                    f"      eval {cfg['name']} {label} [n={args.subsample}]: "
                    f"{'ok' if good else 'FAILED'} ({time.time() - e0:.0f}s)",
                    flush=True,
                )

    ranked = rank(names, args.clean_floor, logs)
    print(f"\nranked {len(ranked)} configurations; wrote {logs}/sweep_ranking.md")
    for i, r in enumerate(ranked[:5], 1):
        print(
            f"  {i}. {r['name']:24s} resilience={r['confidence_resilience']:.4f} "
            f"clean={r['clean_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
