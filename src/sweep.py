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

#: Conditions every configuration is scored under. The two confidence attacks
#: are the objective; `clean` supplies the floor. Deliberately short -- a
#: sweep pays for its conditions once per configuration.
SCORING_CONDITIONS = ("clean", "ace_uint8", "overconf")


#: L_inf budgets, in units of 1/255. 8/255 is the field standard -- Madry's
#: setting and RobustBench's headline column -- so it is the only number in
#: this grid that makes a result comparable to everyone else's, which is what
#: a BASELINE is for. It is included despite the forensic argument against it
#: (a ball that wide erases the small-amplitude high-frequency evidence a
#: deepfake detector reads, and wp1 recorded training collapsing onto a
#: constant output there). That argument is a claim about this task, and a
#: claim is worth testing rather than designing around: if 8/255 collapses,
#: that is the finding that justifies the forensic regime to a reviewer who
#: will otherwise ask why the standard budget is missing.
EPS_LADDER = (1, 2, 4, 8)


def grid_a(epochs: int) -> list[dict]:
    """Strategy A: the knobs of the arms that already exist."""
    cfgs: list[dict] = [
        {"name": "sw_std", "pipe": "standard", "adv_eps": 2 / 255},
    ]
    for eps in EPS_LADDER:
        for steps in (3, 7):
            cfgs.append(
                {
                    "name": f"sw_atpgd_e{eps}_s{steps}",
                    "pipe": "pgd_at",
                    "adv_eps": eps / 255,
                    "adv_steps": steps,
                }
            )
    for eps in EPS_LADDER:
        for beta in (3, 6):
            cfgs.append(
                {
                    "name": f"sw_trades_e{eps}_b{beta}",
                    "pipe": "trades",
                    "adv_eps": eps / 255,
                    "adv_steps": 7,
                    "trades_beta": float(beta),
                }
            )
    return cfgs


def grid_b(epochs: int) -> list[dict]:
    """Strategy B: the AT + consistency-KL hybrid. Recommended to skip."""
    return [
        {
            "name": f"sw_atkl_e{eps}_b{beta}",
            "pipe": "at_kl",
            "adv_eps": eps / 255,
            "adv_steps": 7,
            "at_kl_beta": float(beta),
        }
        for eps in (2, 4)
        for beta in (3, 6)
    ]


def grid_c(epochs: int) -> list[dict]:
    """Strategy C: the confidence-targeted defences."""
    cfgs = [
        {
            "name": f"sw_atconf_e{eps}",
            "pipe": "at_conf",
            "adv_eps": eps / 255,
            "adv_steps": 7,
        }
        for eps in (2, 4)
    ]
    cfgs += [
        {
            "name": f"sw_confreg_l{lam}",
            "pipe": "conf_reg",
            "adv_eps": 2 / 255,
            "lambda_reg": lam,
        }
        for lam in (0.5, 1.0, 2.0)
    ]
    return cfgs


GRIDS = {"A": grid_a, "B": grid_b, "C": grid_c}


def build_grid(strategy: str, epochs: int) -> list[dict]:
    cfgs: list[dict] = []
    for letter in strategy.upper():
        if letter not in GRIDS:
            raise SystemExit(f"unknown strategy {letter!r}; pick from {list(GRIDS)}")
        cfgs.extend(GRIDS[letter](epochs))
    return cfgs


def _overrides(cfg: dict, profile: str, epochs: int, batch: int, workers: int) -> list:
    keys = ("adv_eps", "adv_steps", "trades_beta", "at_kl_beta", "lambda_reg")
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


def rank(names: list[str], clean_floor: float, out_dir: pathlib.Path) -> list[dict]:
    rows = []
    for name in names:
        metrics = _latest_test_metrics(name)
        if not metrics:
            rows.append({"name": name, "status": "no metrics"})
            continue
        clean = metrics.get("nat_accuracy_top1", metrics.get("nat_accuracy"))
        resilience = confidence_resilience(metrics)
        rows.append(
            {
                "name": name,
                "clean_accuracy": clean,
                "confidence_resilience": resilience,
                "clean_fd_auroc": metrics.get("nat_fd_auroc"),
                "ace_uint8_fd_auroc": metrics.get("ace_uint8_fd_auroc"),
                "overconf_fd_auroc": metrics.get("overconf_fd_auroc"),
                "nat_aurc": metrics.get("nat_aurc"),
                "n_operating_points": metrics.get("nat_n_operating_points"),
                # A configuration below the floor is excluded from the
                # ranking rather than deleted: "kept confidence honest by
                # refusing to classify" is a real failure mode and the table
                # should show it happened.
                "meets_floor": (clean is not None and clean >= clean_floor),
                "status": "ok",
            }
        )

    eligible = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("meets_floor")
        and r.get("confidence_resilience") is not None
    ]
    eligible.sort(key=lambda r: r["confidence_resilience"], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": "mean failure-detection AUROC under ace_uint8 + overconf",
        "clean_accuracy_floor": clean_floor,
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
        "ace_uint8 | overconf | n_op |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(eligible, 1):

        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}"

        lines.append(
            f"| {i} | `{r['name']}` | {fmt(r['clean_accuracy'])} | "
            f"{fmt(r['confidence_resilience'])} | {fmt(r['clean_fd_auroc'])} | "
            f"{fmt(r['ace_uint8_fd_auroc'])} | {fmt(r['overconf_fd_auroc'])} | "
            f"{r['n_operating_points']} |"
        )
    excluded = [r for r in rows if r not in eligible]
    if excluded:
        lines += ["", "## Excluded", ""]
        for r in excluded:
            why = r.get("status") if r.get("status") != "ok" else "below clean floor"
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
            ok = _run("train.py", over, logs / f"train_{cfg['name']}.log")
            print(
                f"[{i}/{len(cfgs)}] train {cfg['name']}: "
                f"{'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)",
                flush=True,
            )
            if not ok:
                continue
            for cond in SCORING_CONDITIONS:
                eval_over = [
                    f"experiment.name={cfg['name']}",
                    f"datamodule.datamodule.profile={args.profile}",
                    f"dataloaders.batch_size={args.batch}",
                    f"dataloaders.num_workers={args.workers}",
                ]
                if cond != "clean":
                    eval_over.append(f"+attack={cond}")
                _run("test.py", eval_over, logs / f"eval_{cfg['name']}_{cond}.log")

    ranked = rank(names, args.clean_floor, logs)
    print(f"\nranked {len(ranked)} configurations; wrote {logs}/sweep_ranking.md")
    for i, r in enumerate(ranked[:5], 1):
        print(
            f"  {i}. {r['name']:24s} resilience={r['confidence_resilience']:.4f} "
            f"clean={r['clean_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
