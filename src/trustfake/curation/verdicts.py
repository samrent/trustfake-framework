"""H1-H3 verdicts, mechanically, from the collated table (spec decision rules).

Written before the first fit completed; the rules are applied, not
relitigated. Operationalizations recorded here:

  * Verdicts read the NATURAL-size fits; matched-size is the size-confound
    control and is reported beside them, flagged if it disagrees.
  * "Pooled seed SD" for a pair of arms on one (leg, metric):
    sqrt((var_a + var_b) / 2) over the 3 seeds, ddof=1.
  * H1 metric: detection_auroc, per shifted leg (L2, L3, L4), read
    conjunctively as the spec words it ("C2 beats C1 on L2-L4").
    The null branch compares C0 vs C5b the same way.
  * H2 metric: mean detection_auroc_tampered over the shifted legs.
    Tie or within-noise -> strict-drop (C3a), per the spec.
  * H3: C5b must beat C5a by > 2x pooled SD on >= 2 shifted legs.
  * Watchdog: any adjacent ladder stage that drops fd_auroc by > 0.05 on
    any leg is flagged (no decision rule attached).
  * Borderline (any rule where |delta| < 2x pooled SD): the verdict says
    "add 5 seeds", it does not soften.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHIFTED_LEGS = ("L2", "L3", "L4")
LADDER_ORDER = ("C0", "C1", "C2", "C3a", "C3b", "C4", "C5a", "C5b")


def _pair(table: pd.DataFrame, arm_a: str, arm_b: str, leg: str, metric: str):
    """(mean_b - mean_a, 2x pooled seed SD, per-seed values)."""
    rows = table[(table["size"] == "natural") & (table["leg"] == leg)]
    a = rows[rows["arm"] == arm_a].sort_values("seed")[metric].to_numpy()
    b = rows[rows["arm"] == arm_b].sort_values("seed")[metric].to_numpy()
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), (a, b)
    pooled = (
        float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
        if a.size > 1
        else float("nan")
    )
    return float(b.mean() - a.mean()), 2 * pooled, (a, b)


def h1(table: pd.DataFrame) -> dict:
    """Does curation matter: C2 vs C1 per shifted leg; null check C0 vs C5b."""
    legs = {}
    for leg in SHIFTED_LEGS:
        delta, bar, _ = _pair(table, "C1", "C2", leg, "detection_auroc")
        legs[leg] = {
            "delta": round(delta, 4),
            "two_sd": round(bar, 4),
            "beats": bool(delta > bar),
            "borderline": bool(abs(delta) < bar),
        }
    null_legs = {}
    for leg in SHIFTED_LEGS:
        delta, bar, _ = _pair(table, "C0", "C5b", leg, "detection_auroc")
        null_legs[leg] = {"delta": round(delta, 4), "two_sd": round(bar, 4)}
    curation_null = all(abs(v["delta"]) <= v["two_sd"] for v in null_legs.values())
    verdict = (
        "curation matters (C2 > C1 beyond 2x pooled seed SD on every shifted leg)"
        if all(v["beats"] for v in legs.values())
        else (
            "NULL: C0 ~ C5b on every shifted leg -- curation is not the lever; "
            "record in rejected/ and stop"
            if curation_null
            else "mixed -- read per-leg rows; add 5 seeds where borderline"
        )
    )
    return {
        "rule": "C2 > C1 by 2x pooled SD on L2-L4",
        "legs": legs,
        "null_check_c0_vs_c5b": null_legs,
        "verdict": verdict,
    }


def h2(table: pd.DataFrame) -> dict:
    """Mapping policy: C3 winner on tampered detection under shift."""
    deltas = {}
    for leg in SHIFTED_LEGS:
        delta, bar, _ = _pair(table, "C3a", "C3b", leg, "detection_auroc_tampered")
        deltas[leg] = {"delta_b_minus_a": round(delta, 4), "two_sd": round(bar, 4)}
    mean_delta = float(np.nanmean([v["delta_b_minus_a"] for v in deltas.values()]))
    mean_bar = float(np.nanmean([v["two_sd"] for v in deltas.values()]))
    if np.isnan(mean_delta) or abs(mean_delta) <= mean_bar:
        verdict = (
            "tie or within noise -> strict-drop (C3a), per the "
            "pre-registered default"
        )
        winner = "C3a"
    else:
        winner = "C3b" if mean_delta > 0 else "C3a"
        verdict = f"{winner} wins on tampered detection_auroc under shift"
    return {
        "rule": "winner on detection_auroc_tampered under shift; ties -> C3a",
        "legs": deltas,
        "winner": winner,
        "verdict": verdict,
    }


def h3(table: pd.DataFrame) -> dict:
    """Coreset: C5b must beat C5a beyond seed noise on >= 2 shifted legs."""
    legs = {}
    for leg in SHIFTED_LEGS:
        delta, bar, _ = _pair(table, "C5a", "C5b", leg, "detection_auroc")
        legs[leg] = {
            "delta": round(delta, 4),
            "two_sd": round(bar, 4),
            "beats": bool(delta > bar),
        }
    wins = sum(v["beats"] for v in legs.values())
    verdict = (
        f"coreset earns its keep ({wins}/3 shifted legs beyond noise)"
        if wins >= 2
        else f"coreset dropped permanently ({wins}/3 shifted legs -- rule needs >= 2)"
    )
    return {
        "rule": "C5b > C5a beyond seed noise on >= 2 shifted legs",
        "legs": legs,
        "verdict": verdict,
    }


def watchdog(table: pd.DataFrame) -> list[str]:
    """Flag adjacent-stage fd_auroc drops > 0.05 on any leg (natural size)."""
    flags = []
    natural = table[table["size"] == "natural"]
    means = natural.groupby(["arm", "leg"])["fd_auroc"].mean()
    for earlier, later in zip(LADDER_ORDER, LADDER_ORDER[1:], strict=False):
        for leg in natural["leg"].unique():
            try:
                drop = means[(earlier, leg)] - means[(later, leg)]
            except KeyError:
                continue
            if drop > 0.05:
                flags.append(
                    f"{earlier}->{later} on {leg}: fd_auroc drops {drop:.3f} (> 0.05)"
                )
    return flags


def all_verdicts(table: pd.DataFrame) -> dict:
    return {
        "H1": h1(table),
        "H2": h2(table),
        "H3": h3(table),
        "watchdog_flags": watchdog(table),
    }
