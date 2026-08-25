"""WP4 selective moderation: turn a detector output into a moderation ACTION.

    ALLOW    auto-accept as real          (p_fake <  t_low)
    REVIEW   escalate to a human          (t_low <= p_fake <= t_high, or gated)
    FLAG     auto-flag as fake            (p_fake >  t_high)

Two thresholds on the risk axis p(fake), not one on confidence: moderation
costs are asymmetric (a missed fake ships harm; a false flag annoys a user;
a review costs a moderator's minute), and two thresholds price the three
independently -- a single confidence threshold is the symmetric special case.
For 3-class SID-Set, fake = synthetic OR tampered, so p_fake = 1 - P(real).

An optional uncertainty gate (second axis) forces REVIEW when the uncertainty
score exceeds t_unc, regardless of p_fake. It only ever MOVES items into
review, so it cannot raise residual risk -- and its value is entirely under
attack, where a confidence attack drives uncertainty up and more items cross
the frozen gate into human review instead of being wrongly auto-decided.

Thresholds are fitted on the clean calibration split and frozen. Re-fitting on
attacked data would be an oracle policy (it assumes the moderator knows an
attack is underway -- the very assumption the attack defeats).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ALLOW",
    "REVIEW",
    "FLAG",
    "ModerationPolicy",
    "moderation_actions",
    "evaluate_policy",
    "fit_thresholds",
    "fit_uncertainty_gate",
]

ALLOW, REVIEW, FLAG = 0, 1, 2


def moderation_actions(
    p_fake: np.ndarray,
    t_low: float,
    t_high: float,
    uncertainty: np.ndarray | None = None,
    t_unc: float | None = None,
) -> np.ndarray:
    """Per-item action. p_fake below t_low -> ALLOW, above t_high -> FLAG,
    between -> REVIEW; an uncertainty above t_unc forces REVIEW."""
    p = np.asarray(p_fake, dtype=np.float64)
    out = np.full(p.shape, REVIEW, dtype=np.int8)
    out[p < t_low] = ALLOW
    out[p > t_high] = FLAG
    if uncertainty is not None and t_unc is not None:
        out[np.asarray(uncertainty, dtype=np.float64) > t_unc] = REVIEW
    return out


def evaluate_policy(
    p_fake: np.ndarray,
    y_binary: np.ndarray,
    t_low: float,
    t_high: float,
    uncertainty: np.ndarray | None = None,
    t_unc: float | None = None,
) -> dict:
    """Deployment indicators for one (policy, condition) pair.
    y_binary: 1 = fake (synthetic or tampered), 0 = real."""
    p = np.asarray(p_fake, dtype=np.float64)
    y = np.asarray(y_binary).astype(int)
    a = moderation_actions(p, t_low, t_high, uncertainty, t_unc)
    auto = a != REVIEW
    n = y.size
    n_auto = int(auto.sum())

    decided_pred = (a[auto] == FLAG).astype(int)  # FLAG asserts fake
    fakes, reals = y == 1, y == 0
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "coverage": n_auto / n,
        "review_rate": 1.0 - n_auto / n,
        # NaN, never 0.0: a policy that auto-decides nothing has no residual
        # risk to measure, and 0.0 would make total refusal look like perfect
        # safety -- the most flattering misreading of a collapsed system.
        "residual_risk": (
            float((decided_pred != y[auto]).mean()) if n_auto else float("nan")
        ),
        "n_auto": n_auto,
        "missed_fake_rate": float((a[fakes] == ALLOW).mean()) if fakes.any() else 0.0,
        "false_flag_rate": float((a[reals] == FLAG).mean()) if reals.any() else 0.0,
        "full_coverage_error": float(((p > 0.5).astype(int) != y).mean()),
    }


def fit_thresholds(
    p_fake: np.ndarray,
    y_binary: np.ndarray,
    sla_residual_risk: float = 0.05,
    sla_missed_fake: float | None = None,
    grid: int = 200,
    min_auto: int | None = None,
) -> tuple[float, float]:
    """Minimise review rate subject to the residual-risk SLA (and optionally a
    cap on the missed-fake rate), on the clean calibration split.

    The SLA is a choice, not a fact. If no threshold pair meets it, the policy
    degrades to REVIEW EVERYTHING (t_low=0, t_high=1) rather than returning the
    least-bad infeasible pair -- an infeasible SLA must read as "cannot be
    deployed at this SLA", a finding, not a silent failure.
    """
    p = np.asarray(p_fake, dtype=np.float64)
    y = np.asarray(y_binary).astype(int)
    n = p.size
    # Require enough auto-decisions that the SLA could have been violated at
    # all: the tightest thresholds "satisfy" any SLA by deciding almost nothing.
    if min_auto is None:
        min_auto = int(np.ceil(1.0 / max(sla_residual_risk, 1e-9)))
    qs = np.unique(np.quantile(p, np.linspace(0, 1, grid)))

    # Vectorised over the (t_low, t_high) grid. Sort once; every region count
    # comes from searchsorted + cumulative sums, so the whole search is
    # O(grid^2) arithmetic rather than O(grid^2 * n) policy evaluations.
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    cum_fake = np.concatenate([[0], np.cumsum(ys)])  # fakes among the first k
    total_real = int((y == 0).sum())

    # ALLOW region p < t_low: assert real, so an error is a fake caught there.
    k_low = np.searchsorted(ps, qs, side="left")
    n_allow = k_low.astype(np.float64)
    allow_err = cum_fake[k_low].astype(np.float64)  # fakes with p < t_low
    n_fake_below = cum_fake[k_low]  # for the missed-fake rate
    total_fake = int(cum_fake[-1])

    # FLAG region p > t_high: assert fake, so an error is a real flagged there.
    k_high = np.searchsorted(ps, qs, side="right")
    n_flag = (n - k_high).astype(np.float64)
    reals_below = k_high - cum_fake[k_high]
    flag_err = (total_real - reals_below).astype(np.float64)  # reals with p > t_high

    valid = qs[:, None] <= qs[None, :]  # t_low <= t_high (disjoint regions)

    auto = n_allow[:, None] + n_flag[None, :]
    err = allow_err[:, None] + flag_err[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = np.where(auto > 0, err / auto, np.inf)
    review = 1.0 - auto / n
    # missed fakes = fakes auto-ALLOWED = fakes with p < t_low (t_high-independent)
    missed_fake = (n_fake_below / max(total_fake, 1))[:, None] * np.ones_like(auto)

    feasible = valid & (auto >= min_auto) & np.isfinite(residual)
    feasible &= residual <= sla_residual_risk
    if sla_missed_fake is not None:
        feasible &= missed_fake <= sla_missed_fake
    if not feasible.any():
        return (0.0, 1.0)

    review_masked = np.where(feasible, review, np.inf)
    flat = int(np.argmin(review_masked))
    i, j = flat // qs.size, flat % qs.size
    return (float(qs[i]), float(qs[j]))


def fit_uncertainty_gate(
    uncertainty: np.ndarray, clean_review_budget: float = 0.10
) -> float:
    """Freeze the uncertainty gate at a clean review budget: t_unc is the
    (1 - budget) quantile of clean-calib uncertainty, so it reviews the most
    uncertain `clean_review_budget` fraction of clean items."""
    return float(
        np.quantile(np.asarray(uncertainty, np.float64), 1.0 - clean_review_budget)
    )


@dataclass
class ModerationPolicy:
    """A frozen moderation policy fitted on clean calib."""

    t_low: float
    t_high: float
    real_class: int = 0
    t_unc: float | None = None

    @property
    def infeasible(self) -> bool:
        """True when fitting degraded to review-everything for the SLA."""
        return self.t_low == 0.0 and self.t_high == 1.0

    def p_fake(self, probs: np.ndarray) -> np.ndarray:
        """p(fake) = 1 - P(real) from full class probabilities."""
        return 1.0 - np.asarray(probs, dtype=np.float64)[:, self.real_class]

    def evaluate(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        uncertainty: np.ndarray | None = None,
        use_gate: bool = False,
    ) -> dict:
        y_binary = (np.asarray(targets).astype(int) != self.real_class).astype(int)
        return evaluate_policy(
            self.p_fake(probs),
            y_binary,
            self.t_low,
            self.t_high,
            uncertainty if use_gate else None,
            self.t_unc if use_gate else None,
        )
