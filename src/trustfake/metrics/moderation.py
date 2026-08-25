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
review, so the COUNT of wrong auto-decisions can only fall, and with it
missed_fake_rate and false_flag_rate, whose denominators are fixed class
sizes. residual_risk is NOT among the guarantees: it divides by a shrinking
denominator, so a gate escalating decisions that were correct raises the rate
while shipping strictly less harm. Read it beside coverage, never alone. The
gate's value is entirely under attack, where a confidence attack drives
uncertainty up and more items cross the frozen gate into human review instead
of being wrongly auto-decided.

Thresholds are fitted on the clean calibration split and frozen. Re-fitting on
attacked data would be an oracle policy (it assumes the moderator knows an
attack is underway -- the very assumption the attack defeats).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ALLOW",
    "REVIEW",
    "FLAG",
    "ACTION_NAMES",
    "ModerationPolicy",
    "moderation_actions",
    "p_fake_from",
    "evaluate_policy",
    "fit_thresholds",
    "fit_uncertainty_gate",
    "moderation_report",
]

ALLOW, REVIEW, FLAG = 0, 1, 2
ACTION_NAMES = {ALLOW: "allow", REVIEW: "review", FLAG: "flag"}


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax in float64 with max subtraction.

    float64 is not decoration: a float32 softmax saturates to exactly 1.0 on
    confident rows, which manufactures the tie blocks that make every
    threshold-on-a-score quantity depend on the row order.

    Args:
        logits: ``(N, C)`` raw logits, any float dtype.
        temperature: ``T > 0`` dividing the logits before the softmax.

    Returns:
        ``(N, C)`` float64 probabilities, rows summing to 1.

    Raises:
        ValueError: If ``logits`` is not 2-D, or ``temperature`` is not > 0.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("logits must be 2-D (N, C)")
    if not temperature > 0:  # also rejects NaN
        raise ValueError("temperature must be > 0")
    z = z / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def p_fake_from(
    logits: np.ndarray, temperature: float = 1.0, real_class: int = 0
) -> np.ndarray:
    """p(fake) straight from raw logits, applying the temperature here.

    The one place the logits -> p_fake convention lives. A caller holding raw
    logits (a stored prediction artifact, an attacked forward pass) must not
    re-derive the softmax: a float32 softmax saturates, and a second opinion
    about which column means "fake" is a silent sign error. For binary logits
    with ``real_class=0`` this is exactly ``softmax(logits / T)[:, 1]``.

    Temperature is not free: with two classes p_fake is monotone in the single
    logit margin, so T only rescales it against the FROZEN thresholds; with
    three (real / synthetic / tampered) p_fake sums two columns and T CAN
    reorder items (see ``metrics.calibration.temperature``). Either way it is
    fitted once on clean calib and frozen with the thresholds -- refitting it
    per condition would be the same oracle mistake as refitting the policy.

    Args:
        logits: ``(N, C)`` raw logits.
        temperature: ``T > 0``; 1.0 is a no-op. Fit it with
            ``trustfake.metrics.calibration.fit_temperature``.
        real_class: Index of the real class. Every other class is fake
            (SID-Set: synthetic OR tampered), so p_fake = 1 - P(real).

    Returns:
        ``(N,)`` float64 array of p(fake) in [0, 1].
    """
    return 1.0 - _softmax(logits, temperature)[:, real_class]


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

    Args:
        p_fake: ``(N,)`` p(fake) per item, e.g. from :func:`p_fake_from`.
        y_binary: ``(N,)`` labels, 1 = fake (synthetic or tampered), 0 = real.
        t_low: Below this p_fake, auto-ALLOW.
        t_high: Above this p_fake, auto-FLAG.
        uncertainty: Optional second axis; ``None`` gives the one-axis rule.
        t_unc: Frozen gate on ``uncertainty``; above it, force REVIEW.

    Returns:
        A dict of deployment indicators:

        * ``n``, ``n_auto`` -- items seen, items auto-decided. ``n`` is what
          makes every rate below re-derivable (and a 3-item condition
          distinguishable from a 3000-item one) from the report alone.
        * ``coverage``, ``review_rate`` -- the moderation bill.
        * ``residual_risk`` -- error rate AMONG the auto-decided: what
          actually ships wrong. NaN when nothing is auto-decided.
        * ``full_coverage_error``, ``accuracy`` -- the error rate and
          accuracy of the same detector at ``p > 0.5`` with NO abstention.
          Residual risk is only interpretable against them: abstention has
          bought nothing unless residual_risk < full_coverage_error.
        * ``missed_fake_rate`` -- fakes auto-ALLOWED, over all fakes (harm).
        * ``false_flag_rate`` -- reals auto-FLAGGED, over all reals (annoyance).
        * ``review_of_fakes``, ``review_of_reals`` -- the review load split by
          truth. Without them a gate that escalates the RIGHT traffic and one
          that escalates random traffic are indistinguishable at equal
          review_rate; these two are how the uncertainty gate is shown to
          move fakes, not volume, into the human queue.
    """
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
        "n": int(n),
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
        # The review load split by truth: escalating fakes is the gate working,
        # escalating volume is the gate charging for nothing.
        "review_of_fakes": float((a[fakes] == REVIEW).mean()) if fakes.any() else 0.0,
        "review_of_reals": float((a[reals] == REVIEW).mean()) if reals.any() else 0.0,
        # The no-abstention counterfactual, reported beside the policy so the
        # residual risk is never read on its own.
        "full_coverage_error": float(((p > 0.5).astype(int) != y).mean()),
        "accuracy": float(((p > 0.5).astype(int) == y).mean()),
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

    Args:
        p_fake: ``(N,)`` p(fake) on the CLEAN CALIB split.
        y_binary: ``(N,)`` labels, 1 = fake, 0 = real.
        sla_residual_risk: Max tolerated error rate among auto-decisions.
        sla_missed_fake: Optional second, ASYMMETRIC constraint: max tolerated
            fraction of all fakes that may be auto-ALLOWED. Residual risk
            prices a missed fake and a false flag identically; a platform does
            not. This cap is what lets a missed fake be declared the more
            expensive of the two -- it pushes t_low down (allowing less,
            reviewing more) until the fakes hiding in the ALLOW zone are
            escalated instead. ``None`` leaves the objective symmetric.
        grid: Number of quantiles of ``p_fake`` forming the threshold grid.
        min_auto: Minimum auto-decided items for a pair to count as feasible.
            Defaults to ``ceil(1 / sla_residual_risk)``.

    Returns:
        ``(t_low, t_high)``; ``(0.0, 1.0)`` means review-everything, i.e. no
        pair satisfied the constraints.
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
    # missed fakes = fakes auto-ALLOWED = fakes with p < t_low. It does not
    # depend on t_high, so it stays a column and broadcasts over the grid.
    missed_fake = (n_fake_below / max(total_fake, 1))[:, None]

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

    def p_fake_from_logits(
        self, logits: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """p(fake) from RAW logits, honouring this policy's ``real_class``.

        The same convention as :meth:`p_fake`, for a caller that holds logits
        rather than probabilities and would otherwise re-derive the softmax.
        """
        return p_fake_from(logits, temperature, self.real_class)

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


# --- reporting ---------------------------------------------------------------

_INDICATORS = (
    "accuracy",
    "full_coverage_error",
    "coverage",
    "review_rate",
    "residual_risk",
    "missed_fake_rate",
    "false_flag_rate",
    "review_of_fakes",
    "review_of_reals",
)

_ROW_HEAD = (
    "| condition | n | acc | coverage | review% | resid_risk "
    "| full_cov_err | missed_fake | false_flag |"
)
_ROW_SEP = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

_GATE_HEAD = (
    "| condition | resid 1ax | resid 2ax | review 1ax | review 2ax "
    "| rev_fakes 1ax | rev_fakes 2ax |"
)
_GATE_SEP = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"


def _pct(value: float | None, digits: int = 2) -> str:
    """Percent, or ``n/a`` for a quantity that does not exist -- never 0.0."""
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 4) -> str:
    """Plain float, or ``n/a``."""
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _count(value: float | None) -> str:
    """Integer count, or ``n/a``."""
    return "n/a" if value is None else str(int(value))


def _jsonable(value: float | None) -> float | None:
    """Strict-JSON float: non-finite becomes ``None``, i.e. not measurable.

    ``None`` survives ``json.dumps(..., allow_nan=False)`` and reads as "there
    was nothing to measure"; 0.0 would read as "perfectly safe", which is the
    single most flattering misreading of a policy that decided nothing.
    """
    if value is None:
        return None
    f = float(value)
    return f if np.isfinite(f) else None


def _entry(row: Mapping[str, float]) -> dict:
    """One condition's indicators, JSON-clean."""
    out: dict = {k: _jsonable(row.get(k)) for k in _INDICATORS}
    out["n"] = None if row.get("n") is None else int(row["n"])
    out["n_auto"] = None if row.get("n_auto") is None else int(row["n_auto"])
    return out


def moderation_report(
    per_condition: Mapping[str, Mapping[str, float]],
    gated: Mapping[str, Mapping[str, float]] | None = None,
    policy: ModerationPolicy | None = None,
    sla_residual_risk: float | None = None,
    sla_missed_fake: float | None = None,
    title: str = "WP4 selective moderation",
) -> tuple[str, dict]:
    """Render frozen-policy results as a markdown report and a JSON-clean dict.

    Pure function: it formats what it is handed and returns it. Where the
    report lands -- an artifact directory, a logger, a paper table -- is the
    caller's decision, so nothing here touches the filesystem.

    Two tables. The first is the per-condition deployment row, with
    ``resid_risk`` deliberately adjacent to ``full_cov_err`` so the residual
    risk is never read without the no-abstention error it has to beat. The
    second appears only when ``gated`` is given: the one-axis rule against the
    two-axis rule (p_fake plus the frozen uncertainty gate), including
    ``rev_fakes``, because at equal review rate a gate escalating fakes and a
    gate escalating random traffic are otherwise indistinguishable.

    Args:
        per_condition: ``{condition: evaluate_policy(...)}`` for the one-axis
            rule, in report order (e.g. ``clean`` first, then each attack).
        gated: Optional ``{condition: evaluate_policy(..., uncertainty, t_unc)}``
            for the same conditions. Conditions absent here simply get no
            two-axis row.
        policy: Optional frozen policy, rendered as the header provenance
            (thresholds, gate, and the infeasibility marker).
        sla_residual_risk: Optional SLA the policy was fitted at, recorded so a
            reader can disagree with it explicitly rather than inherit it.
        sla_missed_fake: Optional asymmetric cap the policy was fitted at.
        title: Report heading.

    Returns:
        ``(markdown, payload)``. ``payload`` contains only str/int/float/bool/
        None, so ``json.dumps(payload, allow_nan=False)`` succeeds; a
        non-measurable indicator is ``None``, never 0.0.
    """
    lines = [f"## {title}", ""]
    if policy is not None:
        lines.append(
            "- policy fitted on the clean calibration split and frozen: "
            f"`t_low={policy.t_low:.4f}`, `t_high={policy.t_high:.4f}`"
        )
        if policy.t_unc is not None:
            lines.append(
                f"- uncertainty gate (second axis): `t_unc={policy.t_unc:.5f}`, "
                "fitted on clean calib and never re-fitted under attack"
            )
        if policy.infeasible:
            lines.append(
                "- **SLA INFEASIBLE**: no threshold pair met it, so the policy "
                "degraded to review-everything. A finding, not a failure."
            )
    if sla_residual_risk is not None:
        lines.append(
            f"- SLA: residual risk <= {sla_residual_risk:.2%} "
            "(an SLA is a choice, not a fact)"
        )
    if sla_missed_fake is not None:
        lines.append(f"- SLA: missed fakes <= {sla_missed_fake:.2%} of all fakes")
    lines += [
        "- `resid_risk` = error rate among AUTO-DECIDED items; `full_cov_err` = "
        "error rate of the same detector with no abstention at all. Abstention "
        "bought nothing unless the first is below the second.",
        "",
        _ROW_HEAD,
        _ROW_SEP,
    ]

    conditions: dict[str, dict] = {}
    for name, row in per_condition.items():
        lines.append(
            f"| {name} | {_count(row.get('n'))} | {_num(row.get('accuracy'))} "
            f"| {_pct(row.get('coverage'), 1)} | {_pct(row.get('review_rate'), 1)} "
            f"| {_pct(row.get('residual_risk'))} "
            f"| {_pct(row.get('full_coverage_error'))} "
            f"| {_pct(row.get('missed_fake_rate'))} "
            f"| {_pct(row.get('false_flag_rate'))} |"
        )
        conditions[name] = _entry(row)

    paired = [
        (name, row, gated[name])
        for name, row in per_condition.items()
        if gated is not None and name in gated
    ]
    if paired:
        lines += [
            "",
            "### one axis (p_fake) vs two axes (+ uncertainty gate)",
            "",
            "The gate only ever MOVES items into review, so it ships strictly "
            "less harm -- but `resid 2ax` can still come out ABOVE `resid 1ax`, "
            "because escalating correct auto-decisions shrinks the denominator "
            "faster than the errors. Read it beside `review`. `rev_fakes` "
            "(share of all FAKES escalated) is what shows the review budget "
            "bought the right traffic rather than volume.",
            "",
            _GATE_HEAD,
            _GATE_SEP,
        ]
        for name, one, two in paired:
            lines.append(
                f"| {name} | {_pct(one.get('residual_risk'))} "
                f"| {_pct(two.get('residual_risk'))} "
                f"| {_pct(one.get('review_rate'), 1)} "
                f"| {_pct(two.get('review_rate'), 1)} "
                f"| {_pct(one.get('review_of_fakes'), 1)} "
                f"| {_pct(two.get('review_of_fakes'), 1)} |"
            )
            conditions[name]["two_axis"] = _entry(two)

    payload = {
        "title": title,
        "policy": None
        if policy is None
        else {
            "t_low": float(policy.t_low),
            "t_high": float(policy.t_high),
            "t_unc": None if policy.t_unc is None else float(policy.t_unc),
            "real_class": int(policy.real_class),
            "infeasible": bool(policy.infeasible),
        },
        "sla": {
            "residual_risk": _jsonable(sla_residual_risk),
            "missed_fake": _jsonable(sla_missed_fake),
        },
        "conditions": conditions,
        "note": (
            "residual_risk = error rate among auto-decided items; null means "
            "nothing was auto-decided, so it is not measurable -- never 0.0. "
            "Read it against full_coverage_error, the no-abstention baseline."
        ),
    }
    return "\n".join(lines) + "\n", payload
