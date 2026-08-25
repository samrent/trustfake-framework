"""Selective-classification metrics: AURC, AUGRC, E-AURC and operating points.

Convention (it matters, and it is asserted by the tests): the risk-coverage
curve is evaluated at DISTINCT operating points only -- tie blocks in the
uncertainty score are collapsed, because a threshold inside a tie block
selects a permutation-dependent set of rows. AURC/AUGRC are then the
block-size-weighted mean over those operating points. A naive cumulative
mean over every rank varies across row permutations of the same predictions
(saturated softmax manufactures exactly such tie blocks), and trapezoid
integration over coverage is a third convention in the literature; neither
is used here.

Two weightings are implemented and every table must state which it used:

    ``weights="block"`` (default) gives each distinct operating point the
        size of its tie block. Reduces exactly to the textbook mean over
        k = 1..n when all scores are distinct, so the numbers stay
        comparable to published AURCs, and a 5000-row tie block cannot count
        the same as a singleton.
    ``weights="uniform"`` is the unweighted mean over the achievable
        coverage points.

E-AURC subtracts the AURC of the EMPIRICAL oracle -- the same n and the same
errors, correct predictions ranked first -- never the asymptotic closed form
``r + (1-r)ln(1-r)``, which is off by ~1e-4 at n=1000, the same order as the
effects being reported.

Read :func:`n_operating_points` beside every AURC. AURC is only a precise
number while the score has not saturated into tie blocks: a row whose
``n_operating_points`` is far below ``n`` has an AURC that depends on the
temperature and on how the tie blocks happen to fall, and must not be read
as exact.

Everything here upcasts the uncertainty score to float64 before ranking. See
:func:`_as_score` for why that is not decoration.
"""

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

__all__ = [
    "AURiskCoverage",
    "AUGRiskCoverage",
    "EAURiskCoverage",
    "NumOperatingPoints",
    "RiskAtCoverage",
    "AchievedCoverage",
    "get_selective_classification_metrics",
    "rc_curve",
    "n_operating_points",
    "aurc_from_scores",
    "augrc_from_scores",
    "aurc_oracle",
    "eaurc_from_scores",
    "operating_point_at_coverage",
    "risk_at_coverage",
    "coverage_at_risk",
]


def _as_score(uncertainty: np.ndarray) -> np.ndarray:
    """Validate an uncertainty score and upcast it to float64 for ranking.

    float64 is not decoration. A confidence computed in fp16/fp32 saturates
    to exactly 1.0 for confident rows -- and ``1 - max_prob`` then cancels to
    exactly 0.0 -- which manufactures a tie block out of rows the model
    actually ranked apart. Every such block collapses to a single operating
    point, so the AURC silently becomes a coarser, temperature-dependent
    number. Upcasting here cannot undo saturation that already happened
    upstream (the softmax itself has to be computed in float64); what it
    does is guarantee nothing downstream of this boundary loses another bit,
    and :func:`n_operating_points` is the detector for damage done before
    it.

    Args:
        uncertainty: Per-row uncertainty score, lower = accepted first.

    Returns:
        A 1-D float64 view of ``uncertainty``.

    Raises:
        ValueError: If the input is empty or contains non-finite values.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if u.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(u)):
        raise ValueError("uncertainty contains non-finite values")
    return u


def rc_curve(
    uncertainty: np.ndarray, errors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Risk-coverage curve with tie blocks collapsed.

    Rows are accepted in order of increasing uncertainty. Returns
    ``(coverage, selective_risk, generalized_risk, thresholds)``, each of
    length m = the number of DISTINCT uncertainty values, ordered from
    smallest coverage (most confident operating point) to coverage 1.0.
    Operating point j accepts exactly ``{i : uncertainty_i <= thresholds[j]}``
    -- note the direction: rows are ranked by uncertainty here, so the
    threshold is an upper bound where a confidence-based formulation would
    use a lower one. No operating point inside a tie block is ever
    evaluated, so the curve is a property of the predictions and not of the
    row order. Block sizes are recoverable as ``n * diff(coverage, prepend=0)``
    -- the prepend matters, since a bare ``diff`` drops the first block.
    """
    u = _as_score(uncertainty)
    loss = np.asarray(errors, dtype=np.float64).ravel()
    n = u.size

    order = np.argsort(u, kind="stable")
    u_sorted = u[order]
    cum_loss = np.cumsum(loss[order])

    # last index of each tie block: positions where the next value differs
    block_end = np.flatnonzero(np.diff(u_sorted)) if n > 1 else np.array([], dtype=int)
    idx = np.concatenate([block_end, [n - 1]])

    k = (idx + 1).astype(np.float64)  # number of accepted samples
    coverage = k / n
    selective = cum_loss[idx] / k
    generalized = cum_loss[idx] / n
    thresholds = u_sorted[idx]
    return coverage, selective, generalized, thresholds


def n_operating_points(uncertainty: np.ndarray) -> int:
    """Number of distinct uncertainty values, i.e. usable operating points.

    THE guard on every AURC in a table. The risk-coverage curve can only be
    evaluated at distinct scores, so this count is the true resolution of
    the curve: at ``n_operating_points == n`` the AURC is exact, and as the
    count falls towards 1 the AURC degenerates into the mean error rate. A
    row whose count is far below ``n`` has saturated -- almost always an
    fp16/fp32 softmax pinned at 1.0 -- and its AURC then moves with the
    temperature and with how the blocks happen to fall. Log it next to the
    AURC or the AURC cannot be read.

    Args:
        uncertainty: Per-row uncertainty score.

    Returns:
        The count of distinct values in ``uncertainty``.
    """
    return int(np.unique(_as_score(uncertainty)).size)


def _weighted_mean(
    values: np.ndarray, coverage: np.ndarray, n: int, weights: str = "block"
) -> float:
    """Mean over operating points under the named weighting.

    Args:
        values: Risk at each operating point.
        coverage: Coverage at each operating point.
        n: Number of rows the curve was built from.
        weights: ``"block"`` weights each point by its tie-block size (the
            weights then sum to n); ``"uniform"`` is the plain mean over the
            achievable coverage points. The two coincide when no two scores
            tie.

    Raises:
        ValueError: If ``weights`` is neither ``"block"`` nor ``"uniform"``.
    """
    if weights == "uniform":
        return float(values.mean())
    if weights == "block":
        k = np.rint(coverage * n)
        block_sizes = np.diff(np.concatenate([[0.0], k]))
        return float((values * block_sizes).sum() / block_sizes.sum())
    raise ValueError("weights must be 'block' or 'uniform'")


def aurc_from_scores(
    uncertainty: np.ndarray, errors: np.ndarray, weights: str = "block"
) -> float:
    """Area under the risk-coverage curve: weighted mean of selective risk
    over distinct operating points. Lower is better. Dominated by the
    low-coverage tail, where selective risk divides by a vanishing
    denominator -- the reason AUGRC exists alongside it. Read it beside
    :func:`n_operating_points`, which says how many points it averaged."""
    cov, sel, _, _ = rc_curve(uncertainty, errors)
    return _weighted_mean(sel, cov, np.asarray(uncertainty).size, weights)


def augrc_from_scores(
    uncertainty: np.ndarray, errors: np.ndarray, weights: str = "block"
) -> float:
    """Area under the GENERALIZED risk-coverage curve (arxiv 2407.01032):
    weighted mean of generalized risk, which divides by n rather than by the
    accepted count and so does not blow up as coverage goes to zero.
    Random-ranker anchor ~ r/2 against AURC's ~ r, for error rate r."""
    cov, _, gen, _ = rc_curve(uncertainty, errors)
    return _weighted_mean(gen, cov, np.asarray(uncertainty).size, weights)


def aurc_oracle(errors: np.ndarray, weights: str = "block") -> float:
    """AURC of the empirical oracle on THIS sample: same n, same errors,
    correct predictions ranked strictly first, no ties. With no ties the two
    weightings coincide, so ``weights`` is threaded only for symmetry with
    the AURC it is subtracted from."""
    loss = np.asarray(errors, dtype=np.float64).ravel()
    oracle_loss = np.sort(loss)  # zeros first == correct ranked first
    oracle_uncertainty = np.arange(loss.size, dtype=np.float64)  # no ties
    return aurc_from_scores(oracle_uncertainty, oracle_loss, weights)


def eaurc_from_scores(
    uncertainty: np.ndarray, errors: np.ndarray, weights: str = "block"
) -> float:
    """Excess AURC over the empirical oracle. Zero for a perfect ranker.

    Only PARTIALLY isolates ranking quality from the error rate; use it
    beside AURC as a within-model diagnostic and carry cross-model claims on
    AUROC(failure), which is rate-free by construction."""
    return aurc_from_scores(uncertainty, errors, weights) - aurc_oracle(errors, weights)


def operating_point_at_coverage(
    uncertainty: np.ndarray, errors: np.ndarray, target_coverage: float
) -> tuple[float, float, float, float]:
    """The achievable operating point at or below a target coverage.

    Args:
        uncertainty: Per-row uncertainty score, lower = accepted first.
        errors: Per-row 0/1 loss.
        target_coverage: Requested coverage, in (0, 1].

    Returns:
        ``(coverage, selective_risk, generalized_risk, threshold)`` at the
        largest ACHIEVABLE coverage <= target. The coverage returned is the
        achieved one, which can sit well below the target when a tie block
        straddles it -- report that number, never the target. A table
        printing ``risk@cov0.8`` without the coverage actually reached is
        not reporting a risk at 80% coverage.

        When NO operating point sits at or below the target -- every tie
        block is coarser than it, the extreme case being one block covering
        everything -- there is nothing at or below to return, so the
        SMALLEST achievable coverage comes back and it is ABOVE the target.
        That row is not the risk that was asked for, and the returned
        coverage is the only thing that says so: against a fully saturated
        score the fallback hands back the full-coverage error rate under a
        ``risk@cov0.5`` heading. This is why the achieved coverage travels
        beside the risk everywhere in this module, and why
        `n_operating_points` is logged next to AURC.

    Raises:
        ValueError: If ``target_coverage`` is outside (0, 1].
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    cov, sel, gen, thr = rc_curve(uncertainty, errors)
    ok = np.flatnonzero(cov <= target_coverage + 1e-12)
    # `ok` is empty only when even the most selective operating point already
    # covers more than the target; index 0 is then the closest available.
    j = int(ok[-1]) if ok.size else 0
    return float(cov[j]), float(sel[j]), float(gen[j]), float(thr[j])


def risk_at_coverage(
    uncertainty: np.ndarray, errors: np.ndarray, target_coverage: float
) -> float:
    """Selective risk at the largest achievable coverage <= target. Always
    report :func:`operating_point_at_coverage`'s achieved coverage next to
    it: under heavy ties the achieved coverage can sit far below the target
    -- or, when no operating point reaches the target at all, above it -- and
    the bare number carries no sign of either."""
    return operating_point_at_coverage(uncertainty, errors, target_coverage)[1]


def coverage_at_risk(
    uncertainty: np.ndarray, errors: np.ndarray, max_risk: float
) -> float:
    """Largest coverage whose selective risk <= ``max_risk``; 0.0 if no
    operating point qualifies. The deployment question: how much traffic can
    be auto-decided while holding the error rate under the moderation SLA."""
    cov, sel, _, _ = rc_curve(uncertainty, errors)
    ok = np.flatnonzero(sel <= max_risk + 1e-12)
    return float(cov[ok].max()) if ok.size else 0.0


class AURiskCoverage(Metric):
    """Base class for Area Under the (Generalized) Risk-Coverage Curve
    metrics. See the module docstring for the estimator convention.

    Args:
        weights: ``"block"`` (default) or ``"uniform"``; see
            :func:`_weighted_mean`.
    """

    def __init__(self, weights: str = "block", **kwargs):
        super().__init__(**kwargs)
        if weights not in ("block", "uniform"):
            raise ValueError("weights must be 'block' or 'uniform'")
        self.weights = weights
        self.probs: list[Tensor]
        self.target: list[Tensor]
        self.scores: list[Tensor]
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor, uncertainty_scores: Tensor):
        # float64 at the boundary: the ranking below is only as fine-grained
        # as the score, and an fp16/fp32 score that saturated at the softmax
        # collapses rows the model ranked apart into one tie block. Casting
        # here keeps every later operation exact; see _as_score.
        self.probs.append(probs.detach().cpu().double())
        self.target.append(targets.detach().cpu())
        self.scores.append(uncertainty_scores.detach().cpu().double())

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return aurc_from_scores(scores, errors, self.weights)

    def compute(self) -> float:
        # An empty split is undefined, not an error: a geometry filter or a
        # per-class breakout can legitimately select no rows. `torch.cat` on
        # an empty list raises, so this has to be tested first, and NaN is the
        # honest answer -- the same one the failure-detection metrics give.
        if not self.probs:
            return float("nan")
        probs = torch.cat(self.probs).cpu().numpy()
        targets = torch.cat(self.target).cpu().numpy()
        scores = torch.cat(self.scores).squeeze().cpu().numpy()

        preds = np.argmax(probs, axis=1).astype(np.int64)
        errors = (preds != targets).astype(np.float64)
        return self._value(scores, errors)


class AUGRiskCoverage(AURiskCoverage):
    """Area Under the Generalized Risk-Coverage Curve.

    Source: https://arxiv.org/abs/2407.01032
    """

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return augrc_from_scores(scores, errors, self.weights)


class EAURiskCoverage(AURiskCoverage):
    """Excess AURC over the empirical oracle at the same n."""

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return eaurc_from_scores(scores, errors, self.weights)


class NumOperatingPoints(AURiskCoverage):
    """Distinct operating points behind the AURC logged beside it.

    Logged as a float so it rides in the same collection; read it as a
    count. Equal to n means the AURC is exact; far below n means the score
    saturated and the AURC is a temperature-dependent approximation. See
    :func:`n_operating_points`.
    """

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return float(n_operating_points(scores))


class RiskAtCoverage(AURiskCoverage):
    """Selective risk at the largest achievable coverage <= ``coverage``.

    Log :class:`AchievedCoverage` at the same target beside it: the achieved
    coverage is the honest one and it can sit below the target.

    Args:
        coverage: Target coverage, in (0, 1].
        weights: Unused for a single operating point; accepted so the class
            drops into the same collection as the AURC metrics.
    """

    def __init__(self, coverage: float, weights: str = "block", **kwargs):
        super().__init__(weights=weights, **kwargs)
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        self.coverage = coverage

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return operating_point_at_coverage(scores, errors, self.coverage)[1]


class AchievedCoverage(RiskAtCoverage):
    """Coverage actually reached when asking for ``coverage``.

    Equal to the target only when an operating point lands exactly there;
    under heavy ties it can be far lower, which is precisely when the
    accompanying ``risk@cov`` must not be read as a risk at the target.
    """

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return operating_point_at_coverage(scores, errors, self.coverage)[0]


def _coverage_suffix(coverage: float) -> str:
    """Collection-key suffix for a coverage target.

    ``MetricCollection`` is a ``ModuleDict``, whose keys may not contain a
    ``"."``, so 0.8 is keyed ``0_8``: ``risk@cov0_8`` here where a flat
    result dict would write ``risk@cov0.8``.
    """
    return f"{coverage:g}".replace(".", "_")


def get_selective_classification_metrics(
    coverages: tuple[float, ...] = (0.8, 0.5), weights: str = "block"
) -> MetricCollection:
    """
    Get metrics for multiclass selective classification.

    Args:
        coverages: Coverage targets to report a selective risk at. Each one
            contributes a ``risk@cov<c>`` and the ``achieved_cov@<c>`` that
            qualifies it.
        weights: Operating-point weighting for AURC/AUGRC/E-AURC.

    Returns:
        MetricCollection: AURC, AUGRC, E-AURC, the operating-point count
        that says whether those three are exact, and one risk / achieved
        coverage pair per target.
    """
    metrics: dict[str, Metric] = {
        "aurc": AURiskCoverage(weights=weights),
        "augrc": AUGRiskCoverage(weights=weights),
        "eaurc": EAURiskCoverage(weights=weights),
        "n_operating_points": NumOperatingPoints(),
    }
    for coverage in coverages:
        suffix = _coverage_suffix(coverage)
        metrics[f"risk@cov{suffix}"] = RiskAtCoverage(coverage)
        metrics[f"achieved_cov@{suffix}"] = AchievedCoverage(coverage)
    return MetricCollection(metrics)
