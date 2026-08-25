"""Selective-classification metrics: AURC, AUGRC and E-AURC.

Convention (it matters, and it is asserted by the tests): the risk-coverage
curve is evaluated at DISTINCT operating points only -- tie blocks in the
uncertainty score are collapsed, because a threshold inside a tie block
selects a permutation-dependent set of rows. AURC/AUGRC are then the
block-size-weighted mean over those operating points. A naive cumulative
mean over every rank varies across row permutations of the same predictions
(saturated softmax manufactures exactly such tie blocks), and trapezoid
integration over coverage is a third convention in the literature; neither
is used here.

E-AURC subtracts the AURC of the EMPIRICAL oracle -- the same n and the same
errors, correct predictions ranked first -- never the asymptotic closed form
``r + (1-r)ln(1-r)``, which is off by ~1e-4 at n=1000, the same order as the
effects being reported.
"""

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

__all__ = [
    "AURiskCoverage",
    "AUGRiskCoverage",
    "EAURiskCoverage",
    "get_selective_classification_metrics",
    "rc_curve",
    "aurc_from_scores",
    "augrc_from_scores",
    "aurc_oracle",
    "eaurc_from_scores",
]


def rc_curve(
    uncertainty: np.ndarray, errors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Risk-coverage curve with tie blocks collapsed.

    Rows are accepted in order of increasing uncertainty. Returns
    ``(coverage, selective_risk, generalized_risk)``, each of length m = the
    number of DISTINCT uncertainty values, ordered from smallest coverage
    (most confident operating point) to coverage 1.0. No operating point
    inside a tie block is ever evaluated, so the curve is a property of the
    predictions and not of the row order. Block sizes are recoverable as
    ``n * diff(coverage)``.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    loss = np.asarray(errors, dtype=np.float64).ravel()
    n = u.size
    if n == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(u)):
        raise ValueError("uncertainty contains non-finite values")

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
    return coverage, selective, generalized


def _block_weighted_mean(values: np.ndarray, coverage: np.ndarray, n: int) -> float:
    """Mean over operating points weighted by tie-block size; weights sum to n."""
    k = np.rint(coverage * n)
    weights = np.diff(np.concatenate([[0.0], k]))
    return float((values * weights).sum() / weights.sum())


def aurc_from_scores(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """Area under the risk-coverage curve: block-weighted mean of selective
    risk over distinct operating points. Lower is better. Dominated by the
    low-coverage tail, where selective risk divides by a vanishing
    denominator -- the reason AUGRC exists alongside it."""
    cov, sel, _ = rc_curve(uncertainty, errors)
    return _block_weighted_mean(sel, cov, np.asarray(uncertainty).size)


def augrc_from_scores(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """Area under the GENERALIZED risk-coverage curve (arxiv 2407.01032):
    block-weighted mean of generalized risk, which divides by n rather than
    by the accepted count and so does not blow up as coverage goes to zero.
    Random-ranker anchor ~ r/2 against AURC's ~ r, for error rate r."""
    cov, _, gen = rc_curve(uncertainty, errors)
    return _block_weighted_mean(gen, cov, np.asarray(uncertainty).size)


def aurc_oracle(errors: np.ndarray) -> float:
    """AURC of the empirical oracle on THIS sample: same n, same errors,
    correct predictions ranked strictly first, no ties."""
    loss = np.asarray(errors, dtype=np.float64).ravel()
    oracle_loss = np.sort(loss)  # zeros first == correct ranked first
    oracle_uncertainty = np.arange(loss.size, dtype=np.float64)  # no ties
    return aurc_from_scores(oracle_uncertainty, oracle_loss)


def eaurc_from_scores(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """Excess AURC over the empirical oracle. Zero for a perfect ranker.

    Only PARTIALLY isolates ranking quality from the error rate; use it
    beside AURC as a within-model diagnostic and carry cross-model claims on
    AUROC(failure), which is rate-free by construction."""
    return aurc_from_scores(uncertainty, errors) - aurc_oracle(errors)


class AURiskCoverage(Metric):
    """Base class for Area Under the (Generalized) Risk-Coverage Curve
    metrics. See the module docstring for the estimator convention."""

    def __init__(self):
        super().__init__()
        self.probs: list[Tensor]
        self.target: list[Tensor]
        self.scores: list[Tensor]
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor, uncertainty_scores: Tensor):
        self.probs.append(probs.detach().cpu())
        self.target.append(targets.detach().cpu())
        self.scores.append(uncertainty_scores.detach().cpu())

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return aurc_from_scores(scores, errors)

    def compute(self) -> float:
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
        return augrc_from_scores(scores, errors)


class EAURiskCoverage(AURiskCoverage):
    """Excess AURC over the empirical oracle at the same n."""

    def _value(self, scores: np.ndarray, errors: np.ndarray) -> float:
        return eaurc_from_scores(scores, errors)


def get_selective_classification_metrics() -> MetricCollection:
    """
    Get metrics for multiclass selective classification.
    """
    return MetricCollection(
        {
            "aurc": AURiskCoverage(),
            "augrc": AUGRiskCoverage(),
            "eaurc": EAURiskCoverage(),
        }
    )
