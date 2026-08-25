"""Calibration scores: ECE (binned), NLL and Brier (unbiased, no binning).

All three consume the model's predicted probabilities and the labels. ECE is
sensitive to its binning scheme and domain, so it is reported alongside NLL
and Brier, which are proper scoring rules with no binning. All are multiclass
(3-class here: real / synthetic / tampered).

Both ECE knobs are part of the definition and must be stated wherever the
number is printed:

    ``scheme="equal_width"`` bins uniformly over ``domain``; ``"equal_mass"``
        uses quantile edges, so every bin carries the same number of rows.
        They disagree by more than rounding on a skewed confidence
        distribution -- which is every trained classifier -- because
        equal-width puts almost all the mass in the top bin and then reports
        that bin's average gap as if it were resolved.
    ``domain`` bounds the equal-width grid. Top-label confidence for C
        classes lives in [1/C, 1], so a 15-bin equal-width grid over [0, 1]
        leaves 7 bins empty in the binary case and the resulting number is
        not comparable to a multiclass ECE computed the same way.

ECE is also biased upward by binning, and the bias grows with the bin count:
a perfectly calibrated binary model measures ~0.0056 at 5 bins, ~0.0102 at
15 and ~0.0573 at 200. Report NLL and Brier next to it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassCalibrationError

__all__ = [
    "NegativeLogLikelihood",
    "BrierScore",
    "ExpectedCalibrationError",
    "ece_from_scores",
    "get_calibration_metrics",
]


def ece_from_scores(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    scheme: str = "equal_width",
    domain: tuple[float, float] = (0.0, 1.0),
    norm: str = "l1",
) -> float:
    """Expected calibration error from top-label confidence and correctness.

    Args:
        confidence: Top-label probability per row.
        correct: 1.0 where the top-label prediction was right, else 0.0.
        n_bins: Number of bins.
        scheme: ``"equal_width"`` (uniform edges over ``domain``) or
            ``"equal_mass"`` (quantile edges, equal counts per bin).
        domain: Bounds of the equal-width grid; ignored for equal mass.
        norm: ``"l1"`` is the mass-weighted mean gap (the usual ECE),
            ``"l2"`` its root-mean-square, ``"max"`` the largest gap over
            bins (MCE), which is a worst-bin statistic and not an average.

    Returns:
        The calibration error under those conventions.

    Raises:
        ValueError: If ``scheme`` or ``norm`` is not one of the above.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    acc = np.asarray(correct, dtype=np.float64).ravel()
    n = conf.size

    if scheme == "equal_width":
        edges = np.linspace(domain[0], domain[1], n_bins + 1)
    elif scheme == "equal_mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        # open the outer edges so nothing falls off either end
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        raise ValueError("scheme must be 'equal_width' or 'equal_mass'")
    if norm not in ("l1", "l2", "max"):
        raise ValueError("norm must be 'l1', 'l2' or 'max'")

    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    gaps, props = [], []
    for b in range(n_bins):
        in_bin = idx == b
        count = int(in_bin.sum())
        if count:
            gaps.append(abs(acc[in_bin].mean() - conf[in_bin].mean()))
            props.append(count / n)
    if not gaps:
        return 0.0

    gap = np.asarray(gaps)
    prop = np.asarray(props)
    if norm == "l1":
        return float((gap * prop).sum())
    if norm == "l2":
        return float(np.sqrt((gap**2 * prop).sum()))
    return float(gap.max())


class ExpectedCalibrationError(Metric):
    r"""ECE over top-label confidence, with the scheme/domain/norm exposed.

    The defaults reproduce ``MulticlassCalibrationError(norm="l1")``: 15
    equal-width bins over [0, 1]. The variants exist because that default is
    a choice, not a definition -- see the module docstring.

    Args:
        n_bins: Number of bins.
        scheme: ``"equal_width"`` or ``"equal_mass"``.
        domain: Bounds of the equal-width grid.
        norm: ``"l1"``, ``"l2"`` or ``"max"``.
    """

    higher_is_better = False

    def __init__(
        self,
        n_bins: int = 15,
        scheme: str = "equal_width",
        domain: tuple[float, float] = (0.0, 1.0),
        norm: str = "l1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.scheme = scheme
        self.domain = domain
        self.norm = norm
        self.confidence: list[Tensor]
        self.correct: list[Tensor]
        self.add_state("confidence", default=[], dist_reduce_fx="cat")
        self.add_state("correct", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Accumulate top-label confidence and correctness for one batch.

        Kept in float64: quantile edges and bin means are differences of
        near-1.0 numbers, where fp32 cancellation is the same order as the
        gaps being measured.
        """
        top = probs.detach().double().max(dim=1)
        self.confidence.append(top.values.cpu())
        self.correct.append((top.indices == targets.detach().long()).double().cpu())

    def compute(self) -> Tensor:
        conf = torch.cat(self.confidence).numpy()
        correct = torch.cat(self.correct).numpy()
        return torch.tensor(
            ece_from_scores(
                conf, correct, self.n_bins, self.scheme, self.domain, self.norm
            )
        )


class NegativeLogLikelihood(Metric):
    r"""Mean negative log-likelihood ``-mean(log p[target])``.

    A proper scoring rule with no binning. Lower is better. Moves with
    temperature (unlike accuracy), which is why it is reported next to it.
    """

    higher_is_better = False

    def __init__(self, eps: float = 1e-12, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        p = probs.gather(1, targets.long().view(-1, 1)).squeeze(1).clamp_min(self.eps)
        self.nll_sum += -torch.log(p).sum()
        self.count += targets.numel()

    def compute(self) -> Tensor:
        return self.nll_sum / self.count


class BrierScore(Metric):
    r"""Multiclass Brier score ``mean(sum_c (p_c - y_c)^2)`` with one-hot ``y``.

    A proper scoring rule with no binning. Lower is better.
    """

    higher_is_better = False

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("brier_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, targets.long().view(-1, 1), 1.0)
        self.brier_sum += (probs - onehot).pow(2).sum()
        self.count += targets.numel()

    def compute(self) -> Tensor:
        return self.brier_sum / self.count


def get_calibration_metrics(num_classes: int, n_bins: int = 15) -> MetricCollection:
    """
    ECE (L1 calibration error, `n_bins` equal-width bins), NLL and Brier.

    ECE's binning scheme and bin count are part of its definition and are
    fixed here so reported numbers are comparable; NLL and Brier carry no
    such parameter. ``ece`` stays the equal-width-over-[0,1] L1 number, and
    ``ece_equal_mass`` is reported beside it because the two disagree on a
    skewed confidence distribution and only reporting the flattering one is
    a choice. The binary-domain variant is added for two classes, where
    equal-width over [0, 1] wastes half its bins; the same argument gives a
    [1/C, 1] domain for C classes, which callers can build directly with
    :class:`ExpectedCalibrationError`.

    Returns:
        MetricCollection: the ECE variants, NLL and Brier.
    """
    metrics: dict[str, Metric] = {
        "ece": MulticlassCalibrationError(
            num_classes=num_classes, n_bins=n_bins, norm="l1"
        ),
        "ece_equal_mass": ExpectedCalibrationError(n_bins=n_bins, scheme="equal_mass"),
        "nll": NegativeLogLikelihood(),
        "brier": BrierScore(num_classes=num_classes),
    }
    if num_classes == 2:
        metrics["ece_binary_domain"] = ExpectedCalibrationError(
            n_bins=n_bins, scheme="equal_width", domain=(0.5, 1.0)
        )
    return MetricCollection(metrics)
