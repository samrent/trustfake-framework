"""Calibration scores: ECE (binned), NLL and Brier (unbiased, no binning).

All three consume the model's predicted probabilities and the labels. ECE is
sensitive to its binning scheme and domain, so it is reported alongside NLL
and Brier, which are proper scoring rules with no binning. All are multiclass
(3-class here: real / synthetic / tampered).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassCalibrationError

__all__ = ["NegativeLogLikelihood", "BrierScore", "get_calibration_metrics"]


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
    such parameter.
    """
    return MetricCollection(
        {
            "ece": MulticlassCalibrationError(
                num_classes=num_classes, n_bins=n_bins, norm="l1"
            ),
            "nll": NegativeLogLikelihood(),
            "brier": BrierScore(num_classes=num_classes),
        }
    )
