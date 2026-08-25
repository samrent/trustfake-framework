"""Two AUROCs that are not the same number, and must never share a column.

``fd_auroc`` -- AUROC of FAILURE PREDICTION: does the uncertainty score rank
the model's own mistakes above its correct predictions. 0.5 is chance; a
confidence attack drives it toward 0, which is worse than useless because
the abstention rule then preferentially rejects the predictions that were
right.

``detection_auroc`` -- AUROC of the DETECTION TASK itself: does p(fake) rank
fakes above reals. That is the task metric; the one above is a metric about
the model's self-knowledge. A bare "AUROC" in a table is ambiguous and
reporting one under the other's name is the cheap, embarrassing error, so
both are named in full here and every table must say which it shows.

Both report NaN rather than a number when the quantity is undefined -- see
:class:`FailureAUROC`.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
)
from torchmetrics.utilities.data import dim_zero_cat

__all__ = [
    "FailureAUROC",
    "DetectionAUROC",
    "get_failure_detection_metrics",
    "get_detection_metrics",
]


class FailureAUROC(BinaryAUROC):
    """BinaryAUROC over (uncertainty, errors) that reports NaN when the
    AUROC is undefined, instead of 0.0.

    torchmetrics returns 0.0 for a split with no errors, with no correct
    predictions, or with a constant score. 0.0 is not "undefined" -- in this
    column it reads as *perfectly inverted* failure detection, the signature
    of a successful confidence attack, on a split that merely had nothing to
    rank. NaN cannot be averaged into a table by accident; 0.0 can, and it
    drags the mean toward the attacked regime.

    The constant-score case is the same saturation ``n_operating_points``
    catches on the AURC side: an fp16/fp32 confidence pinned at 1.0 has
    nothing left to rank with.
    """

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate one batch of (uncertainty, errors).

        The score is upcast to float64 first: a ranking is only as
        fine-grained as the score, and a confidence left in fp32 has already
        thrown away the ordering between its most confident rows.
        """
        super().update(preds.detach().double(), target)

    def compute(self) -> Tensor:
        """AUROC, or NaN where it is undefined."""
        if self.thresholds is not None:  # binned mode keeps no raw scores
            return super().compute()
        preds, target = dim_zero_cat(self.preds), dim_zero_cat(self.target)
        undefined = (
            preds.numel() == 0
            or bool(torch.min(target) == torch.max(target))  # no errors, or all errors
            or bool(torch.min(preds) == torch.max(preds))  # constant score
        )
        if undefined:
            return torch.tensor(float("nan"), device=preds.device)
        return super().compute()


class DetectionAUROC(FailureAUROC):
    """AUROC of the detection task: does p(fake) rank fakes above reals.

    Consumes ``(probs, targets)``, not ``(uncertainty, errors)``, which is
    why it cannot ride in the failure-detection collection and gets
    :func:`get_detection_metrics` of its own. Following the repo-wide
    convention (``trustfake.metrics.moderation``) every class other than
    ``real_class`` is fake, so ``p_fake = 1 - P(real)``.

    Args:
        real_class: Index of the real class; SID-Set uses 0, with synthetic
            and tampered both folding to fake.
    """

    def __init__(self, real_class: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.real_class = real_class

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Accumulate one batch of class probabilities and integer labels."""
        p_fake = 1.0 - probs.detach().double()[:, self.real_class]
        y_binary = (targets.detach().long() != self.real_class).long()
        super().update(p_fake, y_binary)


def get_failure_detection_metrics() -> MetricCollection:
    """
    Get metrics for failure detection.

    Returns:
        MetricCollection: AUROC of predicting the model's own errors from
        its uncertainty score, NaN where undefined.
    """
    return MetricCollection(
        {
            "fd_auroc": FailureAUROC(),
        }
    )


def get_detection_metrics(real_class: int = 0) -> MetricCollection:
    """
    Get metrics for the detection task itself (fake vs real).

    Separate from :func:`get_failure_detection_metrics` because it consumes
    ``(probs, targets)`` rather than ``(uncertainty, errors)`` -- and
    because keeping the two AUROCs apart is the whole point.

    Args:
        real_class: Index of the real class; every other class is fake.

    Returns:
        MetricCollection: AUROC of ranking fakes above reals by p(fake).
    """
    return MetricCollection(
        {
            "detection_auroc": DetectionAUROC(real_class=real_class),
        }
    )
