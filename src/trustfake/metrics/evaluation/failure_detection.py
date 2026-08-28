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
``detection_auroc_<class>`` asks the task question of ONE fake modality at a
time (tampered-vs-real, synthetic-vs-real), because the fold's average is
carried by its easiest member -- see :class:`DetectionAUROC`.

Both report NaN rather than a number when the quantity is undefined -- see
:class:`FailureAUROC`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
)
from torchmetrics.utilities.data import dim_zero_cat

from trustfake.metrics.evaluation.classification import resolve_class_names

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
        # Test for an empty split BEFORE concatenating: `dim_zero_cat` raises
        # on an empty list, so the numel() guard below could never be reached.
        # An empty split is undefined, not an error -- a geometry filter or a
        # per-class breakout can legitimately select no rows, and that should
        # report NaN like every other undefined case rather than abort a run
        # that has already done all its work.
        if not self.preds:
            return torch.tensor(float("nan"))
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

    With ``fake_class`` set, the SAME question is asked of one fake modality
    at a time: rows are restricted to {real, that class} and the score stays
    ``p_fake`` -- the quantity the moderation thresholds actually read -- so
    the row answers "can the deployed score separate THIS kind of fake from
    real", not "can some other head". The all-fakes fold cannot answer it:
    on SID-Set the synthetic half is the easy half and carries the average,
    so a tampered-vs-real ranking near chance hides inside a fold that still
    looks healthy. A split holding no rows of the modality reports NaN, like
    every other undefined case (see :class:`FailureAUROC`).

    Args:
        real_class: Index of the real class; SID-Set uses 0, with synthetic
            and tampered both folding to fake.
        fake_class: Restrict the positive rows to this single fake class
            (reals stay the negatives). ``None`` (default) folds every
            non-real class to fake.
    """

    def __init__(self, real_class: int = 0, fake_class: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if fake_class is not None and fake_class == real_class:
            raise ValueError(
                f"fake_class == real_class ({real_class}): the breakout would "
                "compare the real class against itself."
            )
        self.real_class = real_class
        self.fake_class = fake_class

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Accumulate one batch of class probabilities and integer labels."""
        probs = probs.detach().double()
        targets = targets.detach().long()
        if self.fake_class is not None:
            keep = (targets == self.real_class) | (targets == self.fake_class)
            probs, targets = probs[keep], targets[keep]
        p_fake = 1.0 - probs[:, self.real_class]
        y_binary = (targets != self.real_class).long()
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


def get_detection_metrics(
    real_class: int = 0,
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
) -> MetricCollection:
    """
    Get metrics for the detection task itself (fake vs real).

    Separate from :func:`get_failure_detection_metrics` because it consumes
    ``(probs, targets)`` rather than ``(uncertainty, errors)`` -- and
    because keeping the two AUROCs apart is the whole point.

    Args:
        real_class: Index of the real class; every other class is fake.
        num_classes: When given, one per-modality breakout is added per fake
            class (``detection_auroc_synthetic``, ``detection_auroc_tampered``
            on SID-Set) beside the all-fakes fold. ``None`` keeps the fold
            alone -- the pre-breakout behaviour.
        class_names: Names for the breakout keys; see
            :func:`~trustfake.metrics.evaluation.classification.resolve_class_names`.

    Returns:
        MetricCollection: AUROC of ranking fakes above reals by p(fake),
        plus (with ``num_classes``) the same question per fake modality.
    """
    metrics: dict[str, DetectionAUROC] = {
        "detection_auroc": DetectionAUROC(real_class=real_class),
    }
    if num_classes is not None:
        names = resolve_class_names(num_classes, class_names)
        for index in range(num_classes):
            if index == real_class:
                continue
            metrics[f"detection_auroc_{names[index]}"] = DetectionAUROC(
                real_class=real_class, fake_class=index
            )
    # compute_groups=False: the breakouts differ from the fold only in which
    # rows they kept, which is exactly the kind of same-shaped state the
    # grouping optimisation exists to merge. Three binary AUROCs are not
    # worth risking a silent state share for.
    return MetricCollection(metrics, compute_groups=False)
