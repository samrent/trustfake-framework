"""Classification metrics, reported at two averages that are not the same number.

``accuracy`` is the MACRO average: the mean of the per-class recalls, which
is the right headline on SID-Set's unbalanced classes because it refuses to
let the majority class carry the score.

``accuracy_top1`` is plain top-1: the fraction of rows predicted correctly.
It is reported alongside because the claims made about attacks are top-1
claims -- "accuracy identical to 4 dp under ACE" is a statement about the
argmax on every row, and macro accuracy can move while top-1 does not (and
the reverse) whenever the errors are unevenly spread across classes.

The PER-CLASS rows (``recall_real`` / ``recall_synthetic`` /
``recall_tampered``, with precision beside each) exist because SID-Set's
observed failure mode is per-class and both averages hide WHICH class failed.
A detector whose tampered recall has collapsed while synthetic recall is 0.99
posts a macro accuracy that reads as "somewhat degraded" and a detection
AUROC that moves without saying which class moved it -- tampered is a tenth
of the rows and half of the fakes, so every fold dilutes it. The per-class
recall row is the only place in the table where "the model does not detect
tampering" is legible as itself. Precision is reported beside it because the
confusion is asymmetric: a tampered image predicted synthetic and a synthetic
image predicted tampered are different errors with the same macro cost, and
on the fake-vs-real fold the first one is silently forgiven (it still counts
as a caught fake).

Per-class ACCURACY and F1 are deliberately absent: multiclass accuracy at
``average="none"`` IS per-class recall (the same confusion-matrix row twice
under two names), and F1 is derivable from the two columns that are reported.
"""

from collections.abc import Sequence

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.wrappers import ClasswiseWrapper

__all__ = [
    "SID_SET_CLASS_NAMES",
    "resolve_class_names",
    "get_multiclass_classification_metrics",
]

#: SID-Set's label order. The same convention is already hard-coded across
#: the repo as ``real_class = 0`` (`trustfake.metrics.moderation`,
#: `DetectionAUROC`); this names the other two so a logged key says
#: ``recall_tampered`` rather than ``recall_2``.
SID_SET_CLASS_NAMES: tuple[str, ...] = ("real", "synthetic", "tampered")


def resolve_class_names(
    num_classes: int, class_names: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Names used in per-class metric keys.

    ``None`` resolves to :data:`SID_SET_CLASS_NAMES` when the class count
    matches (the repo's one dataset), and to ``class_<i>`` otherwise -- so a
    non-SID-Set head never silently borrows SID-Set's semantics. An explicit
    list must match ``num_classes`` exactly: a mismatch would not crash, it
    would mislabel every per-class row, which is worse.
    """
    if class_names is None:
        if num_classes == len(SID_SET_CLASS_NAMES):
            return SID_SET_CLASS_NAMES
        return tuple(f"class_{i}" for i in range(num_classes))
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries for {num_classes} "
            "classes -- every per-class key would be mislabelled."
        )
    return tuple(class_names)


def get_multiclass_classification_metrics(
    num_classes: int, class_names: Sequence[str] | None = None
) -> MetricCollection:
    """
    Get metrics for multiclass classification.

    Args:
        num_classes: Number of classes.
        class_names: Names for the per-class keys; see
            :func:`resolve_class_names` for the default.

    Returns:
        MetricCollection: macro precision / recall / F1, accuracy at both the
        macro and the top-1 (micro) average, and per-class recall / precision
        under named keys (``recall_tampered``, ...). See the module docstring
        for why the averages AND the per-class rows are all reported.
    """
    names = list(resolve_class_names(num_classes, class_names))
    return MetricCollection(
        {
            "accuracy": Accuracy(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "accuracy_top1": Accuracy(
                task="multiclass", num_classes=num_classes, average="micro"
            ),
            "precision": Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "recall": Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "f1_score": F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "recall_class": ClasswiseWrapper(
                Recall(task="multiclass", num_classes=num_classes, average="none"),
                labels=names,
                prefix="recall_",
            ),
            "precision_class": ClasswiseWrapper(
                Precision(task="multiclass", num_classes=num_classes, average="none"),
                labels=names,
                prefix="precision_",
            ),
        }
    )
