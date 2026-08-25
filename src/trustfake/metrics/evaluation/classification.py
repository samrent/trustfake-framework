"""Classification metrics, reported at two averages that are not the same number.

``accuracy`` is the MACRO average: the mean of the per-class recalls, which
is the right headline on SID-Set's unbalanced classes because it refuses to
let the majority class carry the score.

``accuracy_top1`` is plain top-1: the fraction of rows predicted correctly.
It is reported alongside because the claims made about attacks are top-1
claims -- "accuracy identical to 4 dp under ACE" is a statement about the
argmax on every row, and macro accuracy can move while top-1 does not (and
the reverse) whenever the errors are unevenly spread across classes.
"""

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)


def get_multiclass_classification_metrics(num_classes: int) -> MetricCollection:
    """
    Get metrics for multiclass classification.

    Returns:
        MetricCollection: macro precision / recall / F1 and accuracy at both
        the macro and the top-1 (micro) average. See the module docstring
        for why both accuracies are reported.
    """
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
            # "accuracy_class": ClasswiseWrapper(
            #     Accuracy(task="multiclass", num_classes=num_classes, average="none"),
            #     prefix="accuracy_class_",
            # ),
            # "precision_class": ClasswiseWrapper(
            #     Precision(task="multiclass", num_classes=num_classes, average="none"),
            #     prefix="precision_class_",
            # ),
            # "recall_class": ClasswiseWrapper(
            #     Recall(task="multiclass", num_classes=num_classes, average="none"),
            #     prefix="recall_class_",
            # ),
            # "f1_score_class": ClasswiseWrapper(
            #     F1Score(task="multiclass", num_classes=num_classes, average="none"),
            #     prefix="f1_score_class_",
            # ),
        }
    )
