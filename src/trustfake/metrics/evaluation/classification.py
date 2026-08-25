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
    """
    return MetricCollection(
        {
            "accuracy": Accuracy(
                task="multiclass", num_classes=num_classes, average="macro"
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
