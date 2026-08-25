from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
)


def get_failure_detection_metrics() -> MetricCollection:
    """
    Get metrics for failure detection.
    """
    return MetricCollection(
        {
            "fd_auroc": BinaryAUROC(),
        }
    )
