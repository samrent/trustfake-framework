"""
This module defines a standard training module for classification tasks.
The standard training module is designed to perform "classical" training without
attacks.
"""

from abc import ABC, abstractmethod

import lightning as pl
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from trustfake.metrics.evaluation import (
    get_failure_detection_metrics,
    get_multiclass_classification_metrics,
    get_selective_classification_metrics,
)
from trustfake.models.wrapper import TrustFakeWrapper
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = ["TrainingModule"]


class TrainingModule(ABC, pl.LightningModule):
    """Default training Module for classification.

    This abstract module serves as a base class for specific training modules.
    It defines the essential structure and methods that any derived training module
    must implement.

    Loss computation and metric computation are kept separate (`compute_loss` and
    `update_metrics`) so that subclasses can override either one independently,
    depending on which part of the training logic they need to change.

    .. note::
        The forward pass returns logits, probabilities, label predictions and
        uncertainty.
    """

    def __init__(
        self,
        model: TrustFakeWrapper,
        num_classes: int,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
    ):
        """Initialize the standard training module."""
        super().__init__()
        self.model = model
        self._num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler

        # == Metrics Setup == #
        self.train_classification_metrics = self.classification_metrics.clone(
            prefix="train_"
        )
        self.val_classification_metrics = self.classification_metrics.clone(
            prefix="val_"
        )

        self.train_fd_metrics = self.failure_detection_metrics.clone(prefix="train_")
        self.val_fd_metrics = self.failure_detection_metrics.clone(prefix="val_")

        self.train_selective_classification_metrics = (
            self.selective_classification_metrics.clone(prefix="train_")
        )
        self.val_selective_classification_metrics = (
            self.selective_classification_metrics.clone(prefix="val_")
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.model.on_train_epoch_start()

    def on_train_batch_end(
        self,
        outputs: tuple[Tensor, Tensor, Tensor],
        batch: list[Tensor, Tensor],
        batch_idx: int,
    ):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.model.on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        self._compute_and_log_metrics(
            self.train_classification_metrics,
            self.train_fd_metrics,
            self.train_selective_classification_metrics,
        )
        self.model.on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics(
            self.val_classification_metrics,
            self.val_fd_metrics,
            self.val_selective_classification_metrics,
        )
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr_{i}", param_group["lr"], on_epoch=True, prog_bar=False)

    def _compute_and_log_metrics(
        self,
        classification_metrics: MetricCollection,
        fd_metrics: MetricCollection,
        selective_classification_metrics: MetricCollection,
    ):
        self.log_dict(classification_metrics.compute())
        classification_metrics.reset()
        self.log_dict(fd_metrics.compute())
        fd_metrics.reset()
        self.log_dict(selective_classification_metrics.compute())
        selective_classification_metrics.reset()

    @abstractmethod
    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        """
        Run the forward pass and compute the loss for a batch.

        Subclasses override this method to change how the loss is computed
        (e.g. adversarial training) without affecting metric computation.

        Returns:
            A tuple of the loss and the model output, the latter being used
            for metric computation.
        """
        ...

    def update_metrics(
        self,
        output: ClassificationModelOutput,
        targets: Tensor,
        classification_metrics: MetricCollection,
        fd_metrics: MetricCollection,
        selective_classification_metrics: MetricCollection,
    ):
        """
        Update the classification, failure detection and selective classification
        metrics from a model output.

        Subclasses override this method to change how metrics are computed
        without affecting loss computation.
        """
        classification_metrics.update(output.preds.long(), targets.long())

        errors = output.preds.long() != targets.long()
        fd_metrics.update(output.uncertainty, errors)

        selective_classification_metrics.update(
            output.probs, targets.long(), output.uncertainty
        )

    def training_step(self, batch: list[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the model.
        """
        loss, output = self.compute_loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.update_metrics(
            output,
            batch[1],
            self.train_classification_metrics,
            self.train_fd_metrics,
            self.train_selective_classification_metrics,
        )
        return loss

    def validation_step(self, batch: list[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Validation step for the model.
        """
        loss, output = self.compute_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.update_metrics(
            output,
            batch[1],
            self.val_classification_metrics,
            self.val_fd_metrics,
            self.val_selective_classification_metrics,
        )
        return loss

    def configure_optimizers(self):
        optimization_config = {
            "optimizer": self.optimizer,
        }
        if self.scheduler is not None:
            optimization_config["lr_scheduler"] = {
                "scheduler": self.scheduler,
                "interval": "epoch",
            }
        return optimization_config

    @property
    def classification_metrics(self) -> MetricCollection:
        """
        Property that define the classification metrics for evaluation.

        Returns:
            MetricCollection: A collection of metrics to be used for evaluation.
        """
        return get_multiclass_classification_metrics(num_classes=self._num_classes)

    @property
    def failure_detection_metrics(self) -> MetricCollection:
        """
        Property that define the failure detection metrics for evaluation.

        Returns:
            MetricCollection: A collection of metrics to be used for evaluation.
        """
        return get_failure_detection_metrics()

    @property
    def selective_classification_metrics(self) -> MetricCollection:
        """
        Property that define the selective classification metrics for evaluation.

        Returns:
            MetricCollection: A collection of metrics to be used for selective
            classification.
        """
        return get_selective_classification_metrics()
