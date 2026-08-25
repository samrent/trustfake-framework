from abc import ABC, abstractmethod

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Metric


class TrustFakeWrapper(ABC, pl.LightningModule):
    """
    Abstract base class for all modules.

    This module is used as a forward pass wrapper
        for models in all training and evaluation pipelines

    Data preprocessing is handled by the normalization layer,
        called in the forward pass.
    ... note:: Data preprocessing is done just before model forward pass
        to avoid issues with adversarial robustness L_p balls.

    Args:
        normalization_layer (nn.Module): The normalization layer to
            be applied to the input data.
        model (nn.Module): The model to be trained or evaluated.
        loss_fn (nn.Module | None): The loss function to be used during training.
        uncertainty_score (Metric): The metric to be used
            for uncertainty quantification.
        temperature (float): Post-hoc temperature applied to logits before
            softmax when computing probs/preds/uncertainty. 1.0 is a no-op.
    """

    def __init__(
        self,
        normalization_layer: nn.Module,
        model: nn.Module,
        loss_fn: nn.Module | None,
        uncertainty_score: Metric,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.normalization_layer = normalization_layer
        self.model = model
        self.loss_fn = loss_fn
        self.uncertainty_score = uncertainty_score
        # Temperature scaling of the probabilities (a post-hoc calibration
        # fitted on the calib split; see trustfake.metrics.calibration). The
        # returned logits stay raw -- attacks and the loss act on the model,
        # not on its calibrated probabilities -- while probs, preds and
        # uncertainty are computed from logits / temperature. T > 0 is
        # monotone, so preds and accuracy are unchanged by it.
        self.temperature = temperature

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            logits (torch.Tensor): The raw output of the model
            probs (torch.Tensor): The probabilities of the classes
            preds (torch.Tensor): The predicted class labels
            uncertainty (torch.Tensor): The uncertainty of the predictions
        """
        ...

    def outputs_from_logits(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Derive the full (logits, probs, preds, uncertainty) output from the
        logits of a single forward pass, or return None when this wrapper's
        uncertainty cannot be derived that way (e.g. MC dropout needs
        multiple stochastic passes).

        Used by the evaluation pipe to score an attack's accept-check
        forward directly instead of re-running the model on the perturbed
        batch (see `trustfake.attacks.abc.AttackResult`).
        """
        return None

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.uncertainty_score = self.uncertainty_score.to(self.device)

    def on_train_batch_end(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.uncertainty_score.reset()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.uncertainty_score = self.uncertainty_score.to(self.device)
