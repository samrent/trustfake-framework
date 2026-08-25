"""
This module defines a standard training module for classification tasks.
The standard training module is designed to perform "classical" training
without attacks.
"""

from torch import Tensor

from trustfake.pipes.train import TrainingModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = ["StandardTrainingModule"]


class StandardTrainingModule(TrainingModule):
    """Simple training module for classification.

    The module is designed for "classical" training.

    .. note::
        The forward pass returns logits, probabilities, label predictions and
        uncertainty.
    """

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        """
        Run the forward pass and compute the loss for a batch.
        """
        logits, probs, preds, uncertainty = self.model.forward(batch[0])
        loss = self.model.loss_fn(logits, batch[1])
        output = ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )
        return loss, output
