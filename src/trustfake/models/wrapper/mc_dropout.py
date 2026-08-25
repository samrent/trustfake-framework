import torch
import torch.nn as nn
from torch import Tensor

from trustfake.models.wrapper.abc import TrustFakeWrapper


def _enable_dropout(model: nn.Module) -> None:
    """
    Switch only the Dropout layers of `model` to train mode, keeping every
    other layer (e.g. BatchNorm) in eval mode, so dropout stays stochastic
    at test time.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()


class MCDropoutWrapper(TrustFakeWrapper):
    """
    Wrapper implementing Monte-Carlo Dropout inference: the model is run
    `num_samples` times with dropout kept active, and the returned
    logits/probs/preds are the average over those stochastic passes.
    """

    def __init__(self, *args, num_samples: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            logits (Tensor): The raw model output averaged over the MC samples
            probs (Tensor): The class probabilities averaged over the MC samples
            preds (Tensor): The predicted class labels
            uncertainty (Tensor): The uncertainty of the predictions
        """
        self.uncertainty_score = self.uncertainty_score.to(x.device)
        x = self.normalization_layer(x)

        _enable_dropout(self.model)

        logits_samples = []
        probs_samples = []
        for _ in range(self.num_samples):
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            logits_samples.append(logits)
            probs_samples.append(probs)
            self.uncertainty_score.update(probs)

        logits = torch.stack(logits_samples, dim=0).mean(dim=0)
        probs = torch.stack(probs_samples, dim=0).mean(dim=0)
        preds = torch.argmax(probs, dim=1)

        uncertainty = self.uncertainty_score.compute()
        self.uncertainty_score.reset()

        return logits, probs, preds, uncertainty
