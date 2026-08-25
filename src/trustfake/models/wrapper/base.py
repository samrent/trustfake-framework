import torch
from torch import Tensor

from trustfake.models.wrapper.abc import TrustFakeWrapper


class BaseWrapper(TrustFakeWrapper):
    """
    Base class for all Deep (PyTorch) wrapper in the TrustFake framework.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            logits (Tensor): The raw output of the model
            probs (Tensor): The probabilities of the classes
            preds (Tensor): The predicted class labels
            uncertainty (Tensor): The uncertainty of the predictions
        """
        self.uncertainty_score = self.uncertainty_score.to(x.device)
        x = self.normalization_layer(x)
        logits = self.model(x)
        probs = torch.softmax(logits / self.temperature, dim=1)
        preds = torch.argmax(probs, dim=1)

        uncertainty = self.uncertainty_score(probs)
        self.uncertainty_score.reset()

        return logits, probs, preds, uncertainty

    def outputs_from_logits(
        self, logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Derive (logits, probs, preds, uncertainty) from a single forward's
        logits -- exactly the post-model half of `forward`, which is why
        this wrapper can offer it and a stochastic one cannot.
        """
        self.uncertainty_score = self.uncertainty_score.to(logits.device)
        probs = torch.softmax(logits / self.temperature, dim=1)
        preds = torch.argmax(probs, dim=1)
        uncertainty = self.uncertainty_score(probs)
        self.uncertainty_score.reset()
        return logits, probs, preds, uncertainty
