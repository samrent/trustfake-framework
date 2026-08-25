import torch
from torch import Tensor
from torchmetrics import Metric


class MCDropoutPredictiveEntropy(Metric):
    """
    Monte-Carlo Dropout uncertainty score: the predictive entropy of the
    class probabilities averaged over the stochastic forward passes fed to
    `update` (dropout kept active at inference time), as defined in
    Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016).
    """

    def __init__(self, eps: float = 1e-12, **metric_kwargs):
        super().__init__(**metric_kwargs)
        self.eps = eps
        self._enable_grad = True
        self.probabilities: list[Tensor]
        self.add_state("probabilities", default=[], dist_reduce_fx="cat")

    def update(self, probabilities: Tensor) -> None:
        """
        Args:
            probabilities: A tensor of shape (N, C, *) containing predicted
                probabilities for one stochastic forward pass of a
                multi-class classification task. Called once per pass.
        """
        self.probabilities.append(probabilities.unsqueeze(0))

    def compute(self) -> Tensor:
        if self.probabilities:
            mean_probabilities = torch.cat(self.probabilities, dim=0).mean(dim=0)
            return -torch.sum(
                mean_probabilities * torch.log(mean_probabilities + self.eps), dim=1
            )
