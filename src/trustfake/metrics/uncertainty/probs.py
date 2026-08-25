import torch
from torch import Tensor
from torchmetrics import Metric


class MultiClassMaxProbability(Metric):
    """
    Simplest uncertainty score which 1 - Maximum probability across classes.
    """

    def __init__(self, **metric_kwargs):
        super().__init__(**metric_kwargs)
        self._enable_grad = True
        self.max_probability: list[Tensor]
        self.add_state("max_probability", default=[], dist_reduce_fx="cat")

    def update(self, probabilities: Tensor) -> None:
        """
        Args:
            probabilities: A tensor of shape (N, C, *) containing predicted
                probabilities of a multi-class classification task.
        """
        self.max_probability.append(1 - torch.max(probabilities, dim=1).values)

    def compute(self) -> Tensor:
        if self.max_probability:
            return torch.cat(self.max_probability, dim=0)
