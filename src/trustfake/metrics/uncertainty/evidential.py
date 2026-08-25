import torch
from torch import Tensor
from torchmetrics import Metric

__all__ = ["EvidentialPredictiveEntropy"]


class EvidentialPredictiveEntropy(Metric):
    """Predictive entropy of the evidential posterior mean, the selective
    score of EV-AT (arXiv:2607.03075, Eq. 1):

        u(x) = H[Cat(pi_bar)] = - sum_c pi_bar_c log pi_bar_c,

    where pi_bar = alpha / S is the Dirichlet posterior mean. Higher means
    more uncertain, so it sorts the same direction as the other uncertainty
    scores. The wrapper passes pi_bar in as `probs`, so this is a plain
    Shannon entropy of the given probabilities.
    """

    def __init__(self, eps: float = 1e-12, **metric_kwargs):
        super().__init__(**metric_kwargs)
        self.eps = eps
        self._enable_grad = True
        self.entropy: list[Tensor]
        self.add_state("entropy", default=[], dist_reduce_fx="cat")

    def update(self, probabilities: Tensor) -> None:
        """
        Args:
            probabilities: (N, C) posterior mean probabilities pi_bar.
        """
        p = probabilities.clamp_min(self.eps)
        self.entropy.append(-torch.sum(p * torch.log(p), dim=1))

    def compute(self) -> Tensor:
        if self.entropy:
            return torch.cat(self.entropy, dim=0)
        return torch.empty(0)
