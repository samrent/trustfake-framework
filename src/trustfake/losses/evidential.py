r"""Evidential clean loss L_EV (arXiv:2607.03075 Eq. 2-3; Sensoy et al. 2018).

Given Dirichlet concentrations alpha = evidence + 1 and a one-hot label y:

  * Type-II marginal likelihood (Eq. 2):
        L = -log(alpha_y / S) = log S - log(alpha_y),  S = sum_c alpha_c,
    the negative log of the Dirichlet-categorical marginal E_pi[p(y|pi)].
  * KL regulariser toward the uniform Dirichlet Dir(1) on the *misleading*
    evidence (Eq. 3): the ground-truth evidence is removed first,
        alpha_tilde = y + (1 - y) * alpha   (alpha_tilde_c = 1 at c = y),
    so only wrong-class evidence is penalised toward vacuity. lambda ramps
    from 0 over the first `anneal_epochs` epochs (Sensoy's min(1, t/10)),
    which the training module updates via `set_epoch`.

        L_EV = L + lambda * KL[Dir(alpha_tilde) || Dir(1)].
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["EvidentialLoss", "dirichlet_uniform_kl"]


def dirichlet_uniform_kl(alpha: Tensor) -> Tensor:
    r"""Closed-form KL[Dir(alpha) || Dir(1)], per sample, shape (B,).

    KL = logGamma(S) - sum_c logGamma(alpha_c) - logGamma(C)
         + sum_c (alpha_c - 1) (psi(alpha_c) - psi(S)),  S = sum_c alpha_c.
    """
    c = alpha.shape[1]
    strength = alpha.sum(dim=1)
    term = (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=1)
        - torch.lgamma(torch.tensor(float(c), device=alpha.device))
    )
    digamma_diff = torch.digamma(alpha) - torch.digamma(strength).unsqueeze(1)
    term = term + ((alpha - 1.0) * digamma_diff).sum(dim=1)
    return term


class EvidentialLoss(nn.Module):
    r"""L_EV = NLL(Type-II marginal) + lambda * KL[Dir(alpha_tilde) || Dir(1)].

    Args:
        num_classes: number of classes C.
        lambda_max: maximum KL weight after annealing.
        anneal_epochs: epochs over which lambda ramps 0 -> lambda_max.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_max: float = 1.0,
        anneal_epochs: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_max = lambda_max
        self.anneal_epochs = max(anneal_epochs, 1)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Called by the training module so the KL weight can anneal."""
        self._epoch = epoch

    @property
    def kl_weight(self) -> float:
        return self.lambda_max * min(1.0, self._epoch / self.anneal_epochs)

    def forward(self, alpha: Tensor, targets: Tensor) -> Tensor:
        targets = targets.long()
        strength = alpha.sum(dim=1)
        alpha_y = alpha.gather(1, targets.view(-1, 1)).squeeze(1)
        nll = torch.log(strength) - torch.log(alpha_y)

        onehot = torch.zeros_like(alpha)
        onehot.scatter_(1, targets.view(-1, 1), 1.0)
        alpha_tilde = onehot + (1.0 - onehot) * alpha  # remove GT evidence
        kl = dirichlet_uniform_kl(alpha_tilde)

        return (nll + self.kl_weight * kl).mean()
