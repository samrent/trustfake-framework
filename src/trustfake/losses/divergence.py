r"""Discrepancy D on log-Dirichlet parameters eta = log(alpha), used by EV-AT
both to generate the evidence-targeted adversary and as the robust
evidence-alignment loss L_REA (arXiv:2607.03075 Sec. 3.3).

Aligning in eta-space is the paper's key choice (softmax(eta) = pi_bar, the
posterior mean, Eq. 5). Three instantiations, per the paper's ablation:

  * "ikl"  Improved KL divergence (Cui et al. 2023, Eq. 10) -- the paper's
           default and best. On logits o (= eta here) with s = softmax(o):
             L_IKL = (a/4) * || sg(w_bar_y) * (Dm - Dn) ||^2
                     - b * sg(s_m) . log s_n,
           Dm[j,k] = o_m^j - o_m^k (pairwise logit differences), and
           w_bar_y[j,k] = s_bar_y^j * s_bar_y^k is a *class-wise global*
           weight built from running per-class means of s (updated during
           training via `update_global_stats`). Before any stats accumulate
           it falls back to the sample-wise weight sg(s_m) x sg(s_m) (the DKL
           weight). "Breaking asymmetry": no stop-gradient on Dm.
  * "kl"   KL(s_m || s_n) on the posterior means. Valid, slightly weaker.
  * "l2"   squared distance in eta-space. Valid, slightly weaker.

`m` is the clean side, `n` the adversarial side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["LogDirichletDivergence"]


class LogDirichletDivergence(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: str = "ikl",
        ikl_alpha: float = 1.0,
        ikl_beta: float = 1.0,
        ema: float = 0.9,
        eps: float = 1e-12,
    ):
        super().__init__()
        if mode not in ("ikl", "kl", "l2"):
            raise ValueError(f"mode must be ikl|kl|l2, got {mode!r}")
        self.num_classes = num_classes
        self.mode = mode
        self.ikl_alpha = ikl_alpha
        self.ikl_beta = ikl_beta
        self.ema = ema
        self.eps = eps
        # Per-class running mean of the predictive means s (row c = s_bar_c).
        self.register_buffer(
            "class_means", torch.full((num_classes, num_classes), 1.0 / num_classes)
        )
        self.register_buffer("stats_ready", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update_global_stats(self, probs: Tensor, targets: Tensor) -> None:
        """EMA-update the per-class mean predictive distribution (for w_bar_y)."""
        targets = targets.long()
        for c in targets.unique().tolist():
            mask = targets == c
            batch_mean = probs[mask].mean(dim=0)
            self.class_means[c].mul_(self.ema).add_(batch_mean, alpha=1.0 - self.ema)
        self.stats_ready.fill_(True)

    def _pairwise_diff(self, o: Tensor) -> Tensor:
        """o (B, C) -> Dm (B, C, C) with Dm[:, j, k] = o_j - o_k."""
        return o.unsqueeze(2) - o.unsqueeze(1)

    def forward(
        self, eta_m: Tensor, eta_n: Tensor, targets: Tensor | None = None
    ) -> Tensor:
        s_m = torch.softmax(eta_m, dim=1)
        s_n = torch.softmax(eta_n, dim=1)

        if self.mode == "l2":
            return ((eta_m - eta_n) ** 2).sum(dim=1).mean()

        if self.mode == "kl":
            return (
                (s_m * (torch.log(s_m + self.eps) - torch.log(s_n + self.eps)))
                .sum(dim=1)
                .mean()
            )

        # ikl
        dm = self._pairwise_diff(eta_m)
        dn = self._pairwise_diff(eta_n)
        if targets is not None and bool(self.stats_ready):
            s_bar = self.class_means[targets.long()]  # (B, C)
            weight = s_bar.unsqueeze(2) * s_bar.unsqueeze(1)  # (B, C, C)
        else:
            # DKL fallback: sample-wise weight from the (detached) clean mean.
            sg = s_m.detach()
            weight = sg.unsqueeze(2) * sg.unsqueeze(1)
        weight = weight.detach()  # stop-gradient on the weight

        wmse = (self.ikl_alpha / 4.0) * (weight * (dm - dn) ** 2).sum(dim=(1, 2))
        ce = -self.ikl_beta * (s_m.detach() * torch.log(s_n + self.eps)).sum(dim=1)
        return (wmse + ce).mean()
