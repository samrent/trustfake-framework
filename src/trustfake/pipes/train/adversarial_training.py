"""Adversarial-training baselines: PGD-AT (Madry et al. 2018) and TRADES
(Zhang et al. 2019). These are the robustness arms the harness benchmarks the
evidential method against.

Both generate an inner adversary each step and train on it. For a fair
comparison the arms should be matched on optimiser steps, not wall-clock
(PGD-k costs k+1 forwards per step): set the same max_epochs and schedule for
every arm. An optional epsilon warm-up ramps eps over the first
`eps_warmup_epochs` epochs -- deepfake evidence is small-amplitude
high-frequency residue, and a full ImageNet budget (8/255) can erase it and
collapse training, so a forensic budget with warm-up is the stable regime.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, kl_div, log_softmax

from trustfake.attacks._common import model_logits, project_linf
from trustfake.pipes.train.abc import TrainingModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = ["PGDAdversarialTrainingModule", "TRADESTrainingModule"]


class _AdversarialTrainingBase(TrainingModule):
    """Shared eps warm-up and inner-PGD scaffold."""

    def __init__(
        self,
        *args,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        eps_warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
        self.eps_warmup_epochs = eps_warmup_epochs

    @property
    def current_eps(self) -> float:
        if self.eps_warmup_epochs <= 0:
            return self.eps
        return self.eps * min(1.0, (self.current_epoch + 1) / self.eps_warmup_epochs)

    def _inner_pgd(self, inputs: Tensor, objective, eps: float) -> Tensor:
        """K-step PGD ascending `objective(x_adv) -> scalar` within the eps-ball,
        from a random start. Returns a detached adversarial batch."""
        alpha = (
            self.alpha
            if self.eps_warmup_epochs <= 0
            else 2.5 * eps / max(self.steps, 1)
        )
        x_adv = inputs.detach() + torch.empty_like(inputs).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                loss = objective(x_adv)
                grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = project_linf(x_adv, inputs, eps).clamp(0.0, 1.0)
        return x_adv.detach()

    def _output(self, x: Tensor) -> tuple[Tensor, ClassificationModelOutput]:
        logits, probs, preds, uncertainty = self.model.forward(x)
        return logits, ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )


class PGDAdversarialTrainingModule(_AdversarialTrainingBase):
    """PGD-AT (Madry et al. 2018): train on the worst-case cross-entropy point
    inside the eps-ball. Metrics are reported on the clean forward."""

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        eps = self.current_eps

        def objective(x_adv: Tensor) -> Tensor:
            return cross_entropy(model_logits(self.model, x_adv), y)

        x_adv = self._inner_pgd(x, objective, eps)
        adv_logits = model_logits(self.model, x_adv)
        loss = self.model.loss_fn(adv_logits, y)

        _, clean_output = self._output(x)
        return loss, clean_output


class TRADESTrainingModule(_AdversarialTrainingBase):
    """TRADES (Zhang et al. 2019): natural CE plus a robustness term
    beta * KL(f(x) || f(x_adv)), with x_adv maximising that KL. beta trades
    natural accuracy against robustness."""

    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        eps = self.current_eps

        clean_logits, clean_output = self._output(x)
        clean_logp = log_softmax(clean_logits.detach(), dim=1)

        def objective(x_adv: Tensor) -> Tensor:
            adv_logp = log_softmax(model_logits(self.model, x_adv), dim=1)
            return kl_div(adv_logp, clean_logp, log_target=True, reduction="batchmean")

        x_adv = self._inner_pgd(x, objective, eps)

        natural = cross_entropy(clean_logits, y)
        adv_logp = log_softmax(model_logits(self.model, x_adv), dim=1)
        robust = kl_div(
            adv_logp,
            log_softmax(clean_logits, dim=1),
            log_target=True,
            reduction="batchmean",
        )
        loss = natural + self.beta * robust
        return loss, clean_output
