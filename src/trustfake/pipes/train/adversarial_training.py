"""Adversarial-training arms: the robustness baselines the evidential method
is benchmarked against, plus the two hybrids the sweep compares them to.

  * PGD-AT (Madry et al. 2018)  -- CE on the worst-case point
  * TRADES (Zhang et al. 2019)  -- CE on clean + beta * KL(clean || adv)
  * AT+KL  (hybrid)             -- CE on ADVERSARIAL + beta * consistency KL
  * MART   (Wang et al. 2020)   -- margin-aware AT, misclassified-weighted KL

All four generate an inner adversary each step and train on it, and all four
compose with `AWPMixin` (weight-space inner maximisation, off by default).

For a fair comparison the arms must be matched on optimiser steps, not
wall-clock: PGD-k costs k+1 forwards per step, so an arm given equal minutes
is given fewer updates, and "our method wins" then means "our method got more
training". Set the same max_epochs and schedule for every arm.

An optional epsilon warm-up ramps eps over the first `eps_warmup_epochs`
epochs -- deepfake evidence is small-amplitude high-frequency residue, and a
full ImageNet budget (8/255) can erase it and collapse training onto a
constant output, which is silent if only the final accuracy is read.

AT+KL is not TRADES renamed. TRADES puts cross-entropy on the CLEAN forward
and uses the KL as the only robustness pressure; AT+KL puts cross-entropy on
the ADVERSARIAL forward and adds the KL on top -- a genuine adversarial
training + regulariser stack. Which forward carries the label term is the
whole distinction, and swapping the two by accident is the classic TRADES
misimplementation: it still trains, and the result still looks plausible.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, kl_div, log_softmax, nll_loss, softmax

from trustfake.attacks._common import model_logits, project_linf
from trustfake.pipes.train.abc import TrainingModule
from trustfake.pipes.train.awp import AWPMixin
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = [
    "PGDAdversarialTrainingModule",
    "TRADESTrainingModule",
    "HybridAdversarialTrainingModule",
    "MARTTrainingModule",
]


class _AdversarialTrainingBase(AWPMixin, TrainingModule):
    """Shared eps warm-up, inner-PGD scaffold and robust model selection."""

    def __init__(
        self,
        *args,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        eps_warmup_epochs: int = 0,
        robust_val_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
        self.eps_warmup_epochs = eps_warmup_epochs
        self.robust_val_steps = robust_val_steps

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

    def _pgd_ce(self, x: Tensor, y: Tensor, eps: float) -> Tensor:
        """The Madry inner maximisation: worst-case cross-entropy."""

        def objective(x_adv: Tensor) -> Tensor:
            return cross_entropy(model_logits(self.model, x_adv), y)

        return self._inner_pgd(x, objective, eps)

    def _output(self, x: Tensor) -> tuple[Tensor, ClassificationModelOutput]:
        logits, probs, preds, uncertainty = self.model.forward(x)
        return logits, ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )

    def validation_step(self, batch: list[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Standard validation, plus robust accuracy when asked for.

        Selecting a defence arm on clean macro-F1 selects it on the thing it
        deliberately trades away: the checkpoint kept is then the least
        robust epoch that happened to fit the clean split best. Set
        `robust_val_steps > 0` and point the checkpoint callback at
        `val_robust_accuracy` to select each arm on what it actually
        optimises. It costs one PGD run per validation batch, which is why it
        is opt-in rather than always on.
        """
        loss = super().validation_step(batch, batch_idx)
        if self.robust_val_steps > 0:
            x, y = batch[0], batch[1].long()
            steps, self.steps = self.steps, self.robust_val_steps
            try:
                x_adv = self._pgd_ce(x, y, self.current_eps)
            finally:
                self.steps = steps
            with torch.no_grad():
                preds = self.model(x_adv)[2]
            self.log(
                "val_robust_accuracy",
                (preds.long() == y).float().mean(),
                on_epoch=True,
                prog_bar=False,
            )
        return loss


class PGDAdversarialTrainingModule(_AdversarialTrainingBase):
    """PGD-AT (Madry et al. 2018): train on the worst-case cross-entropy point
    inside the eps-ball. Metrics are reported on the clean forward."""

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        x_adv = self._pgd_ce(x, y, self.current_eps)
        self.apply_awp(x, x_adv, y)

        adv_logits = model_logits(self.model, x_adv)
        loss = self.model.loss_fn(adv_logits, y)

        _, clean_output = self._output(x)
        return loss, clean_output


class TRADESTrainingModule(_AdversarialTrainingBase):
    """TRADES (Zhang et al. 2019): natural CE plus a robustness term
    beta * KL(f(x) || f(x_adv)), with x_adv maximising that KL. beta trades
    natural accuracy against robustness.

    The inner maximisation is on the KL, NOT on cross-entropy -- see the
    module docstring for why that swap is the classic misimplementation."""

    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        eps = self.current_eps

        # Reference for the inner maximisation, at the current weights.
        with torch.no_grad():
            reference_logp = log_softmax(model_logits(self.model, x), dim=1)

        def objective(x_adv: Tensor) -> Tensor:
            adv_logp = log_softmax(model_logits(self.model, x_adv), dim=1)
            return kl_div(
                adv_logp, reference_logp, log_target=True, reduction="batchmean"
            )

        x_adv = self._inner_pgd(x, objective, eps)
        self.apply_awp(x, x_adv, y)

        # Both sides of the loss are evaluated after the weight perturbation,
        # so the two terms describe the same model. Computing the clean side
        # before `apply_awp` would mix a term at w with a term at w + v.
        clean_logits, clean_output = self._output(x)
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


class HybridAdversarialTrainingModule(_AdversarialTrainingBase):
    """AT+KL hybrid: adversarial cross-entropy plus a consistency KL between
    the adversarial and the (detached) clean prediction.

    The question it answers is whether *layering* defences beats a well-tuned
    single one -- which in the literature it frequently does not, so this arm
    exists to be measured rather than assumed. The clean side is detached so
    the KL pulls the adversarial prediction toward the clean one and not the
    other way round; without the detach the model can satisfy the term by
    degrading its clean prediction, which is the cheaper solution and looks
    like success on the loss curve.
    """

    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        x_adv = self._pgd_ce(x, y, self.current_eps)
        self.apply_awp(x, x_adv, y)

        clean_logits, clean_output = self._output(x)
        adv_logits = model_logits(self.model, x_adv)

        adversarial = cross_entropy(adv_logits, y)
        consistency = kl_div(
            log_softmax(adv_logits, dim=1),
            log_softmax(clean_logits.detach(), dim=1),
            log_target=True,
            reduction="batchmean",
        )
        loss = adversarial + self.beta * consistency
        return loss, clean_output


class MARTTrainingModule(_AdversarialTrainingBase):
    """MART (Wang et al., ICLR 2020): Misclassification-Aware adveRsarial
    Training.

    Two changes to PGD-AT, both about the examples the model gets wrong.
    The adversarial term is margin-aware: on top of cross-entropy it adds
    `log(1 - p_k)` for the runner-up class k, which pushes the boundary away
    from the nearest wrong class rather than only raising the true-class
    score. And the KL consistency term is weighted by `1 - p_y(x_clean)`, so
    examples the model is already unsure about dominate the robustness term --
    the ones near the boundary are the ones an attacker will use.

    That weighting is why MART belongs in a *selective* harness: it is the
    only classical AT arm whose objective is explicitly a function of the
    model's own confidence, which makes it the fairest classical comparison
    for a method that targets the confidence axis.
    """

    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        x_adv = self._pgd_ce(x, y, self.current_eps)
        self.apply_awp(x, x_adv, y)

        adv_logits = model_logits(self.model, x_adv)
        clean_logits, clean_output = self._output(x)

        adv_probs = softmax(adv_logits, dim=1)
        # Runner-up class: the second-largest probability, or the largest
        # when the top one is already the true class.
        top2 = adv_probs.argsort(dim=1)[:, -2:]
        runner_up = torch.where(top2[:, -1] == y, top2[:, -2], top2[:, -1])
        boosted = cross_entropy(adv_logits, y) + nll_loss(
            torch.log(1.0001 - adv_probs + 1e-12), runner_up
        )

        clean_probs = softmax(clean_logits, dim=1)
        true_prob = clean_probs.gather(1, y[:, None]).squeeze(1)
        per_sample_kl = kl_div(
            torch.log(adv_probs + 1e-12), clean_probs, reduction="none"
        ).sum(dim=1)
        # Weight by how unsure the model already was: 1 - p_y(clean).
        weighted_kl = (per_sample_kl * (1.0000001 - true_prob)).mean()

        loss = boosted + self.beta * weighted_kl
        return loss, clean_output
