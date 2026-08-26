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
from trustfake.pipes.train._common import RobustValidationMixin, eval_mode
from trustfake.pipes.train.abc import TrainingModule
from trustfake.pipes.train.awp import AWPMixin
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = [
    "PGDAdversarialTrainingModule",
    "TRADESTrainingModule",
    "HybridAdversarialTrainingModule",
    "MARTTrainingModule",
]


class _AdversarialTrainingBase(AWPMixin, RobustValidationMixin, TrainingModule):
    """Shared eps warm-up, inner-PGD scaffold and robust model selection."""

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
        # None means "use the standard rule", evaluated against whatever eps
        # and step count are in force at the time. Storing a number here
        # instead froze it against the CONSTRUCTOR's step count, which is the
        # wrong one during robust validation (it runs fewer steps) and made
        # the attack silently depend on whether eps warm-up happened to be on.
        self._alpha_override = alpha
        self.eps_warmup_epochs = eps_warmup_epochs

    @property
    def current_eps(self) -> float:
        if self.eps_warmup_epochs <= 0:
            return self.eps
        return self.eps * min(1.0, (self.current_epoch + 1) / self.eps_warmup_epochs)

    @property
    def alpha(self) -> float:
        """PGD step size for the CURRENT eps and step count.

        Derived rather than stored. The standard rule is 2.5*eps/steps
        (Madry; RobustBench), and both inputs move: eps ramps under warm-up,
        and `steps` is temporarily lowered during robust validation. A frozen
        value silently means a different attack in each of those cases -- with
        `robust_val_steps=3` against a training `steps=10` it cannot even
        reach the ball boundary (3 x 0.25 eps = 0.75 eps), so the reported
        robust accuracy is of a weaker attack than the one being trained on.
        """
        if self._alpha_override is not None:
            return self._alpha_override
        return 2.5 * self.current_eps / max(self.steps, 1)

    def _inner_pgd(self, inputs: Tensor, objective, eps: float) -> Tensor:
        """K-step PGD ascending `objective(x_adv) -> scalar` within the eps-ball,
        from a random start. Returns a detached adversarial batch.

        Runs with the model in eval mode. Every attack in `trustfake.attacks`
        does; in train mode each of the k iterates would be folded into every
        BatchNorm running estimate, and each forward would normalise by a
        perturbed batch's statistics rather than the deployed ones.
        """
        alpha = (
            self._alpha_override
            if self._alpha_override is not None
            else 2.5 * eps / max(self.steps, 1)
        )
        x_adv = inputs.detach() + torch.empty_like(inputs).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
        with eval_mode(self.model):
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
        """A TRAINING forward: gradients flow and BatchNorm updates.

        Use this only when the returned logits feed the loss. For a forward
        whose output is reported rather than trained on, use
        `_metrics_output`.
        """
        logits, probs, preds, uncertainty = self.model.forward(x)
        return logits, ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )

    def _metrics_output(self, x: Tensor) -> ClassificationModelOutput:
        """A REPORTING forward: eval mode, no gradients, no BatchNorm update.

        Arms that train only on the adversarial batch still report their
        metrics on the clean one. Running that report in train mode makes it
        a second BatchNorm update per step on data the arm never trains on,
        which quietly shifts the running statistics of the deployed model.
        """
        with torch.no_grad(), eval_mode(self.model):
            logits, probs, preds, uncertainty = self.model.forward(x)
        return ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )


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
        loss = self.model.loss_fn(self.model.loss_input(adv_logits), y)

        return loss, self._metrics_output(x)


class TRADESTrainingModule(_AdversarialTrainingBase):
    """TRADES (Zhang et al. 2019): natural CE plus a robustness term
    beta * KL(f(x) || f(x_adv)), with x_adv maximising that KL. beta trades
    natural accuracy against robustness.

    The inner maximisation is on the KL, NOT on cross-entropy -- see the
    module docstring for why that swap is the classic misimplementation."""

    def __init__(self, *args, beta: float = 6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def awp_objective(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> Tensor:
        """The TRADES loss, not cross-entropy.

        The published TRADES-AWP (`utils_awp.py::TradesAWP.calc_awp`) ascends
        the full TRADES objective; ascending CE instead measures a different
        modifier from the one the literature calls "AWP on top of TRADES".
        """
        clean_logits = model_logits(self.model, inputs)
        adv_logp = log_softmax(model_logits(self.model, x_adv), dim=1)
        return cross_entropy(clean_logits, targets.long()) + self.beta * kl_div(
            adv_logp,
            log_softmax(clean_logits, dim=1),
            log_target=True,
            reduction="batchmean",
        )

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        eps = self.current_eps

        # Reference for the inner maximisation, at the current weights. Eval
        # mode: this forward exists to steer the attack, not to train on.
        with torch.no_grad(), eval_mode(self.model):
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

    def awp_objective(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> Tensor:
        """The AT+KL loss, so the weight adversary attacks what is trained."""
        clean_logits = model_logits(self.model, inputs)
        adv_logits = model_logits(self.model, x_adv)
        return cross_entropy(adv_logits, targets.long()) + self.beta * kl_div(
            log_softmax(adv_logits, dim=1),
            log_softmax(clean_logits.detach(), dim=1),
            log_target=True,
            reduction="batchmean",
        )

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

    def _mart_loss(
        self, clean_logits: Tensor, adv_logits: Tensor, targets: Tensor
    ) -> Tensor:
        """MART's objective, shared by training and the weight adversary."""
        adv_probs = softmax(adv_logits, dim=1)
        # Runner-up class: the second-largest probability, or the largest
        # when the top one is already the true class.
        top2 = adv_probs.argsort(dim=1)[:, -2:]
        runner_up = torch.where(top2[:, -1] == targets, top2[:, -2], top2[:, -1])
        boosted = cross_entropy(adv_logits, targets) + nll_loss(
            torch.log(1.0001 - adv_probs + 1e-12), runner_up
        )

        clean_probs = softmax(clean_logits, dim=1)
        true_prob = clean_probs.gather(1, targets[:, None]).squeeze(1)
        per_sample_kl = kl_div(
            torch.log(adv_probs + 1e-12), clean_probs, reduction="none"
        ).sum(dim=1)
        # Weight by how unsure the model already was: 1 - p_y(clean).
        weighted_kl = (per_sample_kl * (1.0000001 - true_prob)).mean()
        return boosted + self.beta * weighted_kl

    def awp_objective(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> Tensor:
        """The MART loss, so the weight adversary attacks what is trained."""
        return self._mart_loss(
            model_logits(self.model, inputs),
            model_logits(self.model, x_adv),
            targets.long(),
        )

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        x_adv = self._pgd_ce(x, y, self.current_eps)
        self.apply_awp(x, x_adv, y)

        adv_logits = model_logits(self.model, x_adv)
        clean_logits, clean_output = self._output(x)

        loss = self._mart_loss(clean_logits, adv_logits, y)
        return loss, clean_output
