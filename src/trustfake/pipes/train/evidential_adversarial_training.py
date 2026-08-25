"""Evidential Adversarial Training (EV-AT) module (arXiv:2607.03075, Eq. 4).

Min-max per batch:
  1. generate x_adv by maximising the log-Dirichlet drift D(eta, eta_adv)
     (the evidence-targeted adversary);
  2. update the model to minimise L_EV(alpha, y) + beta * L_REA(eta, eta_adv),
     the clean evidential loss plus the robust evidence-alignment loss.

The adversary and L_REA share one LogDirichletDivergence instance, so the IKL
class-wise global statistics are updated once and used by both. The evidential
loss (model.loss_fn) has its KL weight annealed per epoch.
"""

from __future__ import annotations

from torch import Tensor

from trustfake.attacks import EvidenceTargetedPGD
from trustfake.losses import LogDirichletDivergence
from trustfake.pipes.train.abc import TrainingModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = ["EvidentialAdversarialTrainingModule"]


class EvidentialAdversarialTrainingModule(TrainingModule):
    """EV-AT training module. Requires an evidential wrapper and an
    EvidentialLoss as ``model.loss_fn``.

    Args:
        beta: weight of the robust evidence-alignment term L_REA.
        divergence_mode: discrepancy for the adversary and L_REA (ikl|kl|l2).
        adv_eps: L_inf budget of the evidence-targeted adversary.
        adv_steps: PGD steps of the adversary.
    """

    def __init__(
        self,
        *args,
        beta: float = 1.0,
        divergence_mode: str = "ikl",
        adv_eps: float = 8 / 255,
        adv_steps: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not hasattr(self.model, "dirichlet"):
            raise TypeError(
                f"{type(self.model).__name__} is not an evidential wrapper; "
                "EV-AT needs a `dirichlet` method (use wrapper=evidential)."
            )
        if not hasattr(self.model.loss_fn, "set_epoch"):
            raise TypeError(
                "EV-AT needs an EvidentialLoss as model.loss_fn (use loss=evidential)."
            )
        self.beta = beta
        self.divergence = LogDirichletDivergence(
            self._num_classes, mode=divergence_mode
        )
        self.adversary = EvidenceTargetedPGD(
            eps=adv_eps, steps=adv_steps, divergence=self.divergence
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # anneal the evidential-loss KL weight (Sensoy's lambda ramp)
        self.model.loss_fn.set_epoch(self.current_epoch)

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1]

        # Clean path (with gradient): evidential loss + the reference eta.
        logits, probs, preds, uncertainty = self.model.forward(x)
        alpha, eta_clean = self.model.dirichlet(logits)
        l_ev = self.model.loss_fn(alpha, y)

        # Inner max: evidence-targeted adversary (returns a detached x_adv).
        x_adv = self.adversary(self.model, x, y)

        # Robust evidence alignment on the adversarial path (with gradient).
        adv_logits = self.model.forward(x_adv)[0]
        _, eta_adv = self.model.dirichlet(adv_logits)
        l_rea = self.divergence(eta_clean, eta_adv, y)

        loss = l_ev + self.beta * l_rea

        # Refresh the IKL class-wise global statistics for the next step.
        self.divergence.update_global_stats(probs.detach(), y)

        # Guard so compute_loss stays callable outside a Trainer (tests).
        if self._trainer is not None:
            self.log("train_l_ev", l_ev, on_epoch=True)
            self.log("train_l_rea", l_rea, on_epoch=True)

        output = ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )
        return loss, output
