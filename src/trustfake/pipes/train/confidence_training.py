"""Confidence-targeted defences: the two arms that train against the failure
this harness exists to measure.

Every other arm in the suite defends the *label* axis. PGD-AT, TRADES, AT+KL
and MART all assume the adversary wants to change the prediction, and they are
evaluated on whether it still lands. But the attack that breaks a moderation
layer does not change the prediction at all -- ACE and the over-confidence
attack leave accuracy bit-identical and move only the confidence attached to
it, so selective risk collapses while an accuracy monitor reports a healthy
system. Nothing that defends the label axis is even aimed at that.

These two arms are aimed at it, in the two shapes the problem admits:

  * `ConfidenceAdversarialTrainingModule` (train-time, adversarial). The
    inner maximisation inflates confidence instead of destroying accuracy;
    the outer loss then demands correctness on those confidence-inflated
    inputs. The model is being taught that an attacker cannot manufacture a
    confident *wrong* answer.
  * `ConfidenceRegularisedTrainingModule` (train-time, no adversary). A
    direct penalty on confident mistakes. No inner attack at all, so it costs
    one forward per step -- and it optimises the failure-prediction property
    the moderation layer reads, rather than a proxy for it.

Both must be reported as NON-ADAPTIVE results unless an attacker is
subsequently optimised against them. A defence evaluated only against the
attack it was trained on is the standard way robustness claims dissolve
(Athalye et al. 2018), and the honest framing here is that these are measured
against a fixed attack. Even a negative result is worth having: "confidence
adversarial training does not restore failure-AUROC under an adaptive attack"
is a finding, because the question is open.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, softmax

from trustfake.attacks._common import model_logits
from trustfake.pipes.train._common import RobustValidationMixin, eval_mode
from trustfake.pipes.train.abc import TrainingModule
from trustfake.pipes.train.adversarial_training import _AdversarialTrainingBase
from trustfake.pipes.train.awp import AWPMixin
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = [
    "ConfidenceAdversarialTrainingModule",
    "ConfidenceRegularisedTrainingModule",
]


class ConfidenceAdversarialTrainingModule(_AdversarialTrainingBase):
    """Adversarial training whose inner maximisation is the *confidence*
    attack (Ledda et al. 2025), not cross-entropy.

    The inner loop freezes the model's own clean prediction `yhat` and then
    *descends* cross-entropy toward it -- pushing probability mass onto the
    already-predicted class, i.e. inflating confidence without ever consulting
    a label. This is the over-confidence attack, run as the inner adversary.

    The outer loss is ordinary cross-entropy against the true label on those
    inflated inputs. The composition is the point: the adversary produces the
    most confident version of the model's current answer, and the objective
    demands that answer be right. Where the model would have been confidently
    wrong, the two disagree and the gradient is large.

    Note the inner attack is label-free, which makes it the realisable threat
    model -- an attacker in production does not have the ground truth. Only
    the outer loss uses labels, as training must.
    """

    def _pgd_overconfidence(self, x: Tensor, eps: float) -> Tensor:
        """Inner max: worst-case confidence inflation on the frozen prediction.

        Implemented by ascending the NEGATIVE cross-entropy to `yhat`, so it
        reuses the shared `_inner_pgd` scaffold (which ascends) without
        inverting the step sign in two places -- a sign error here produces a
        working, plausible-looking attack that does the exact opposite.
        """
        # Eval mode: freezing the prediction is a read of the model, not a
        # training forward, so it must not update BatchNorm either.
        with torch.no_grad(), eval_mode(self.model):
            yhat = self.model(x)[2].detach().long()

        def objective(x_adv: Tensor) -> Tensor:
            return -cross_entropy(model_logits(self.model, x_adv), yhat)

        return self._inner_pgd(x, objective, eps)

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        x_adv = self._pgd_overconfidence(x, self.current_eps)
        self.apply_awp(x, x_adv, y)

        adv_logits = model_logits(self.model, x_adv)
        loss = self.model.loss_fn(adv_logits, y)

        return loss, self._metrics_output(x)


class ConfidenceRegularisedTrainingModule(
    AWPMixin, RobustValidationMixin, TrainingModule
):
    """Cross-entropy plus a penalty on confident mistakes.

    ``L = CE(f(x), y) + lambda * mean( max_k p_k(x) * 1[argmax != y] )``

    No inner attack: one forward per step, the same cost as standard
    training. The indicator is detached, so the penalty scales the confidence
    of the errors without back-propagating through the decision of *which*
    samples are errors -- that decision is a step function and has no useful
    gradient.

    What it optimises is exactly the quantity the WP4 moderation layer reads.
    Selective risk is governed by whether confidence *ranks* mistakes below
    correct answers; penalising the confidence of mistakes attacks that
    ranking directly, rather than hoping it improves as a side effect of
    better accuracy. A model with the same accuracy and a better ranking is
    strictly more useful under abstention, and no accuracy-based objective
    can express that preference.

    Args:
        lambda_reg: Weight of the confidence penalty. 0 reduces this arm to
            standard training, which is the control it should be read against.
    """

    def __init__(self, *args, lambda_reg: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg

    def _objective(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, tuple]:
        forward = self.model.forward(x)
        logits = forward[0]
        confidence = softmax(logits, dim=1).amax(dim=1)
        wrong = (forward[2].long() != y).float().detach()
        penalty = (confidence * wrong).mean()
        return (
            self.model.loss_fn(logits, y) + self.lambda_reg * penalty,
            penalty,
            forward,
        )

    def awp_objective(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> Tensor:
        """This arm has no inner adversary, so `x_adv` IS the clean batch and
        the weight adversary ascends the arm's own objective on it."""
        return self._objective(inputs, targets.long())[0]

    def compute_loss(
        self, batch: list[Tensor, Tensor]
    ) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()

        # This arm generates no adversarial batch, so the clean one is passed
        # for both. Omitting this call entirely -- which is what happened --
        # left `awp_gamma` accepted from the config, stored, and never read:
        # the with/without-AWP ablation would have produced two bit-identical
        # arms and read as "AWP does not help the confidence-regularised arm".
        self.apply_awp(x, x, y)

        loss, penalty, (logits, probs, preds, uncertainty) = self._objective(x, y)

        # Log only from the training path. `validation_step` calls this same
        # method, and the two hooks have different `on_step` defaults, so a
        # bare `train_confidence_penalty` column ended up holding the
        # VALIDATION value while the training value hid in `..._epoch`.
        if self._trainer is not None and self.training:
            self.log("train_confidence_penalty", penalty, on_epoch=True)

        output = ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )
        return loss, output
