"""Label-free confidence-shift attacks (Ledda et al. 2025).

Both attack the confidence attached to the model's own clean prediction --
they never use ground truth, which is ACE's practical weakness. They form a
pair the WP4 moderation layer prices asymmetrically: over-confidence inflates
residual risk (auto-deciding work it gets wrong), under-confidence inflates
review rate (abstaining on work it could do).
"""

import torch
from torch.nn.functional import cross_entropy, kl_div, log_softmax

from trustfake.attacks._common import model_logits, project_linf
from trustfake.attacks.abc import AdversarialAttack, AttackDirection, AttackFamily
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["OverConfidence", "UnderConfidence"]


class _ConfidenceShift(AdversarialAttack):
    """Shared PGD scaffold; subclasses supply the per-step objective on the
    frozen clean prediction. Both descend their loss (minimise)."""

    family = AttackFamily.CONFIDENCE

    def __init__(
        self,
        eps: float = 4 / 255,
        alpha: float | None = None,
        steps: int = 20,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.alpha = alpha if alpha is not None else max(eps / 8, 0.5 / 255)
        self.steps = steps

    def _loss(self, logits: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()

        with torch.no_grad():
            yhat = model(inputs)[2].detach()  # frozen clean prediction (label-free)

        x_adv = inputs.clone().detach()
        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                loss = self._loss(model_logits(model, x_adv), yhat)
                grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() - self.alpha * grad.sign()  # descend (minimise)
            x_adv = self._clamp(project_linf(x_adv, inputs, self.eps))

        model.train(was_training)
        return x_adv.detach()


class OverConfidence(_ConfidenceShift):
    r"""Minimise H(f(x+d), onehot(yhat)) with yhat frozen: push mass TOWARD the
    predicted class. Label-preserving by construction (the argmax can only be
    reinforced), so accuracy is unchanged while failure detection degrades --
    an accuracy monitor sees a healthy system.
    """

    direction = AttackDirection.OVER

    @property
    def name(self) -> str:
        return "overconf"

    def _loss(self, logits: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, yhat)  # descend -> confidence up


class UnderConfidence(_ConfidenceShift):
    r"""Minimise KL(f(x+d) || uniform): drive the prediction toward maximum
    entropy (Ledda et al. Eq. 11). The uniform target's optimum is at margin
    zero, so it stops at maximum uncertainty rather than flipping the label --
    but it is NOT label-preserving by construction, so preservation is measured
    (via AttackResult), not assumed.
    """

    direction = AttackDirection.UNDER

    @property
    def name(self) -> str:
        return "underconf"

    def _loss(self, logits: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        logp = log_softmax(logits, dim=1)
        uniform = torch.full_like(logp, -torch.log(torch.tensor(float(logp.shape[1]))))
        return kl_div(logp, uniform, log_target=True, reduction="batchmean")
