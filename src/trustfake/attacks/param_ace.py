import torch
from torch.nn.functional import cross_entropy

from trustfake.attacks._common import model_logits, project_linf
from trustfake.attacks.abc import (
    AdversarialAttack,
    AttackDirection,
    AttackFamily,
    AttackResult,
)
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["ParamACE"]


class ParamACE(AdversarialAttack):
    r"""
    (eta, omega)-ACE: parameterised Attack on Confidence Estimation.

    From Buerger et al., "Towards Certification of Uncertainty Calibration
    under Adversarial Attacks" (arXiv:2405.13922), Sec. 2.2, generalising
    Galil & El-Yaniv's ACE into a family:

        max   eta * L_CE(f(x + gamma), omega)
        s.t.  ||gamma||_inf <= eps  and  F(x + gamma) = F(x)

    - ``eta`` in {+1, -1}: direction. eta=+1 pushes the cross-entropy to
      ``omega`` UP, lowering confidence in that label; eta=-1 pushes it
      down, raising confidence.
    - ``omega`` in {"true", "pred"}: the label the confidence is measured
      against -- the ground truth ``y`` or the model's clean prediction
      ``yhat``. With "pred" the attack needs no labels.

    The prediction is held fixed (``F(x+gamma) = F(x)``) as an explicit
    constraint, so accuracy is unchanged by construction -- realised here as
    a per-step accept test (a step is kept only while the argmax is
    preserved), whose accepted logits are the reported ones (see
    ``AttackResult``). Optimisation is projected gradient ascent within the
    L_inf eps-ball; the paper fixes the objective but not the solver.

    Galil & El-Yaniv's ACE is the special case that lowers confidence where
    the model is right and raises it where it is wrong; see ``ACE`` for that
    behaviour with a per-sample epsilon search. ``ParamACE`` instead fixes
    (eta, omega) across the batch and takes fixed-size ascent steps.

    Args:
        eps (float): L_inf budget.
        eta (int): +1 (lower confidence) or -1 (raise confidence).
        omega (str): "true" (use y) or "pred" (use the clean prediction).
        alpha (float | None): Step size. Defaults to 2.5 * eps / steps.
        steps (int): Number of ascent steps.
        clip_min, clip_max (float): Valid input range.
    """

    family = AttackFamily.CONFIDENCE
    direction = AttackDirection.BOTH
    # omega='true' requires ground truth; the other omega modes do not.
    uses_labels = True

    def __init__(
        self,
        eps: float = 0.005,
        eta: int = 1,
        omega: str = "pred",
        alpha: float | None = None,
        steps: int = 10,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        if eta not in (1, -1):
            raise ValueError(f"eta must be +1 or -1, got {eta}")
        if omega not in ("true", "pred"):
            raise ValueError(f"omega must be 'true' or 'pred', got {omega!r}")
        self.eta = eta
        self.omega = omega
        self.alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
        self.steps = steps

    @property
    def name(self) -> str:
        return "param_ace"

    def run(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AttackResult:
        was_training = model.training
        model.eval()

        with torch.no_grad():
            clean_logits, _, preds, _ = model(inputs)
        preds = preds.detach()

        if self.omega == "true":
            if targets is None:
                model.train(was_training)
                raise ValueError("omega='true' requires ground-truth targets.")
            omega_label = targets.long()
        else:
            omega_label = preds

        out = inputs.clone()
        out_logits = clean_logits.detach().clone()
        x_adv = inputs.clone().detach()

        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model_logits(model, x_adv)
                # ascend eta * CE(., omega): grad of that objective
                obj = self.eta * cross_entropy(logits, omega_label)
                grad = torch.autograd.grad(obj, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = self._clamp(project_linf(x_adv, inputs, self.eps))

            # accept test: keep the step only where the argmax is preserved
            with torch.no_grad():
                step_logits, _, step_preds, _ = model(x_adv)
            keep = step_preds == preds
            out[keep] = x_adv[keep]
            out_logits[keep] = step_logits[keep]

        model.train(was_training)
        return AttackResult(
            perturbed=out.detach(),
            effective_eps=(out - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=out_logits.detach(),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
