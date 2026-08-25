import torch

from trustfake.attacks.abc import AdversarialAttack, AttackFamily
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["UncertaintyFGSM"]


class UncertaintyFGSM(AdversarialAttack):
    """
    Fast Gradient Sign Method targeting the model's uncertainty score instead
    of its classification loss, cf. "Disrupting Deep Uncertainty Estimation
    Without Harming Accuracy". Single-step attack that perturbs each input in
    the direction that maximizes the uncertainty score:

        x_adv = clip(x + eps * sign(grad_x uncertainty(model(x))))

    This is label-free: no ground-truth or predicted class is involved, only
    the uncertainty output of `model`. It requires the configured
    `uncertainty_score` to keep gradients enabled through its
    `update`/`compute` (see `Metric._enable_grad`); otherwise `grad_x` will be
    disconnected from `x` and this attack will raise.

    Args:
        eps (float): Maximum L_inf perturbation size, in the same units as
            `inputs` (e.g. 8/255 for [0, 1]-scaled images).
        clip_min (float): Minimum valid value for a perturbed input.
        clip_max (float): Maximum valid value for a perturbed input.
    """

    family = AttackFamily.UNCERTAINTY

    def __init__(
        self, eps: float = 8 / 255, clip_min: float = 0.0, clip_max: float = 1.0
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)

    @property
    def name(self) -> str:
        return "uncertainty_fgsm"

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()

        with torch.enable_grad():
            x = inputs.clone().detach().requires_grad_(True)

            _, _, _, uncertainty = model(x)

            grad = torch.autograd.grad(uncertainty.mean(), x)[0]

            perturbed = x.detach() + self.eps * grad.sign()
            perturbed = self._clamp(perturbed)

        model.train(was_training)
        return perturbed.detach()
