import torch

from trustfake.attacks.abc import AdversarialAttack
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["FGSM"]


class FGSM(AdversarialAttack):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015).

    Single-step attack that perturbs each input in the direction that
    maximizes the model's loss:

        x_adv = clip(x + eps * sign(grad_x loss(model(x), y)))

    Args:
        eps (float): Maximum L_inf perturbation size, in the same units as
            `inputs` (e.g. 8/255 for [0, 1]-scaled images).
        use_labels (bool): Attack the supplied ground truth instead of the
            model's own clean prediction. A strictly stronger threat model --
            the attacker is assumed to know the answer -- so a row produced
            this way is a different attack, not a variant of the same one.
        clip_min (float): Minimum valid value for a perturbed input.
        clip_max (float): Maximum valid value for a perturbed input.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        use_labels: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.uses_labels = use_labels

    @property
    def name(self) -> str:
        return "fgsm"

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if model.loss_fn is None:
            msg = "FGSM requires `model.loss_fn` to be set."
            raise ValueError(msg)

        was_training = model.training
        model.eval()

        with torch.enable_grad():
            x = inputs.clone().detach().requires_grad_(True)

            logits, _, preds, _ = model(x)

            if not self.uses_labels or targets is None:
                # Avoid label leaking: attack the model's own prediction.
                targets = preds.detach()

            loss = model.loss_fn(logits, targets.long())
            grad = torch.autograd.grad(loss, x)[0]

            perturbed = x.detach() + self.eps * grad.sign()
            perturbed = self._clamp(perturbed)

        model.train(was_training)
        return perturbed.detach()
