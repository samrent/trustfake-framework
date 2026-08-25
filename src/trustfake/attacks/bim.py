import torch

from trustfake.attacks._common import model_logits, project_linf
from trustfake.attacks.abc import AdversarialAttack
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["BIM"]


class BIM(AdversarialAttack):
    """
    Basic Iterative Method (Kurakin, Goodfellow & Bengio, 2017).

    Iterative FGSM with no random start and a fixed small step, clipped to
    the eps-ball each iteration. It is exactly PGD without the random start,
    kept as its own attack because that is how it is reported in the
    literature and because the no-restart version is the weaker, more
    reproducible baseline. Deterministic by construction.

    Attacks the model's own clean prediction to avoid label leaking.

    Args:
        eps (float): L_inf budget.
        alpha (float | None): Step size. Defaults to eps / steps.
        steps (int): Number of gradient steps.
        use_labels (bool): Attack the supplied ground truth instead of the
            model's own clean prediction -- a strictly stronger threat model.
        clip_min, clip_max (float): Valid input range.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        alpha: float | None = None,
        steps: int = 10,
        use_labels: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.alpha = alpha if alpha is not None else eps / max(steps, 1)
        self.steps = steps
        self.uses_labels = use_labels

    @property
    def name(self) -> str:
        return "bim"

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()

        if not self.uses_labels or targets is None:
            with torch.no_grad():
                targets = model(inputs)[2].detach()

        x_adv = inputs.clone().detach()
        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                loss = model.loss_fn(model_logits(model, x_adv), targets.long())
                grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = self._clamp(project_linf(x_adv, inputs, self.eps))

        model.train(was_training)
        return x_adv.detach()
