import torch

from trustfake.attacks._common import model_logits, project_linf
from trustfake.attacks.abc import AdversarialAttack
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["PGD"]


class PGD(AdversarialAttack):
    """
    Projected Gradient Descent, L_inf (Madry et al., ICLR 2018).

    Multi-step FGSM projected back into the eps-ball after each step, with a
    random start inside the ball. Random start matters: without it PGD is a
    multi-step FGSM that can stall in a flat region and overstate
    robustness. The start is drawn from a per-call seeded generator so the
    attack stays deterministic for a fixed model and input (a requirement of
    the evaluation harness).

    Attacks the model's own clean prediction, not the ground truth, to avoid
    label leaking.

    Args:
        eps (float): L_inf budget.
        alpha (float | None): Step size. Defaults to 2.5 * eps / steps, the
            standard rule (Madry et al.; RobustBench).
        steps (int): Number of gradient steps.
        random_start (bool): Start from a random point in the eps-ball.
        seed (int): Seed for the random start, for determinism.
        clip_min, clip_max (float): Valid input range.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        alpha: float | None = None,
        steps: int = 10,
        random_start: bool = True,
        seed: int = 0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
        self.steps = steps
        self.random_start = random_start
        self.seed = seed

    @property
    def name(self) -> str:
        return "pgd"

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()

        with torch.no_grad():
            targets = model(inputs)[2].detach()  # clean predictions

        x_adv = inputs.clone().detach()
        if self.random_start and self.eps > 0:
            gen = torch.Generator(device=inputs.device).manual_seed(self.seed)
            noise = torch.empty_like(inputs).uniform_(
                -self.eps, self.eps, generator=gen
            )
            x_adv = self._clamp(project_linf(x_adv + noise, inputs, self.eps))

        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                loss = model.loss_fn(model_logits(model, x_adv), targets.long())
                grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = self._clamp(project_linf(x_adv, inputs, self.eps))

        model.train(was_training)
        return x_adv.detach()
