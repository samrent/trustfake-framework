import torch

from trustfake.attacks._common import model_logits, project_l2, project_linf
from trustfake.attacks.abc import AdversarialAttack
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["PGD", "PGDL2"]


class PGD(AdversarialAttack):
    """
    Projected Gradient Descent, L_inf (Madry et al., ICLR 2018).

    Multi-step FGSM projected back into the eps-ball after each step, with a
    random start inside the ball. Random start matters: without it PGD is a
    multi-step FGSM that can stall in a flat region and overstate
    robustness. The start is drawn from a per-call seeded generator so the
    attack stays deterministic for a fixed model and input (a requirement of
    the evaluation harness).

    Attacks the model's own clean prediction by default, not the ground
    truth, to avoid label leaking. Set ``use_labels=True`` to attack the
    supplied targets instead -- a strictly stronger threat model (the
    attacker is assumed to know the answer), and therefore a different row in
    a results table, not a variant of the same one. `uses_labels` on the
    instance records which was used.

    Args:
        eps (float): L_inf budget.
        alpha (float | None): Step size. Defaults to 2.5 * eps / steps, the
            standard rule (Madry et al.; RobustBench).
        steps (int): Number of gradient steps.
        random_start (bool): Start from a random point in the eps-ball.
        use_labels (bool): Attack the supplied ground truth instead of the
            model's own clean prediction.
        seed (int): Seed for the random start, for determinism.
        clip_min, clip_max (float): Valid input range.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        alpha: float | None = None,
        steps: int = 10,
        random_start: bool = True,
        use_labels: bool = False,
        seed: int = 0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
        self.steps = steps
        self.random_start = random_start
        self.uses_labels = use_labels
        self.seed = seed

    @property
    def name(self) -> str:
        return "pgd"

    def _random_start(self, inputs: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=inputs.device).manual_seed(self.seed)
        noise = torch.empty_like(inputs).uniform_(-self.eps, self.eps, generator=gen)
        return self._clamp(project_linf(inputs + noise, inputs, self.eps))

    def _project(self, perturbed: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return project_linf(perturbed, inputs, self.eps)

    def _step_direction(self, grad: torch.Tensor) -> torch.Tensor:
        """Steepest ascent under the attack's norm: the sign for L_inf."""
        return grad.sign()

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
                targets = model(inputs)[2].detach()  # clean predictions

        x_adv = inputs.clone().detach()
        if self.random_start and self.eps > 0:
            x_adv = self._random_start(inputs)

        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                loss = model.loss_fn(model_logits(model, x_adv), targets.long())
                grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * self._step_direction(grad)
            x_adv = self._clamp(self._project(x_adv, inputs))

        model.train(was_training)
        return x_adv.detach()


class PGDL2(PGD):
    """
    Projected Gradient Descent, L2.

    The same fixed-budget attack as `PGD` in a different geometry: steps move
    along the *normalised gradient* rather than its sign, and each step is
    projected back into an L2 ball. Worth reporting alongside the L_inf
    version because robustness does not transfer between norms -- a detector
    hardened against an L_inf adversary can be undefended against an L2 one of
    comparable perceptual size, and reporting only L_inf hides that. Unlike
    the minimum-norm L2 attacks (DeepFool, C&W, BB), `eps` here is the search
    budget, not a cap on the answer.

    Note the L_inf contract does NOT hold: an L2 ball of radius eps permits a
    single pixel to move by eps, so `effective_eps` can reach eps but the
    perturbation is not eps-bounded in the L_inf sense the other attacks
    promise. Report the norm alongside the budget.

    Args:
        eps (float): L2 budget.
        alpha (float | None): Step size. Defaults to 2.5 * eps / steps.
        steps, random_start, use_labels, seed, clip_min, clip_max: as `PGD`.
    """

    norm = "l2"

    @property
    def name(self) -> str:
        return "pgd_l2"

    def _random_start(self, inputs: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator(device=inputs.device).manual_seed(self.seed)
        noise = torch.empty_like(inputs).normal_(0.0, 1.0, generator=gen)
        # A random direction scaled to a random radius inside the ball.
        flat = noise.flatten(1)
        direction = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        radius = (
            torch.rand(inputs.shape[0], 1, device=inputs.device, generator=gen)
            * self.eps
        )
        noise = (direction * radius).view_as(inputs)
        return self._clamp(project_l2(inputs + noise, inputs, self.eps))

    def _project(self, perturbed: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return project_l2(perturbed, inputs, self.eps)

    def _step_direction(self, grad: torch.Tensor) -> torch.Tensor:
        flat = grad.flatten(1)
        norm = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (flat / norm).view_as(grad)
