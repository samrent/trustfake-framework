import torch

from trustfake.attacks._common import project_linf
from trustfake.attacks.abc import AdversarialAttack, AttackFamily, AttackResult
from trustfake.losses import LogDirichletDivergence
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["EvidenceTargetedPGD"]


class EvidenceTargetedPGD(AdversarialAttack):
    r"""Evidence-targeted PGD (arXiv:2607.03075, Sec. 3.3): the inner
    maximisation of EV-AT, also usable as an evaluation attack against an
    evidential detector.

    Unlike a prediction-targeted attack, it does not maximise a
    classification loss; it maximises the drift of the Dirichlet posterior in
    log-concentration space,

        L_attack = D(eta_clean, eta(x_adv)),   eta = log(alpha),

    via K-step PGD from a random start x0 = x + U(-eps, eps), keeping the
    perturbation in the L_inf eps-ball. D is the same discrepancy used for
    robust evidence alignment (default IKL), so the adversary and the
    alignment loss are consistent. This searches for perturbations that
    corrupt the evidential posterior (and thus the selective score) rather
    than only flipping the argmax.

    Requires an evidential wrapper (one exposing ``dirichlet(logits) ->
    (alpha, eta)``). The random start is seeded for determinism.

    Args:
        eps: L_inf budget.
        steps: PGD steps K.
        alpha: step size; defaults to 2.5 * eps / steps.
        mode: divergence mode when no shared divergence is provided.
        divergence: an existing LogDirichletDivergence to share (e.g. the
            training module's L_REA, so global IKL stats are shared);
            built lazily from the class count otherwise.
        seed: seed for the random start.
        clip_min, clip_max: valid input range.
    """

    family = AttackFamily.EVIDENCE

    def __init__(
        self,
        eps: float = 0.03137,
        steps: int = 10,
        alpha: float | None = None,
        mode: str = "ikl",
        divergence: LogDirichletDivergence | None = None,
        seed: int = 0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.steps = steps
        self.alpha = alpha if alpha is not None else 2.5 * eps / max(steps, 1)
        self.mode = mode
        self.seed = seed
        self._divergence = divergence

    @property
    def name(self) -> str:
        return "evidence_pgd"

    def _get_divergence(self, num_classes: int, device) -> LogDirichletDivergence:
        if self._divergence is None:
            self._divergence = LogDirichletDivergence(num_classes, mode=self.mode)
        return self._divergence.to(device)

    def run(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AttackResult:
        if not hasattr(model, "dirichlet"):
            msg = (
                f"{type(model).__name__} is not an evidential wrapper; "
                "EvidenceTargetedPGD needs a `dirichlet` method."
            )
            raise TypeError(msg)

        was_training = model.training
        model.eval()

        with torch.no_grad():
            clean_logits, _, preds, _ = model(inputs)
            _, eta_clean = model.dirichlet(clean_logits)
            eta_clean = eta_clean.detach()
        divergence = self._get_divergence(clean_logits.shape[1], inputs.device)

        x_adv = inputs.clone().detach()
        if self.eps > 0:
            gen = torch.Generator(device=inputs.device).manual_seed(self.seed)
            noise = torch.empty_like(inputs).uniform_(
                -self.eps, self.eps, generator=gen
            )
            x_adv = self._clamp(project_linf(x_adv + noise, inputs, self.eps))

        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model(x_adv)[0]
                _, eta_adv = model.dirichlet(logits)
                # maximise drift: ascend D(eta_clean, eta_adv)
                loss = divergence(eta_clean, eta_adv, targets)
                grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = self._clamp(project_linf(x_adv, inputs, self.eps))

        with torch.no_grad():
            adv_logits = model(x_adv)[0]

        model.train(was_training)
        return AttackResult(
            perturbed=x_adv.detach(),
            effective_eps=(x_adv - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds.detach(),
            accepted_logits=adv_logits.detach(),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
