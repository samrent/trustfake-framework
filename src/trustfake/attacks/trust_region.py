import torch

from trustfake.attacks._common import model_logits, project_linf
from trustfake.attacks.abc import AdversarialAttack
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["TrustRegion"]


class TrustRegion(AdversarialAttack):
    r"""Trust Region attack, L_inf (Yao, Gholami, Xu, Keutzer & Mahoney,
    CVPR 2019).

    An adaptive-step first-order attack. Each iteration ascends the
    cross-entropy of the model's clean prediction (label-free, untargeted)
    with a signed-gradient step whose size is the trust-region radius; the
    radius grows when the linear model predicted the loss change well and
    shrinks when it did not, so the step adapts to the local curvature
    instead of using a fixed PGD stride. The perturbation is projected into
    the global L_inf eps-ball each step. Deterministic (no random start).

    Args:
        eps: L_inf budget.
        steps: number of trust-region iterations.
        r_init: initial trust-region radius (defaults to eps / 4).
        r_max: maximum radius (defaults to eps).
        grow: radius multiplier on a good linear-model agreement.
        shrink: radius multiplier on a poor agreement.
        rho: agreement threshold on the actual/predicted loss-change ratio.
        clip_min, clip_max: valid input range.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 20,
        r_init: float | None = None,
        r_max: float | None = None,
        grow: float = 1.5,
        shrink: float = 0.5,
        rho: float = 0.25,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.steps = steps
        self.r_init = r_init if r_init is not None else eps / 4
        self.r_max = r_max if r_max is not None else eps
        self.grow = grow
        self.shrink = shrink
        self.rho = rho

    @property
    def name(self) -> str:
        return "tr"

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()

        with torch.no_grad():
            yhat = model(inputs)[2].detach()  # clean prediction (label-free)

        x_adv = inputs.clone().detach()
        # per-sample trust-region radius
        r = torch.full(
            (inputs.shape[0],) + (1,) * (inputs.ndim - 1),
            float(self.r_init),
            device=inputs.device,
        )

        for _ in range(self.steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                loss_vec = torch.nn.functional.cross_entropy(
                    model_logits(model, x_adv), yhat, reduction="none"
                )
                grad = torch.autograd.grad(loss_vec.sum(), x_adv)[0]
            loss_before = loss_vec.detach()

            step = r * grad.sign()
            # linear-model predicted increase for a sign step of size r:
            # <g, r*sign(g)> = r * ||g||_1
            predicted = r.flatten(1).squeeze(1) * grad.detach().abs().flatten(1).sum(1)

            candidate = self._clamp(
                project_linf(x_adv.detach() + step, inputs, self.eps)
            )
            with torch.no_grad():
                loss_after = torch.nn.functional.cross_entropy(
                    model_logits(model, candidate), yhat, reduction="none"
                )
            actual = loss_after - loss_before
            ratio = actual / predicted.clamp_min(1e-12)

            # accept the step (ascent); adapt the radius per sample
            x_adv = candidate
            good = (ratio >= self.rho).view(r.shape)
            r = torch.where(
                good, (r * self.grow).clamp(max=self.r_max), r * self.shrink
            )

        model.train(was_training)
        return x_adv.detach()
