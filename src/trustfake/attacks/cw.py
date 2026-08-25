import torch

from trustfake.attacks._common import model_logits, project_l2
from trustfake.attacks.abc import AdversarialAttack, AttackResult
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["CarliniWagner"]


class CarliniWagner(AdversarialAttack):
    r"""
    Carlini & Wagner L2 attack (IEEE S&P 2017).

    Optimises a perturbation via the tanh change of variables (which keeps
    the image in range without a hard clip) to minimise

        ||delta||_2^2  +  c * max(Z(x')_y - max_{i != y} Z(x')_i, -kappa)

    where the margin term drives the true-class logit below the strongest
    other class by at least ``kappa``. Untargeted, attacking the model's
    clean prediction to avoid label leaking. The best (smallest-norm)
    successful perturbation seen across steps is kept per sample.

    ``eps`` is not a search budget here -- C&W minimises distortion -- but an
    L2 cap applied to the result, which also satisfies L_inf <= eps
    (``||v||_inf <= ||v||_2``). Report that eps is an L2 radius. A single
    constant ``c`` is used rather than the paper's outer binary search, for
    a bounded, deterministic per-batch cost; raise ``c`` to trade norm for
    success rate. Deterministic.

    Args:
        eps (float): L2 radius the perturbation is capped to.
        c (float): Trade-off between perturbation norm and the margin term.
        kappa (float): Confidence margin; larger forces a wider logit gap.
        steps (int): Adam optimisation steps.
        lr (float): Adam learning rate on the tanh variable.
        clip_min, clip_max (float): Valid input range.
    """

    norm = "l2"
    minimum_norm = True

    def __init__(
        self,
        eps: float = 0.5,
        c: float = 1.0,
        kappa: float = 0.0,
        steps: int = 50,
        lr: float = 0.01,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    @property
    def name(self) -> str:
        return "cw"

    def _to_model_space(self, w: torch.Tensor) -> torch.Tensor:
        span = self.clip_max - self.clip_min
        return self.clip_min + (torch.tanh(w) + 1.0) * 0.5 * span

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        span = self.clip_max - self.clip_min
        t = ((x - self.clip_min) / span * 2.0 - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.atanh(t)

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
        n = inputs.shape[0]
        idx = torch.arange(n, device=inputs.device)
        other_mask = torch.ones_like(clean_logits, dtype=torch.bool)
        other_mask[idx, preds] = False

        w = self._to_tanh_space(inputs).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        best = inputs.clone().detach()
        best_norm = torch.full((n,), float("inf"), device=inputs.device)

        for _ in range(self.steps):
            optimizer.zero_grad()
            x_adv = self._to_model_space(w)
            logits = model_logits(model, x_adv)
            real = logits[idx, preds]
            other = logits.masked_fill(~other_mask, float("-inf")).max(dim=1).values
            margin = torch.clamp(real - other, min=-self.kappa)
            l2sq = (x_adv - inputs).flatten(1).pow(2).sum(dim=1)
            loss = (l2sq + self.c * margin).sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_adv = self._to_model_space(w)
                flipped = model(x_adv)[2] != preds
                norm = (x_adv - inputs).flatten(1).norm(dim=1)
                improve = flipped & (norm < best_norm)
                best[improve] = x_adv[improve]
                best_norm[improve] = norm[improve]

        best = self._clamp(project_l2(best.detach(), inputs, self.eps))
        with torch.no_grad():
            final_logits, _, final_preds, _ = model(best)

        model.train(was_training)
        return AttackResult(
            perturbed=best.detach(),
            effective_eps=(best - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=final_logits.detach(),
            l2_norm=(best - inputs).flatten(1).norm(dim=1).detach(),
            # Measured after the eps cap: a flip that only survives outside
            # the reported budget is not a success inside it.
            success=(final_preds.detach() != preds),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
