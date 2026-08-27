"""Gradient-free confidence attacks -- the decisive test of the paper's premise.

Every other confidence attack in this suite (ACE, OverConfidence,
UnderConfidence) navigates by the model's gradient. The evidential head masks
gradients: on the undefended evidential model PGD reports 0.54 robust accuracy
while gradient-free Square reports 0.004. So a confidence-robustness number
taken from a gradient attack against an evidential model may be measuring a
stalled attacker rather than a robust defence -- exactly the failure mode this
harness exists to catch.

`QueryConfidence` removes the gradient. It is a random-search (Square-style)
attack that pushes the model's OWN uncertainty score while holding the argmax
fixed, using forward passes only. If a defence's failure-detection AUROC
survives THIS, the confidence robustness is real; if it survives the gradient
attacks but folds here, it was gradient masking. It is the confidence-axis
analogue of what Square already did on the prediction axis.

The label-free frame is inherited from Ledda et al.: the attack knows only the
model's clean prediction yhat, never the ground truth. `direction="over"`
minimises uncertainty (drives confidence up in yhat -- confidently wrong where
the model is wrong, which is what destroys the selective signal);
`direction="under"` maximises it. The argmax-preservation accept test makes it
a pure CONFIDENCE attack: accuracy is unchanged by construction, so any change
in selective risk is attributable to the confidence axis alone.
"""

from __future__ import annotations

import torch

from trustfake.attacks.abc import (
    AdversarialAttack,
    AttackDirection,
    AttackFamily,
    AttackResult,
)
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["QueryConfidence"]


class QueryConfidence(AdversarialAttack):
    """Square-style, gradient-free confidence attack.

    Args:
        eps: L_inf budget.
        n_queries: forward passes per sample (the query budget). A confidence
            nudge under an argmax constraint needs far fewer than a prediction
            flip; a few hundred suffices and keeps a full-suite run tractable.
        direction: "over" minimises the uncertainty score, "under" maximises
            it. Over-confidence is the one that inverts failure detection.
        p_init: initial fraction of pixels a patch covers; it anneals down as
            the budget is spent, the Square schedule.
        seed: fixes the random search so the attack is deterministic per input.
    """

    family = AttackFamily.CONFIDENCE
    direction = AttackDirection.BOTH
    uses_labels = False

    def __init__(
        self,
        eps: float = 8 / 255,
        n_queries: int = 400,
        direction: str = "over",
        p_init: float = 0.3,
        seed: int = 0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        if direction not in ("over", "under"):
            raise ValueError(f"direction must be over|under, got {direction!r}")
        self.n_queries = n_queries
        self._dir = direction
        self.p_init = p_init
        self.seed = seed

    @property
    def name(self) -> str:
        return f"query_{self._dir}conf"

    def _p(self, it: int) -> float:
        # Square's piecewise-constant schedule, compressed to the budget.
        frac = it / max(self.n_queries, 1)
        for cut, val in ((0.5, 0.5), (0.2, 0.7), (0.1, 0.85), (0.05, 0.92)):
            if frac >= cut:
                return self.p_init * val
        return self.p_init

    def run(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AttackResult:
        was_training = model.training
        model.eval()
        gen = torch.Generator(device=inputs.device).manual_seed(self.seed)
        b, c, h, w = inputs.shape

        with torch.no_grad():
            clean_logits, _, yhat, u_clean = model(inputs)
            yhat = yhat.detach()

            best = inputs.clone()
            best_u = u_clean.detach().clone()
            best_logits = clean_logits.detach().clone()
            sign = 1.0 if self._dir == "over" else -1.0  # over = minimise u

            def better(u_new):
                # A candidate is better when it moves uncertainty the wanted
                # way; sign folds the two directions into one comparison.
                return (sign * u_new) < (sign * best_u)

            for it in range(self.n_queries):
                p = self._p(it)
                s = max(1, int(round((p * h * w) ** 0.5)))
                s = min(s, h, w)
                # One random square patch per sample, one random channel,
                # perturbation set to +/- eps at full budget.
                y0 = torch.randint(
                    0, h - s + 1, (b,), generator=gen, device=inputs.device
                )
                x0 = torch.randint(
                    0, w - s + 1, (b,), generator=gen, device=inputs.device
                )
                ch = torch.randint(0, c, (b,), generator=gen, device=inputs.device)
                signs = torch.where(
                    torch.rand(b, generator=gen, device=inputs.device) < 0.5,
                    -self.eps,
                    self.eps,
                )
                cand = best.clone()
                rows = torch.arange(s, device=inputs.device)
                for i in range(b):
                    ys = (y0[i] + rows).clamp_max(h - 1)
                    xs = (x0[i] + rows).clamp_max(w - 1)
                    patch = inputs[i, ch[i]][ys][:, xs] + signs[i]
                    cand[i, ch[i]].index_put_(
                        (ys.unsqueeze(1), xs.unsqueeze(0)),
                        patch.clamp(self.clip_min, self.clip_max),
                    )
                cand = torch.min(
                    torch.max(cand, inputs - self.eps), inputs + self.eps
                ).clamp(self.clip_min, self.clip_max)

                logits, _, preds, u = model(cand)
                # Accept only where the argmax is preserved (pure confidence
                # attack) AND uncertainty moved the wanted way.
                take = (preds == yhat) & better(u)
                if bool(take.any()):
                    m = take.view(b, 1, 1, 1)
                    best = torch.where(m, cand, best)
                    best_u = torch.where(take, u, best_u)
                    best_logits = torch.where(take.view(b, 1), logits, best_logits)

        model.train(was_training)
        return AttackResult(
            perturbed=best.detach(),
            effective_eps=(best - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=yhat,
            accepted_logits=best_logits.detach(),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
