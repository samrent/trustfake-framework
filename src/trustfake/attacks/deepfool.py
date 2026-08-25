import torch

from trustfake.attacks._common import finalise_minimum_norm, model_logits, project_l2
from trustfake.attacks.abc import AdversarialAttack, AttackResult
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["DeepFool"]


class DeepFool(AdversarialAttack):
    """
    DeepFool (Moosavi-Dezfooli, Fawzi & Frossard, CVPR 2016).

    A minimum-norm attack: at each step it linearises the classifier and
    takes the smallest L2 step that reaches the nearest decision boundary,
    stopping as soon as the prediction changes. Unlike a fixed-budget attack
    it does not use eps to drive the search; instead ``eps`` caps the result
    to the L2 ball of that radius. Because ``||v||_inf <= ||v||_2``, that cap
    also satisfies the framework's L_inf <= eps contract, so DeepFool composes
    with the rest of the suite; report that eps is an L2 radius here.

    A small overshoot factor pushes just past the boundary so the flip is
    numerically stable. Deterministic.

    A sample the attack cannot flip inside the cap is returned UNPERTURBED
    with ``success=False`` and ``l2_norm == 0``, as in every other min-norm
    attack here: capping the failed walk instead would report ``l2_norm ==
    eps`` on a failure, which is indistinguishable from a genuine
    eps-cost break.

    Args:
        eps (float): L2 radius the perturbation is capped to.
        steps (int): Maximum linearisation steps.
        overshoot (float): Multiplier applied to each boundary step (>1).
        clip_min, clip_max (float): Valid input range.
    """

    norm = "l2"
    minimum_norm = True

    def __init__(
        self,
        eps: float = 0.5,
        steps: int = 50,
        overshoot: float = 1.02,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        witness_seed: int = 0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.steps = steps
        self.overshoot = overshoot
        # Seeds the random restart of the witness retry in
        # `finalise_minimum_norm`; DeepFool itself is deterministic.
        self.witness_seed = witness_seed

    @property
    def name(self) -> str:
        return "deepfool"

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
        n_classes = clean_logits.shape[1]

        x_adv = inputs.clone().detach()
        active = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        expand = (-1,) + (1,) * (inputs.ndim - 1)

        for _ in range(self.steps):
            if not active.any():
                break
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model_logits(model, x_adv)
            # Jacobian of the logits wrt input, one class at a time.
            grads = []
            for c in range(n_classes):
                g = torch.autograd.grad(
                    logits[:, c].sum(), x_adv, retain_graph=(c < n_classes - 1)
                )[0]
                grads.append(g.flatten(1))
            grads = torch.stack(grads, dim=1)  # (B, C, D)
            logits = logits.detach()

            g_pred = grads[torch.arange(len(preds)), preds]  # (B, D)
            f_pred = logits[torch.arange(len(preds)), preds]  # (B,)

            best_r = torch.zeros_like(x_adv.detach().flatten(1))
            # Whether a MEANINGFUL nearest-boundary candidate was located for
            # this sample: `w` is a difference of logit gradients, and where
            # it vanishes (a locally flat model) the clamp below is the only
            # thing keeping `dist = |f| / ||w||` finite -- the resulting
            # `r = (dist / ||w||) * w` is then an enormous step along a
            # direction that carries no information. Gating on it keeps such
            # a sample where it is, so the attack reports failure instead of
            # a wild perturbation.
            found = torch.zeros(len(preds), dtype=torch.bool, device=inputs.device)
            min_dist = torch.full(
                (len(preds),), float("inf"), device=inputs.device, dtype=inputs.dtype
            )
            for c in range(n_classes):
                w = grads[:, c] - g_pred  # (B, D)
                f = logits[:, c] - f_pred  # (B,)
                raw_wn = w.norm(dim=1)
                wn = raw_wn.clamp_min(1e-8)
                dist = f.abs() / wn
                take = (c != preds) & (dist < min_dist) & (raw_wn > 1e-8)
                r = (dist / wn)[:, None] * w
                best_r = torch.where(take[:, None], r, best_r)
                min_dist = torch.where(take, dist, min_dist)
                found = found | take

            step = self.overshoot * best_r.view_as(x_adv)
            # Rank-agnostic: `inputs` is not required to be an image batch.
            moving = (active & found).view(expand).to(step.dtype)
            x_adv = self._clamp(x_adv.detach() + step * moving)

            with torch.no_grad():
                _, _, new_preds, _ = model(x_adv)
            active = active & (new_preds == preds)

        candidate = self._clamp(project_l2(x_adv.detach(), inputs, self.eps))
        # Success measured after the eps cap, and every failure returned
        # clean, so a failed sample's reported norm is 0 rather than the cap
        # -- see `finalise_minimum_norm` for why that consistency matters.
        x_adv, success, final_logits = finalise_minimum_norm(
            model,
            inputs,
            candidate,
            preds,
            clean_logits.detach(),
            eps=self.eps,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            seed=self.witness_seed,
        )
        delta = (x_adv - inputs).flatten(1)

        model.train(was_training)
        return AttackResult(
            perturbed=x_adv,
            effective_eps=delta.abs().amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=final_logits,
            l2_norm=delta.norm(dim=1).detach(),
            success=success,
            minimised_norm="l2",
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
