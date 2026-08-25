"""PDPGD: primal-dual proximal gradient descent minimum-norm attack.

Matyasko & Chau, "PDPGD: Primal-Dual Proximal Gradient Descent Adversarial
Attack" (2021). A native reimplementation from the paper's algorithm -- the
authors' code is not on PyPI, and vendoring an unverifiable research repo to
run inside the evaluation loop is the failure mode this suite is built to
avoid: a subtly wrong attack does not announce itself, it just reports
robustness the model does not have.
"""

from __future__ import annotations

import torch

from trustfake.attacks._common import (
    class_margin,
    finalise_minimum_norm,
    model_logits,
    project_l1_ball,
    project_l2,
    project_linf,
)
from trustfake.attacks.abc import AdversarialAttack, AttackResult
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["PDPGD"]


class PDPGD(AdversarialAttack):
    r"""PDPGD minimum-norm attack (Matyasko & Chau, 2021), L_inf or L2.

    Where PGD fixes a budget and maximises damage inside it, PDPGD fixes the
    goal -- change the prediction -- and minimises the norm needed to reach
    it, by solving the constrained problem

    .. math::
        \min_\delta \|\delta\|_p \quad\text{s.t.}\quad
        b(x + \delta) \le 0,\qquad
        b(x) = z_y(x) - \max_{c \neq y} z_c(x)

    with a primal-dual scheme. The primal step is proximal gradient descent
    on the Lagrangian: a gradient step on the (dual-weighted) margin followed
    by the proximal operator of the norm, which is what actually shrinks the
    perturbation -- for L_inf that operator is
    ``prox_{t||.||_inf}(v) = v - proj_{||u||_1 <= t}(v)`` by Moreau
    decomposition, so it clips the largest coordinates rather than scaling
    everything down. The dual step is multiplicative ascent on the
    constraint, ``lambda <- lambda * exp(alpha * b)``, which keeps
    ``lambda > 0`` without a projection and raises the pressure on the
    constraint exactly on the samples that are still not adversarial.

    **Both halves are scale-free, and that is load-bearing.** A decision
    boundary is invariant to the model's logit scale -- multiply every logit
    by a constant and no sample changes class -- so the norm needed to cross
    it is invariant too, and an attack whose step size depends on that scale
    is not measuring the boundary. Two things enforce it here:

    * the primal step takes a UNIT step in the norm being minimised
      (``sign(grad)`` for L_inf, ``grad/||grad||_2`` for L2) scaled by
      ``lr * lambda``, so its magnitude is in *input* units. An unnormalised
      ``lr * lambda * grad`` instead moves by whatever the logit scale
      happens to be: on a small-logit model it never travels far enough to
      cross, reports ``success=False`` on samples a plain fixed-budget PGD
      breaks at the same eps, and the run looks completed.
    * the proximal threshold is a FRACTION of the perturbation's own current
      scale (``lr * prox_weight * ||delta||_1`` for L_inf, whose prox takes
      an L1 radius; ``lr * prox_weight * ||delta||_2`` for L2). An absolute
      threshold is inert as soon as ``delta`` is larger than it -- and since
      the shrinking half of the method is the half that makes the reported
      norm minimal, an inert prox turns PDPGD into ordinary gradient descent
      that reports whatever norm it stopped at.

    The dual normaliser is per-sample for the same reason it is deterministic
    elsewhere in the suite: ``margin`` is divided by that sample's own clean
    margin, not by the batch maximum, so a sample's result does not depend on
    which other samples happened to share its batch.

    The best (smallest-norm) adversarial iterate is kept per sample, so the
    result never degrades over iterations. Samples never driven across the
    boundary are returned unperturbed and flagged in
    ``AttackResult.success``: a min-norm attack reports failure rather than
    returning a perturbation it could not make work.

    ``eps`` does not drive the search -- it caps the result, in the norm
    given by ``norm``. With ``norm="linf"`` that cap is the framework's L_inf
    contract directly; with ``norm="l2"`` it is an L2 radius, which implies
    the same L_inf bound.

    The minimised quantity is reported under the norm it was minimised in:
    ``AttackResult.minimised_norm`` names it and ``AttackResult.minimised``
    returns it (``effective_eps`` for L_inf, ``l2_norm`` for L2).
    ``l2_norm`` is left unset in L_inf mode rather than filled with the L2
    norm of an L_inf-minimised perturbation, which is a different quantity
    and would silently share an axis with the L2 attacks' answers.

    Deterministic: no random start.

    Args:
        eps: Radius the final perturbation is capped to, in `norm`.
        steps: Primal-dual iterations.
        norm: ``"linf"`` or ``"l2"`` -- the norm being minimised.
        lr: Initial primal step size in INPUT units (the gradient is
            normalised), cosine-annealed to ``lr * lr_final``.
        lr_final: Fraction of `lr` the schedule ends at.
        dual_lr: Multiplicative dual ascent rate.
        lam_init: Initial dual variable.
        prox_weight: Fraction of the perturbation's own scale the proximal
            operator removes per step, before the `lr` schedule: the
            threshold is ``lr * prox_weight * scale(delta)``. Raise it to
            shrink harder at the cost of crossing the boundary less often.
        clip_min, clip_max: Valid input range.
    """

    minimum_norm = True

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 100,
        norm: str = "linf",
        lr: float = 0.02,
        lr_final: float = 0.01,
        dual_lr: float = 0.1,
        lam_init: float = 1.0,
        prox_weight: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        if norm not in ("linf", "l2"):
            raise ValueError(f"norm must be 'linf' or 'l2', got {norm!r}")
        self.steps = steps
        self.norm = norm
        self.lr = lr
        self.lr_final = lr_final
        self.dual_lr = dual_lr
        self.lam_init = lam_init
        self.prox_weight = prox_weight

    @property
    def name(self) -> str:
        return "pdpgd"

    def _prox(self, delta: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Proximal operator of `threshold * ||.||_p`, per sample.

        `threshold` is a per-sample tensor, shape (B,): the operator's
        strength has to track each sample's own perturbation scale, so a
        single scalar for the batch is not enough.
        """
        if self.norm == "linf":
            # Moreau: prox_{t||.||_inf}(v) = v - proj_{||u||_1 <= t}(v).
            return delta - project_l1_ball(delta, threshold)
        # Block soft-threshold: prox_{t||.||_2}(v) = v * max(0, 1 - t/||v||).
        flat = delta.flatten(1)
        norm = flat.norm(dim=1).clamp_min(1e-12)
        scale = (1.0 - threshold / norm).clamp_min(0.0)
        return (flat * scale[:, None]).view_as(delta)

    def _prox_scale(self, delta: torch.Tensor) -> torch.Tensor:
        """The perturbation's own scale, in the units the prox threshold takes.

        The L_inf prox is driven by an L1 radius and the L2 prox by an L2
        radius, so the threshold that removes a fixed FRACTION of the
        perturbation each step is that fraction times this. Using an absolute
        threshold instead makes the operator inert the moment `delta` grows
        past it -- which is exactly when the shrinking is supposed to start.
        """
        flat = delta.flatten(1)
        if self.norm == "linf":
            return flat.abs().sum(dim=1)
        return flat.norm(dim=1)

    def _step_direction(self, grad: torch.Tensor) -> torch.Tensor:
        """Unit step in the norm being minimised (steepest descent there).

        Normalised, so ``lr`` is a distance in input space rather than a
        distance multiplied by the model's logit scale. Every other min-norm
        attack in the suite is scale-invariant; this is what makes this one
        so too.
        """
        if self.norm == "linf":
            return grad.sign()
        flat = grad.flatten(1)
        return (flat / flat.norm(dim=1).clamp_min(1e-12)[:, None]).view_as(grad)

    def _norm_of(self, delta: torch.Tensor) -> torch.Tensor:
        flat = delta.flatten(1)
        if self.norm == "linf":
            return flat.abs().amax(dim=1)
        return flat.norm(dim=1)

    def _project(self, perturbed: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if self.norm == "linf":
            return project_linf(perturbed, inputs, self.eps)
        return project_l2(perturbed, inputs, self.eps)

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
        clean_logits = clean_logits.detach()
        # Per-sample reference for the dual normaliser below. Scales with the
        # logit scale exactly as `margin` does, so their ratio does not.
        clean_margin = class_margin(clean_logits, preds).abs().clamp_min(1e-12)

        batch = inputs.shape[0]
        expand = (-1,) + (1,) * (inputs.ndim - 1)
        delta = torch.zeros_like(inputs)
        lam = torch.full(
            (batch,), float(self.lam_init), device=inputs.device, dtype=inputs.dtype
        )

        best = torch.zeros_like(inputs)
        best_norm = torch.full(
            (batch,), float("inf"), device=inputs.device, dtype=inputs.dtype
        )

        for step in range(self.steps):
            # Cosine schedule on the primal step size (the paper's decaying
            # step; the exact schedule is a hyperparameter, the decay is not).
            cos = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * step / self.steps)))
            lr = self.lr * (self.lr_final + (1 - self.lr_final) * float(cos))

            d = delta.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model_logits(model, self._clamp(inputs + d))
                margin = class_margin(logits, preds)
                grad = torch.autograd.grad(margin.sum(), d)[0]
            margin = margin.detach()

            # Primal: dual-weighted descent on the margin, then the prox of
            # the norm (this is the step that shrinks delta). The direction
            # is a unit step in the minimised norm, so `lr` is a distance in
            # input space and nothing here depends on the logit scale.
            d = delta - lr * lam.view(expand) * self._step_direction(grad)
            d = self._prox(d, lr * self.prox_weight * self._prox_scale(d))
            # Stay inside the box in *perturbation* space, so the prox and
            # the clip cannot fight each other across iterations.
            delta = (self._clamp(inputs + d) - inputs).detach()

            # Dual: raise the pressure where the constraint is still violated
            # (margin > 0, i.e. not yet adversarial), release it where it is
            # satisfied. Multiplicative, so lambda stays positive.
            #
            # Normalised PER SAMPLE, by that sample's own clean margin. The
            # batch maximum would throttle every sample's dual rate by the
            # single largest-margin sample present, making one sample's
            # reported norm a function of its batch-mates and of the shuffle
            # -- a robustness number that moves when the dataloader does.
            scale = (margin / clean_margin).clamp(-1.0, 1.0)
            lam = (lam * torch.exp(self.dual_lr * scale)).clamp(1e-4, 1e4)

            # Keep the smallest-norm adversarial iterate seen so far.
            with torch.no_grad():
                _, _, cur_preds, _ = model(self._clamp(inputs + delta))
            adversarial = cur_preds.detach() != preds
            cur_norm = self._norm_of(delta)
            improved = adversarial & (cur_norm < best_norm)
            best = torch.where(improved.view(expand), delta, best)
            best_norm = torch.where(improved, cur_norm, best_norm)

        found = torch.isfinite(best_norm)
        candidate = self._clamp(self._project(inputs + best, inputs))
        # Samples never driven across the boundary are returned clean.
        candidate = torch.where(found.view(expand), candidate, inputs)

        # Success measured after the eps cap, and every failure returned
        # clean, so a failed sample's reported norm is 0 rather than the cap.
        x_adv, success, final_logits = finalise_minimum_norm(
            model, inputs, candidate, preds, clean_logits
        )
        delta = (x_adv - inputs).flatten(1)

        model.train(was_training)
        return AttackResult(
            perturbed=x_adv,
            effective_eps=delta.abs().amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=final_logits,
            # Only when L2 is what was minimised; in L_inf mode the answer is
            # `effective_eps` and `minimised` returns it.
            l2_norm=delta.norm(dim=1).detach() if self.norm == "l2" else None,
            success=success,
            minimised_norm=self.norm,
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
