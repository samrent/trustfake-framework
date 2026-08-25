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

    The best (smallest-norm) adversarial iterate is kept per sample, so the
    result never degrades over iterations. Samples never driven across the
    boundary are returned unperturbed and flagged in
    ``AttackResult.success``: a min-norm attack reports failure rather than
    returning a perturbation it could not make work.

    ``eps`` does not drive the search -- it caps the result, in the norm
    given by ``norm``. With ``norm="linf"`` that cap is the framework's L_inf
    contract directly; with ``norm="l2"`` it is an L2 radius, which implies
    the same L_inf bound.

    Deterministic: no random start.

    Args:
        eps: Radius the final perturbation is capped to, in `norm`.
        steps: Primal-dual iterations.
        norm: ``"linf"`` or ``"l2"`` -- the norm being minimised.
        lr: Initial primal step size, cosine-annealed to ``lr * lr_final``.
        lr_final: Fraction of `lr` the schedule ends at.
        dual_lr: Multiplicative dual ascent rate.
        lam_init: Initial dual variable.
        prox_weight: Weight of the norm term relative to the margin term;
            the proximal threshold each step is ``lr * prox_weight``.
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
        prox_weight: float = 0.05,
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

    def _prox(self, delta: torch.Tensor, threshold: float) -> torch.Tensor:
        """Proximal operator of `threshold * ||.||_p`, per sample."""
        if threshold <= 0:
            return delta
        if self.norm == "linf":
            # Moreau: prox_{t||.||_inf}(v) = v - proj_{||u||_1 <= t}(v).
            return delta - project_l1_ball(delta, threshold)
        # Block soft-threshold: prox_{t||.||_2}(v) = v * max(0, 1 - t/||v||).
        flat = delta.flatten(1)
        norm = flat.norm(dim=1).clamp_min(1e-12)
        scale = (1.0 - threshold / norm).clamp_min(0.0)
        return (flat * scale[:, None]).view_as(delta)

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
            _, _, preds, _ = model(inputs)
        preds = preds.detach()

        batch = inputs.shape[0]
        expand = (-1,) + (1,) * (inputs.ndim - 1)
        delta = torch.zeros_like(inputs)
        lam = torch.full((batch,), float(self.lam_init), device=inputs.device)

        best = torch.zeros_like(inputs)
        best_norm = torch.full((batch,), float("inf"), device=inputs.device)

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
            # the norm (this is the step that shrinks delta).
            d = delta - lr * lam.view(expand) * grad
            d = self._prox(d, lr * self.prox_weight)
            # Stay inside the box in *perturbation* space, so the prox and
            # the clip cannot fight each other across iterations.
            delta = (self._clamp(inputs + d) - inputs).detach()

            # Dual: raise the pressure where the constraint is still violated
            # (margin > 0, i.e. not yet adversarial), release it where it is
            # satisfied. Multiplicative, so lambda stays positive.
            scale = margin / margin.abs().amax().clamp_min(1e-12)
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
        x_adv = self._clamp(self._project(inputs + best, inputs))
        # Samples never driven across the boundary are returned clean.
        x_adv = torch.where(found.view(expand), x_adv, inputs)

        with torch.no_grad():
            final_logits, _, final_preds, _ = model(x_adv)

        model.train(was_training)
        return AttackResult(
            perturbed=x_adv.detach(),
            effective_eps=(x_adv - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=final_logits.detach(),
            l2_norm=(x_adv - inputs).flatten(1).norm(dim=1).detach(),
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
