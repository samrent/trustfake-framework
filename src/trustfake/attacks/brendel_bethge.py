"""Brendel & Bethge minimum-norm attack (NeurIPS 2019).

A native reimplementation, so the suite carries no non-PyPI research
dependency. The reference implementation (and foolbox's port of it) solves the
per-step trust-region problem with a specialised numba solver that folds the
box constraint into the QP; here the same problem is solved in closed form
without the box, and the box is enforced by clipping the result.

That difference is why the trust-region radius has to adapt. The closed form
satisfies the linearised boundary constraint exactly, but clipping afterwards
does not preserve it: on a sample sitting against the edge of the valid pixel
range the clip can push the point back across the boundary, and the step is
rejected. With a fixed radius the attack then retries the identical rejected
step every iteration and stalls at its starting point -- while still passing
every generic contract test and still reporting a perturbation norm, which
would simply be the norm of a random adversarial image. Shrinking the radius
on rejection, and letting it recover on acceptance, makes the step small
enough that the clip stops binding. `tests/attacks/test_min_norm_attacks.py`
pins the resulting norms against DeepFool so a regression here shows up as a
number rather than as a silently weaker attack.
"""

from __future__ import annotations

import torch

from trustfake.attacks._common import class_margin, model_logits, project_l2
from trustfake.attacks.abc import AdversarialAttack, AttackResult
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["BrendelBethge"]


class BrendelBethge(AdversarialAttack):
    r"""Brendel & Bethge attack, L2 (Brendel, Rauber, Kümmerer, Ustyuzhaninov
    & Bethge, NeurIPS 2019).

    A *minimum-norm* attack that runs backwards from the usual direction: it
    starts from a point that is already adversarial and walks it back toward
    the clean input while staying on the adversarial side of the decision
    boundary. That makes it a robustness *measurement* rather than a
    fixed-budget attack -- it answers "how little does it take?", which is
    exactly the quantity a fixed-eps attack cannot report.

    Each step solves, in closed form,

    .. math::
        \min_\delta \|(x_k + \delta) - x_0\|_2^2
        \quad\text{s.t.}\quad g^\top\delta = c,\ \ \|\delta\|_2 \le r_k

    where :math:`g = \nabla_x b(x_k)` is the gradient of the boundary function
    :math:`b(x) = z_y(x) - \max_{c \neq y} z_c(x)` (negative iff adversarial)
    and :math:`c` is the linearised offset that lands the step on the boundary
    with a small margin to spare. The solution is the projection of the
    straight-line direction :math:`x_0 - x_k` onto that hyperplane, pulled
    back into the trust region along the component that is free to shrink:
    the minimum-norm point satisfying the constraint is orthogonal to the
    residual, so the radius split is exact, not a line search.

    Like the other minimum-norm attacks in the suite (DeepFool, C&W), ``eps``
    is not what drives the search: it is an L2 cap applied to the result, and
    because :math:`\|v\|_\infty \le \|v\|_2` that cap also satisfies the
    framework's L_inf contract. Report that ``eps`` is an L2 radius here.
    ``AttackResult.effective_eps`` is the L_inf actually applied, as
    everywhere else in the suite; ``l2_norm`` on the result carries the
    quantity the attack actually minimises.

    Deterministic: the random starting points come from a per-call seeded
    generator.

    Args:
        eps: L2 radius the final perturbation is capped to.
        steps: Boundary-following iterations. Under-budgeting this does not
            fail loudly -- it returns the norm the walk happened to reach,
            which reads as a robust model. Check convergence by re-running
            with more steps and confirming the reported norm stops moving.
        init_candidates: Random starting points tried per sample, on top of
            the deterministic ones (another sample in the batch, and the two
            constant corner images).
        init_search_steps: Bisection steps used to move a starting point down
            to the boundary before following it.
        lr: Initial trust-region radius, as a fraction of the current
            distance to the clean input. Smaller is slower and tracks the
            boundary more closely.
        lr_shrink: Radius multiplier applied when a step is rejected.
        lr_grow: Radius multiplier applied when a step is accepted, capped
            back at `lr`.
        margin: Boundary offset kept in logit units, so a step lands just
            inside the adversarial region rather than exactly on it.
        clip_min, clip_max: Valid input range.
        seed: Seed for the random starting points.
    """

    norm = "l2"
    minimum_norm = True

    def __init__(
        self,
        eps: float = 0.5,
        steps: int = 100,
        init_candidates: int = 8,
        init_search_steps: int = 10,
        lr: float = 0.2,
        lr_shrink: float = 0.5,
        lr_grow: float = 1.3,
        margin: float = 1e-3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.steps = steps
        self.init_candidates = init_candidates
        self.init_search_steps = init_search_steps
        self.lr = lr
        self.lr_shrink = lr_shrink
        self.lr_grow = lr_grow
        self.margin = margin
        self.seed = seed

    @property
    def name(self) -> str:
        return "bb"

    def _starting_points(
        self, model: TrustFakeWrapper, inputs: torch.Tensor, preds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find an adversarial starting point per sample.

        Tries, in order of how close they tend to land: another sample from
        the batch, the two constant corner images, then uniform random
        images. Each candidate that flips the prediction is bisected down
        toward the clean input, so the walk starts near the boundary rather
        than out in the corner. Returns the starting points and a mask of the
        samples for which one was found.
        """
        gen = torch.Generator(device=inputs.device).manual_seed(self.seed)

        candidates = [
            torch.roll(inputs, shifts=1, dims=0),
            torch.full_like(inputs, self.clip_min),
            torch.full_like(inputs, self.clip_max),
        ]
        for _ in range(self.init_candidates):
            noise = torch.empty_like(inputs).uniform_(
                self.clip_min, self.clip_max, generator=gen
            )
            candidates.append(noise)

        start = inputs.clone()
        found = torch.zeros(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        best_dist = torch.full((inputs.shape[0],), float("inf"), device=inputs.device)

        for candidate in candidates:
            with torch.no_grad():
                flipped = model(candidate)[2].detach() != preds
            if not flipped.any():
                continue

            # Bisect along [x_0, candidate] for the smallest step that is
            # still adversarial: `hi` is always adversarial, `lo` never is.
            lo = torch.zeros(inputs.shape[0], device=inputs.device)
            hi = torch.ones(inputs.shape[0], device=inputs.device)
            expand = (-1,) + (1,) * (inputs.ndim - 1)
            for _ in range(self.init_search_steps):
                mid = (lo + hi) / 2
                probe = inputs + mid.view(expand) * (candidate - inputs)
                with torch.no_grad():
                    adv = model(probe)[2].detach() != preds
                hi = torch.where(adv, mid, hi)
                lo = torch.where(adv, lo, mid)
            bisected = inputs + hi.view(expand) * (candidate - inputs)

            dist = (bisected - inputs).flatten(1).norm(dim=1)
            take = flipped & (dist < best_dist)
            start = torch.where(take.view(expand), bisected, start)
            best_dist = torch.where(take, dist, best_dist)
            found = found | flipped

        return self._clamp(start), found

    @staticmethod
    def _trust_region_step(
        to_clean: torch.Tensor,
        grad: torch.Tensor,
        offset: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form solution of the per-step trust-region problem.

        Minimises ``||to_clean - delta||^2`` over ``{delta : grad . delta =
        offset, ||delta|| <= radius}``. The minimum-norm point meeting the
        equality is ``d_min = offset * grad / ||grad||^2``; the remaining
        freedom lies in the null space of ``grad`` and is therefore orthogonal
        to it, so the radius splits exactly by Pythagoras and no line search
        is needed. When even ``d_min`` overshoots the radius the constraint is
        unreachable this step and the step degenerates to the largest move
        toward it that fits.
        """
        g = grad.flatten(1)
        d = to_clean.flatten(1)
        g_sq = g.pow(2).sum(dim=1).clamp_min(1e-12)

        # Minimum-norm point on the constraint hyperplane.
        d_min = (offset / g_sq)[:, None] * g
        # Unconstrained-by-radius optimum: project `d` onto the hyperplane.
        d_star = d + ((offset - (g * d).sum(dim=1)) / g_sq)[:, None] * g
        residual = d_star - d_min  # lies in null(g), orthogonal to d_min

        min_norm = d_min.norm(dim=1)
        res_norm = residual.norm(dim=1).clamp_min(1e-12)
        room = (radius.pow(2) - min_norm.pow(2)).clamp_min(0.0).sqrt()
        t = (room / res_norm).clamp(max=1.0)
        delta = d_min + t[:, None] * residual

        # Constraint unreachable inside the trust region: move as far along
        # the constraint gradient as the radius allows.
        unreachable = min_norm > radius
        fallback = (radius / min_norm.clamp_min(1e-12))[:, None] * d_min
        delta = torch.where(unreachable[:, None], fallback, delta)
        return delta.view_as(to_clean)

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

        x_adv, found = self._starting_points(model, inputs, preds)
        # Samples with no adversarial start stay clean: an attack that cannot
        # reach the other side reports that, it does not fake a perturbation.
        x_adv = torch.where(found.view((-1,) + (1,) * (inputs.ndim - 1)), x_adv, inputs)

        # Per-sample trust-region radius, as a fraction of the remaining
        # distance. Adapted below; see the module docstring for why a fixed
        # one deadlocks against the box constraint.
        lr = torch.full((inputs.shape[0],), float(self.lr), device=inputs.device)

        for _ in range(self.steps):
            if not found.any():
                break
            x_adv = x_adv.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                boundary = class_margin(model_logits(model, x_adv), preds)
                grad = torch.autograd.grad(boundary.sum(), x_adv)[0]
            boundary = boundary.detach()
            x_cur = x_adv.detach()

            to_clean = inputs - x_cur
            radius = lr * to_clean.flatten(1).norm(dim=1)
            # Land on the boundary with `margin` of logit to spare, on the
            # adversarial side.
            offset = -(boundary + self.margin)

            step = self._trust_region_step(to_clean, grad, offset, radius)
            candidate = self._clamp(x_cur + step)

            with torch.no_grad():
                still_adv = model(candidate)[2].detach() != preds
            closer = (candidate - inputs).flatten(1).norm(dim=1) < to_clean.flatten(
                1
            ).norm(dim=1)
            accepted = still_adv & closer & found
            x_adv = torch.where(
                accepted.view((-1,) + (1,) * (inputs.ndim - 1)), candidate, x_cur
            )

            # Shrink where the linearisation (or the clip) did not hold, and
            # let a working radius recover toward its initial value.
            lr = torch.where(
                accepted,
                (lr * self.lr_grow).clamp(max=self.lr),
                lr * self.lr_shrink,
            )

        x_adv = self._clamp(project_l2(x_adv.detach(), inputs, self.eps))
        with torch.no_grad():
            final_logits, _, final_preds, _ = model(x_adv)

        model.train(was_training)
        return AttackResult(
            perturbed=x_adv.detach(),
            effective_eps=(x_adv - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=final_logits.detach(),
            l2_norm=(x_adv - inputs).flatten(1).norm(dim=1).detach(),
            # After the eps cap, not before: a perturbation that only flips
            # the label outside the reported budget has not succeeded within
            # it, and reporting otherwise would overstate the attack.
            success=(final_preds.detach() != preds),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
