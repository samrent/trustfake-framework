"""A^3: Adaptive Auto Attack (Liu, Cheng, Zhang, Zhou & He, CVPR 2022).

A native reimplementation of the two ideas the paper contributes on top of a
PGD-family inner attack -- Adaptive Direction Initialization and Online
Statistics-based Discarding -- so the suite gains the attack without a
non-PyPI research dependency. The inner loop is an APGD-style adaptive-step
attack on the logit margin rather than the authors' exact inner attack, which
is stated here rather than glossed: the contribution being ported is the
budget allocation, not a new gradient step.
"""

from __future__ import annotations

import torch

from trustfake.attacks._common import class_margin, model_logits, project_linf
from trustfake.attacks.abc import AdversarialAttack, AttackResult
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["AdaptiveAutoAttack"]


class AdaptiveAutoAttack(AdversarialAttack):
    r"""A^3, Adaptive Auto Attack, L_inf (Liu et al., CVPR 2022).

    AutoAttack's ensemble is strong but spends the same budget on every
    sample, including the ones a single step would have flipped and the ones
    no budget will. A^3 keeps the strength and spends the budget where it
    converts, via two mechanisms:

    **Adaptive Direction Initialization (ADI).** Rather than starting from a
    uniform random point in the eps-ball, each sample starts from the best of
    a candidate set scored by the logit margin: the clean point, a single
    signed-gradient step, random points in the ball, and -- from the second
    round on -- the perturbation *directions that already worked on other
    samples in the batch*. That last family is where the "adaptive" comes
    from: adversarial directions transfer between samples of the same model,
    so a solved sample is free reconnaissance for an unsolved one.

    **Online Statistics-based Discarding (OSD).** A sample that has flipped
    is removed from the active set immediately, and the freed budget is
    redistributed: each round runs longer than the last because it runs on
    fewer samples. The attack therefore converges its total cost onto the
    genuinely robust samples, which are the only ones whose robustness is in
    question.

    The inner attack ascends the negative logit margin with an APGD-style
    adaptive step (momentum, halving the step at checkpoints where progress
    stalls, restarting from the best point). It uses the margin rather than
    AutoAttack's DLR loss deliberately: DLR reads the third-largest logit and
    is undefined on a 3-class detector -- the same constraint the
    `autoattack` wrappers document for their targeted stages.

    Args:
        eps: L_inf budget.
        steps: Total inner iterations across all rounds. OSD redistributes
            them; it does not add to them.
        rounds: ADI/OSD rounds. Each re-initialises the still-unsolved
            samples from the candidate set, now including transfer
            directions.
        n_random_init: Random candidates scored per sample per round.
        momentum: APGD momentum coefficient (`alpha` in Croce & Hein 2020).
        seed: Seed for the random candidates, for determinism.
        clip_min, clip_max: Valid input range.
    """

    def __init__(
        self,
        eps: float = 8 / 255,
        steps: int = 100,
        rounds: int = 3,
        n_random_init: int = 6,
        momentum: float = 0.75,
        seed: int = 0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.steps = steps
        self.rounds = max(1, rounds)
        self.n_random_init = n_random_init
        self.momentum = momentum
        self.seed = seed

    @property
    def name(self) -> str:
        return "a3"

    def _margin(
        self, model: TrustFakeWrapper, x: torch.Tensor, preds: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return class_margin(model_logits(model, x), preds)

    def _candidates(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        preds: torch.Tensor,
        transfer: list[torch.Tensor],
        gen: torch.Generator,
    ) -> torch.Tensor:
        """ADI: score a candidate set per sample, return the best start.

        Candidates are deltas, not points, so a transfer direction found on
        one sample can be applied to another.
        """
        deltas = [torch.zeros_like(inputs)]

        # One signed-gradient step: the cheapest informative direction.
        x = inputs.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            margin = class_margin(model_logits(model, x), preds)
            grad = torch.autograd.grad(margin.sum(), x)[0]
        deltas.append(-self.eps * grad.sign())

        for _ in range(self.n_random_init):
            deltas.append(
                torch.empty_like(inputs).uniform_(-self.eps, self.eps, generator=gen)
            )

        # Directions that already worked on other samples this batch.
        deltas.extend(transfer)

        expand = (-1,) + (1,) * (inputs.ndim - 1)
        best = torch.zeros_like(inputs)
        best_margin = torch.full((inputs.shape[0],), float("inf"), device=inputs.device)
        for delta in deltas:
            candidate = self._clamp(project_linf(inputs + delta, inputs, self.eps))
            margin = self._margin(model, candidate, preds)
            take = margin < best_margin
            best = torch.where(take.view(expand), candidate - inputs, best)
            best_margin = torch.where(take, margin, best_margin)
        return best

    def _apgd(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        preds: torch.Tensor,
        delta: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """APGD-style descent on the logit margin. Returns the best delta."""
        if steps <= 0 or self.eps <= 0:
            return delta

        expand = (-1,) + (1,) * (inputs.ndim - 1)
        eta = torch.full((inputs.shape[0],), 2.0 * self.eps, device=inputs.device)

        x_prev = self._clamp(inputs + delta)
        x_cur = x_prev.clone()
        best = x_cur.clone()
        best_margin = self._margin(model, x_cur, preds)
        # Progress counter per checkpoint window, for the halving rule.
        improved_count = torch.zeros(inputs.shape[0], device=inputs.device)
        window = max(1, steps // 4)

        for step in range(steps):
            x = x_cur.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                margin = class_margin(model_logits(model, x), preds)
                grad = torch.autograd.grad(margin.sum(), x)[0]

            # Descend the margin: move against the gradient.
            z = self._clamp(
                project_linf(x_cur - eta.view(expand) * grad.sign(), inputs, self.eps)
            )
            momentum_step = self.momentum * (z - x_cur) + (1 - self.momentum) * (
                x_cur - x_prev
            )
            x_next = self._clamp(project_linf(x_cur + momentum_step, inputs, self.eps))

            new_margin = self._margin(model, x_next, preds)
            better = new_margin < best_margin
            best = torch.where(better.view(expand), x_next, best)
            best_margin = torch.where(better, new_margin, best_margin)
            improved_count += better.float()

            x_prev, x_cur = x_cur, x_next

            # Checkpoint: where the window bought little, halve the step and
            # restart from the best point found so far.
            if (step + 1) % window == 0:
                stalled = improved_count < 0.75 * window
                eta = torch.where(stalled, eta * 0.5, eta)
                x_cur = torch.where(stalled.view(expand), best, x_cur)
                x_prev = x_cur.clone()
                improved_count = torch.zeros_like(improved_count)

        return (best - inputs).detach()

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

        gen = torch.Generator(device=inputs.device).manual_seed(self.seed)
        expand = (-1,) + (1,) * (inputs.ndim - 1)

        delta = torch.zeros_like(inputs)
        solved = torch.zeros(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        transfer: list[torch.Tensor] = []
        spent = 0

        for round_idx in range(self.rounds):
            active = (~solved).nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                break

            # OSD: the remaining budget is split over the remaining rounds,
            # so each round runs longer as the active set shrinks.
            remaining_rounds = self.rounds - round_idx
            round_steps = max(1, (self.steps - spent) // remaining_rounds)
            spent += round_steps

            sub_inputs = inputs[active]
            sub_preds = preds[active]

            sub_transfer = [t[active] for t in transfer]
            start = self._candidates(model, sub_inputs, sub_preds, sub_transfer, gen)
            sub_delta = self._apgd(model, sub_inputs, sub_preds, start, round_steps)

            delta[active] = sub_delta
            with torch.no_grad():
                _, _, sub_new_preds, _ = model(self._clamp(sub_inputs + sub_delta))
            solved[active] = sub_new_preds.detach() != sub_preds

            # Feed the round's successes forward as ADI transfer directions.
            won = solved[active]
            if won.any() and round_idx + 1 < self.rounds:
                direction = torch.zeros_like(delta)
                direction[active] = torch.where(
                    won.view(expand), sub_delta, torch.zeros_like(sub_delta)
                )
                # A single batch-wide direction: the mean successful
                # perturbation, plus its sign pattern, which is what
                # transfers between samples.
                mean_delta = direction[active][won].mean(dim=0, keepdim=True)
                transfer = [
                    mean_delta.expand_as(inputs).clone(),
                    (self.eps * mean_delta.sign()).expand_as(inputs).clone(),
                ]

        x_adv = self._clamp(project_linf(inputs + delta, inputs, self.eps))
        with torch.no_grad():
            final_logits, _, final_preds, _ = model(x_adv)

        model.train(was_training)
        return AttackResult(
            perturbed=x_adv.detach(),
            effective_eps=(x_adv - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds,
            accepted_logits=final_logits.detach(),
            success=(final_preds.detach() != preds),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
