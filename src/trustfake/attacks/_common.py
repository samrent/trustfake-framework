"""Shared helpers for attack implementations."""

from __future__ import annotations

import torch
import torch.nn as nn

from trustfake.models.wrapper import TrustFakeWrapper

__all__ = [
    "model_logits",
    "LogitsAdapter",
    "project_linf",
    "project_l2",
    "project_l1_ball",
    "class_margin",
    "l2_witness_search",
    "shrink_towards",
    "finalise_minimum_norm",
]


def model_logits(model: TrustFakeWrapper, x: torch.Tensor) -> torch.Tensor:
    """Logits only from a TrustFakeWrapper's 4-tuple forward."""
    return model(x)[0]


class LogitsAdapter(nn.Module):
    """Present a TrustFakeWrapper as a plain ``x -> logits`` module.

    Third-party attack packages (e.g. AutoAttack) call ``model(x)`` and
    expect logits; the framework wrapper returns
    ``(logits, probs, preds, uncertainty)``. This adapter bridges the two
    and keeps normalization inside the forward, where the wrapper puts it.
    """

    def __init__(self, wrapper: TrustFakeWrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapper(x)[0]


def project_linf(
    perturbed: torch.Tensor, inputs: torch.Tensor, eps: float
) -> torch.Tensor:
    """Project `perturbed` into the L_inf eps-ball around `inputs`."""
    return inputs + (perturbed - inputs).clamp(-eps, eps)


def project_l2(
    perturbed: torch.Tensor, inputs: torch.Tensor, eps: float
) -> torch.Tensor:
    """Project `perturbed` into the L2 eps-ball around `inputs`, per sample.

    Because ``||v||_inf <= ||v||_2``, an L2 projection to radius eps also
    satisfies an L_inf <= eps budget -- which is why L2 min-norm attacks
    (DeepFool, C&W) fit the framework's L_inf attack contract when capped
    this way.
    """
    delta = perturbed - inputs
    flat = delta.flatten(1)
    norm = flat.norm(dim=1).clamp_min(1e-12)
    factor = (eps / norm).clamp(max=1.0)
    return inputs + (flat * factor[:, None]).view_as(delta)


def project_l1_ball(
    v: torch.Tensor, radius: float | torch.Tensor, iterations: int = 40
) -> torch.Tensor:
    """Project each sample of `v` onto the L1 ball of the given radius.

    Needed by the minimum-norm proximal attacks: the proximal operator of the
    L_inf norm is, by Moreau decomposition,
    ``prox_{t||.||_inf}(v) = v - proj_{||u||_1 <= t}(v)``.

    The projection is a soft-threshold ``sign(v) * relu(|v| - theta)`` for the
    unique ``theta >= 0`` with ``sum(relu(|v| - theta)) = radius``. That theta
    is usually found by sorting and taking a cumulative sum; here it is found
    by bisection instead, because **the sort-and-cumsum route cannot run on
    GPU in this harness**: `cumsum` has no deterministic CUDA kernel, and the
    evaluation pipeline sets ``torch.use_deterministic_algorithms(True)``, so
    the sorted version raises rather than silently returning a
    non-reproducible attack. Bisection uses only sums and maxima, which are
    deterministic on every backend, and on a full-resolution image it is also
    the cheaper of the two.

    The residual is monotonically decreasing in theta, so bisection converges
    linearly; 40 halvings take the bracket far below float32 resolution.

    Args:
        v: Batch of tensors, projected per sample over all non-batch dims.
        radius: Scalar radius, or a per-sample tensor of shape (B,).
        iterations: Bisection steps used to solve for the threshold.
    """
    flat = v.flatten(1)
    batch = flat.shape[0]
    if not isinstance(radius, torch.Tensor):
        radius = torch.full((batch,), float(radius), device=v.device, dtype=v.dtype)
    radius = radius.to(v.dtype).clamp_min(0.0)

    absv = flat.abs()
    outside = absv.sum(dim=1) > radius

    # theta = 0 leaves the L1 norm unchanged (residual > 0 when outside);
    # theta = max|v| zeroes everything (residual = -radius < 0). The root is
    # bracketed by construction.
    lo = torch.zeros(batch, device=v.device, dtype=v.dtype)
    hi = absv.amax(dim=1)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        residual = (absv - mid[:, None]).clamp_min(0.0).sum(dim=1) - radius
        too_small = residual > 0  # threshold not yet large enough
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)

    theta = (lo + hi) / 2
    projected = flat.sign() * (absv - theta[:, None]).clamp_min(0.0)

    return torch.where(outside[:, None], projected, flat).view_as(v)


def l2_witness_search(
    model: TrustFakeWrapper,
    inputs: torch.Tensor,
    preds: torch.Tensor,
    eps: float,
    steps: int = 50,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Is ANY adversarial example reachable inside ``||delta||_2 <= eps``?

    A fixed-budget attack that flips a sample inside a radius is a *witness*
    that the true minimum norm is at most that radius. So a minimum-norm
    attack capped at the same radius that reports failure on that sample is
    not measuring the model -- it is reporting its own search having stalled,
    as robustness the model does not have. That is the one direction an
    attack is never allowed to be wrong in, because it is invisible: a
    stalled search and a genuinely robust sample produce the same row.

    This closes it by construction. Whatever the norm-minimising walk left
    behind, every sample it failed on gets one more chance from a plain
    projected descent on the class margin inside the ball -- so the attack
    is at least as strong as a fixed-budget attack at its own cap, by
    definition rather than by luck. Two restarts (the clean point and a
    random point in the ball), because a curved boundary can leave a single
    descent short of a flip that a different start finds immediately.

    Returns:
        ``(perturbed, found)`` -- the first adversarial iterate located for
        each sample, and a mask of which samples were solved. Unsolved
        samples come back clean.
    """
    found = torch.zeros(preds.shape[0], dtype=torch.bool, device=inputs.device)
    best = inputs.clone()
    if eps <= 0 or steps <= 0:
        return best, found

    expand = (-1,) + (1,) * (inputs.ndim - 1)
    alpha = 2.5 * eps / steps
    generator = torch.Generator(device=inputs.device).manual_seed(seed)

    for restart in range(2):
        if restart == 0:
            x = inputs.clone()
        else:
            noise = torch.empty_like(inputs).normal_(0.0, 1.0, generator=generator)
            flat = noise.flatten(1)
            direction = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            radius = (
                torch.rand(
                    inputs.shape[0], 1, device=inputs.device, generator=generator
                )
                * eps
            )
            start = inputs + (direction * radius).view_as(inputs)
            x = start.clamp(clip_min, clip_max)

        for _ in range(steps):
            x = x.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                margin = class_margin(model_logits(model, x), preds)
                grad = torch.autograd.grad(margin.sum(), x)[0]
            # DESCEND the margin -- drive it below zero -- along the unit L2
            # direction, so the step is a distance in input space.
            flat_grad = grad.flatten(1)
            unit = flat_grad / flat_grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            x = x.detach() - alpha * unit.view_as(grad)
            x = project_l2(x, inputs, eps).clamp(clip_min, clip_max)

            with torch.no_grad():
                flipped = model(x)[2].detach() != preds
            take = flipped & ~found
            best = torch.where(take.view(expand), x, best)
            found = found | flipped
            if bool(found.all()):
                return best.detach(), found

    return best.detach(), found


def shrink_towards(
    model: TrustFakeWrapper,
    inputs: torch.Tensor,
    x_adv: torch.Tensor,
    preds: torch.Tensor,
    steps: int = 12,
) -> torch.Tensor:
    """Bisect along ``[inputs, x_adv]`` for the smallest still-adversarial point.

    A witness search answers "is the cap reachable", not "how little does it
    take" -- left alone it would report a norm at the cap, which is a worse
    minimum-norm estimate than the attack could give. Bisection costs a
    handful of forwards and recovers most of the difference. `hi` is kept
    adversarial throughout, so the returned point always is.
    """
    expand = (-1,) + (1,) * (inputs.ndim - 1)
    lo = torch.zeros(preds.shape[0], device=inputs.device, dtype=inputs.dtype)
    hi = torch.ones_like(lo)
    for _ in range(steps):
        mid = (lo + hi) / 2
        probe = inputs + mid.view(expand) * (x_adv - inputs)
        with torch.no_grad():
            adversarial = model(probe)[2].detach() != preds
        hi = torch.where(adversarial, mid, hi)
        lo = torch.where(adversarial, lo, mid)
    return (inputs + hi.view(expand) * (x_adv - inputs)).detach()


def finalise_minimum_norm(
    model: TrustFakeWrapper,
    inputs: torch.Tensor,
    x_adv: torch.Tensor,
    preds: torch.Tensor,
    clean_logits: torch.Tensor,
    *,
    eps: float | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    witness_steps: int = 50,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Final accounting shared by every minimum-norm attack.

    Success is measured on the *capped* candidate: a flip that only survives
    outside the reported eps is not a success inside it. Every sample that
    fails that test is then returned CLEAN, so its reported norm is exactly
    0 rather than whatever the search happened to leave behind.

    That last part is the point. Without it, an attack that walks the sample
    somewhere and then gets capped reports ``l2_norm == eps`` on a failure,
    while an attack that gives up reports 0 -- the same outcome, opposite
    numbers, and a curve that mixes them is measuring which implementation
    it ran rather than which model. Making the failure value a constant 0
    across all four attacks makes `success` the only thing that separates
    "unbreakable" from "broken for free", which is what it is for.

    Before that accounting, every sample the attack failed on is retried by
    `l2_witness_search` inside the same cap, and any rescue is shrunk back
    toward the clean input. That makes "at least as strong as a fixed-budget
    attack at the same radius" a structural property of every minimum-norm
    attack in the suite rather than something each one has to get right on
    its own -- and it is the property whose absence silently overstates
    robustness.

    Args:
        model: The model under attack, already in eval mode.
        inputs: Clean inputs.
        x_adv: The attack's candidate, already capped to `eps`.
        preds: The model's clean predictions.
        clean_logits: Logits on `inputs`, reported back for failed samples.
        eps: The attack's L2 cap. None disables the witness retry.
        clip_min, clip_max: Valid input range for the retry.
        witness_steps: Descent steps per restart in the retry. 0 disables it.
        seed: Seed for the retry's random restart, for determinism.

    Returns:
        ``(perturbed, success, logits)`` -- the batch to report, the
        per-sample success flag, and the logits of the forward that decided
        it (clean logits where the attack failed).
    """
    with torch.no_grad():
        logits, _, adv_preds, _ = model(x_adv)
    success = adv_preds.detach() != preds
    expand = (-1,) + (1,) * (inputs.ndim - 1)

    # Anything the norm-minimising walk failed on gets one more chance inside
    # the same cap. Without it the attack reports the model as robust exactly
    # where its own search stalled -- see `l2_witness_search`.
    if eps is not None and witness_steps > 0 and not bool(success.all()):
        rescued, found = l2_witness_search(
            model, inputs, preds, eps, witness_steps, clip_min, clip_max, seed
        )
        newly = found & ~success
        if bool(newly.any()):
            # Shrink back toward the clean input: the witness proves the cap
            # is reachable, bisection recovers how little it actually took.
            rescued = shrink_towards(model, inputs, rescued, preds)
            x_adv = torch.where(newly.view(expand), rescued, x_adv)
            with torch.no_grad():
                logits, _, adv_preds, _ = model(x_adv)
            success = adv_preds.detach() != preds

    perturbed = torch.where(success.view(expand), x_adv, inputs).detach()
    reported = torch.where(success[:, None], logits.detach(), clean_logits.detach())
    return perturbed, success, reported


def class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """``z_label - max_{c != label} z_c``: the boundary function.

    Negative exactly when the sample is misclassified relative to `labels`,
    zero on the decision boundary. Minimum-norm attacks drive it to (just
    below) zero rather than maximising a loss, which is why they need it as a
    signed quantity rather than as a cross-entropy.
    """
    target = logits.gather(1, labels[:, None]).squeeze(1)
    others = logits.scatter(1, labels[:, None], float("-inf"))
    return target - others.amax(dim=1)
