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


def finalise_minimum_norm(
    model: TrustFakeWrapper,
    inputs: torch.Tensor,
    x_adv: torch.Tensor,
    preds: torch.Tensor,
    clean_logits: torch.Tensor,
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

    Args:
        model: The model under attack, already in eval mode.
        inputs: Clean inputs.
        x_adv: The attack's candidate, already capped to `eps`.
        preds: The model's clean predictions.
        clean_logits: Logits on `inputs`, reported back for failed samples.

    Returns:
        ``(perturbed, success, logits)`` -- the batch to report, the
        per-sample success flag, and the logits of the forward that decided
        it (clean logits where the attack failed).
    """
    with torch.no_grad():
        logits, _, adv_preds, _ = model(x_adv)
    success = adv_preds.detach() != preds

    expand = (-1,) + (1,) * (inputs.ndim - 1)
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
