"""Shared helpers for attack implementations."""

from __future__ import annotations

import torch
import torch.nn as nn

from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["model_logits", "LogitsAdapter", "project_linf", "project_l2"]


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
