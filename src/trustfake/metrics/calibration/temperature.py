"""Temperature scaling: the calibration baseline the harness benchmarks against.

A single scalar T > 0 divides the logits before the softmax. It cannot change
the argmax (T > 0 is monotone per logit), so accuracy is untouched; it only
rescales confidence. T is fitted by minimising NLL on a held-out calibration
split and then frozen -- fitting it on the reported split is the classic
self-own, and the shard-disjoint manifest makes it structurally impossible
here (see trustfake.data.manifest).

Note this is a genuinely 3-class calibration: unlike a 2-class model, where
MSP is monotone in the single logit margin and temperature cannot reorder
samples, with three classes temperature CAN change the confidence ranking, so
it is a live variable -- exactly what makes it a meaningful baseline for a
selective-classification method.
"""

from __future__ import annotations

import torch
from torch.nn.functional import cross_entropy

from trustfake.logging import get_logger

logger = get_logger("calibration")

__all__ = ["fit_temperature", "calibrate_temperature"]


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bounds: tuple[float, float] = (1e-2, 1e2),
    max_iter: int = 100,
) -> float:
    """
    Fit the temperature that minimises multiclass NLL of ``softmax(logits / T)``
    against ``targets``. Optimised over ``log T`` (unconstrained) with LBFGS,
    then clamped to ``bounds``.

    Args:
        logits: Raw logits, shape (N, C).
        targets: Integer class labels, shape (N,).
        bounds: (min, max) clamp on the returned temperature.
        max_iter: LBFGS iterations.

    Returns:
        The fitted temperature as a float.
    """
    logits = logits.detach()
    targets = targets.detach().long()
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)  # T = exp(0) = 1
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().clamp(*bounds).item())
    logger.info(f"Fitted temperature T = {temperature:.4f}")
    return temperature


@torch.no_grad()
def calibrate_temperature(
    wrapper: torch.nn.Module,
    calib_loader,
    device: torch.device | str = "cpu",
    bounds: tuple[float, float] = (1e-2, 1e2),
) -> float:
    """
    Collect raw logits and labels over ``calib_loader`` using ``wrapper`` and
    fit a temperature on them. The wrapper's own temperature is temporarily
    reset to 1.0 for the pass so fitting sees raw logits regardless of any
    previously set value.

    Args:
        wrapper: A TrustFakeWrapper; its forward returns (logits, ...).
        calib_loader: DataLoader over the calibration split.
        device: Device to run the calibration pass on.
        bounds: Clamp on the returned temperature.

    Returns:
        The fitted temperature.
    """
    previous = getattr(wrapper, "temperature", 1.0)
    wrapper.temperature = 1.0
    wrapper.eval()
    wrapper.to(device)

    all_logits, all_targets = [], []
    for inputs, targets in calib_loader:
        logits = wrapper(inputs.to(device))[0]
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if not all_logits:
        logger.warning("Empty calibration loader; leaving temperature unchanged.")
        wrapper.temperature = previous
        return float(previous)

    temperature = fit_temperature(
        torch.cat(all_logits), torch.cat(all_targets), bounds=bounds
    )
    return temperature
