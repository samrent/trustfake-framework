"""Device selection: one place that resolves CUDA / MPS (Apple Metal) / CPU.

Priority is CUDA, then MPS, then CPU, so the same code runs on a CUDA box
(e.g. a 3090) and on an Apple-silicon machine (Metal) without changes. Override
with the ``TRUSTFAKE_DEVICE`` environment variable or the ``prefer`` argument.

The Lightning Trainer selects its own accelerator via ``accelerator: auto`` in
the trainer config; ``lightning_accelerator`` exposes the same resolution for
completeness. ``resolve_device`` is for the standalone passes that run outside
the Trainer (temperature and moderation fitting in ``test.py``, the baselines
and demo scripts).
"""

from __future__ import annotations

import os

import torch

from trustfake.logging import get_logger

logger = get_logger("device")

__all__ = ["resolve_device", "lightning_accelerator"]


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Args:
        prefer: explicit device string ("cuda", "mps", "cpu", "cuda:1", ...).
            Falls back to the priority order if that device is unavailable.
            Defaults to the ``TRUSTFAKE_DEVICE`` env var when unset.
    """
    choice = prefer if prefer is not None else os.environ.get("TRUSTFAKE_DEVICE")
    if choice:
        choice = choice.lower()
        if choice.startswith("cuda") and torch.cuda.is_available():
            return torch.device(choice)
        if choice == "mps" and _mps_available():
            return torch.device("mps")
        if choice == "cpu":
            return torch.device("cpu")
        logger.warning(
            f"Requested device '{choice}' is unavailable; falling back to auto."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif _mps_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.debug(f"Resolved device: {device}")
    return device


def lightning_accelerator(prefer: str | None = None) -> str:
    """Lightning accelerator string matching `resolve_device`'s choice."""
    device = resolve_device(prefer)
    return {"cuda": "gpu", "mps": "mps", "cpu": "cpu"}[device.type]
