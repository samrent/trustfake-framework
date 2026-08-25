"""Wrappers around the fra31 `autoattack` package.

These expose APGD, FAB, Square and the AutoAttack ensemble through the
framework's `AdversarialAttack` interface. The package calls `model(x)` and
expects logits, so each wrapper drives the model through a `LogitsAdapter`.

A note on binary detectors (recorded, not hypothetical): AutoAttack's
targeted stages read the 3rd- and 4th-largest logits, so `apgd-t` / `fab-t`
raise `IndexError` on a 2-logit model, and `version="standard"` only reaches
them once points survive `apgd-ce` -- i.e. it breaks exactly against a
*robust* detector. `AutoAttackLinf` therefore composes untargeted stages
(`apgd-ce`, `fab`, `square`) and sets FAB's target-class count so it never
reaches for logits that do not exist. For a 3-class model the standard
ensemble is available via `version="standard"`.

All wrappers take a `seed` so the randomised components are deterministic
for a fixed model and input, as the evaluation harness requires.
"""

from __future__ import annotations

import torch

from trustfake.attacks._common import LogitsAdapter
from trustfake.attacks.abc import AdversarialAttack
from trustfake.logging import get_logger
from trustfake.models.wrapper import TrustFakeWrapper

logger = get_logger("autoattack")

__all__ = ["APGD", "FAB", "SquareAttack", "AutoAttackLinf"]


class _AutoAttackBase(AdversarialAttack):
    """Shared plumbing: build an AutoAttack driver in custom mode over a
    LogitsAdapter and run the chosen component(s)."""

    _attacks_to_run: list[str] = []

    def __init__(
        self,
        eps: float = 8 / 255,
        seed: int = 0,
        n_iter: int = 100,
        n_queries: int = 5000,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.seed = seed
        self.n_iter = n_iter
        self.n_queries = n_queries

    def _build(self, model: TrustFakeWrapper):
        from autoattack import AutoAttack

        adapter = LogitsAdapter(model)
        device = next(model.parameters()).device
        aa = AutoAttack(
            adapter,
            norm="Linf",
            eps=self.eps,
            version="custom",
            attacks_to_run=list(self._attacks_to_run),
            seed=self.seed,
            verbose=False,
            device=device,
        )
        aa.apgd.n_iter = self.n_iter
        aa.apgd.n_restarts = 1
        aa.fab.n_restarts = 1
        aa.fab.n_target_classes = 1  # never reach for logits that may not exist
        aa.square.n_queries = self.n_queries
        return aa

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.eps == 0:
            return inputs.clone()
        was_training = model.training
        model.eval()
        if targets is None:
            with torch.no_grad():
                targets = model(inputs)[2].detach()
        aa = self._build(model)
        with torch.enable_grad():
            adv = aa.run_standard_evaluation(inputs, targets.long(), bs=inputs.shape[0])
        model.train(was_training)
        # The package already clips to [0,1]/eps; clamp defensively to the
        # configured range and re-project for exact contract compliance.
        adv = self._clamp(inputs + (adv - inputs).clamp(-self.eps, self.eps))
        return adv.detach()


class APGD(_AutoAttackBase):
    """Auto-PGD, cross-entropy (Croce & Hein, ICML 2020). Parameter-free
    step-size PGD; the core component of AutoAttack."""

    _attacks_to_run = ["apgd-ce"]

    @property
    def name(self) -> str:
        return "apgd"


class FAB(_AutoAttackBase):
    """Fast Adaptive Boundary attack (Croce & Hein, ICML 2020). Minimum-norm;
    ``eps`` bounds the reported perturbation."""

    _attacks_to_run = ["fab"]

    @property
    def name(self) -> str:
        return "fab"


class SquareAttack(_AutoAttackBase):
    """Square Attack (Andriushchenko et al., ECCV 2020). Query-based,
    gradient-free, L_inf; bounded by ``n_queries``."""

    _attacks_to_run = ["square"]

    @property
    def name(self) -> str:
        return "square"


class AutoAttackLinf(_AutoAttackBase):
    """AutoAttack ensemble (Croce & Hein, ICML 2020), composed to run on any
    class count: untargeted apgd-ce + fab + square. See the module docstring
    for why the targeted stages are excluded."""

    _attacks_to_run = ["apgd-ce", "fab", "square"]

    @property
    def name(self) -> str:
        return "autoattack"
