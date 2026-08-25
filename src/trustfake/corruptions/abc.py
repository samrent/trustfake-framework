"""Common-corruption evaluation conditions: the benchmark's third column.

A keystone robustness table has three columns -- clean, adversarial and
common corruption -- and they answer different questions. The adversarial
column asks what a worst-case, budget-bounded attacker can do. The
corruption column asks what the ordinary life of an image on a platform
does to it: re-encoded on upload, thumbnailed for a feed, screenshotted,
re-uploaded. A detector can be robust to one and brittle to the other, so
neither substitutes for the other and the two must never be averaged into
a single "robustness" number.

Corruptions here reuse the attack interface -- `name` and
`run(model, inputs, targets) -> AttackResult` -- so the existing
evaluation pipe scores them with no changes to it, and they declare
`AttackFamily.CORRUPTION` so a results table can keep the columns apart.

WHERE THE CORRUPTION IS APPLIED, AND WHY IT IS NOT WHERE WP1 APPLIED IT
----------------------------------------------------------------------
These act on the MODEL INPUT: the float tensor in [0, 1] the datamodule
already produced, i.e. AFTER the resize to 224x224. That is the ImageNet-C
convention, and it is deliberately not what the WP1 feature extractor did
(`wp1/src/features.py:apply_condition` corrupts the PIL image at its
original resolution, before the model transform).

The difference is not cosmetic. A 1024x1024 original re-encoded at JPEG
q40 and *then* downsampled to 224 loses most of its blocking artifacts to
the resampling filter, so the pre-resize condition reads as far milder
than it is -- and its severity varies with the original resolution, which
on SID-Set is itself class-correlated (see `trustfake.data.baselines`).
Corrupting the model input instead:

  * models the platform re-encoding what it actually *serves* -- the
    resized derivative a moderation system is handed, not the master file;
  * fixes the corruption strength on one grid, so a row is comparable
    across images whose originals differ by an order of magnitude in size,
    and comparable across the ladder's rungs;
  * keeps the condition in the same space as the adversarial
    perturbations, which is the only way the two columns are commensurable.

What it does NOT model is the destruction of high-frequency forensic
traces that pre-resize compression causes -- a strictly harsher condition.
A report must say which of the two it measured; this package measures the
post-resize one.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from trustfake.attacks.abc import (
    AdversarialAttack,
    AttackDirection,
    AttackFamily,
    AttackResult,
)
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["ImageCorruption"]


class ImageCorruption(AdversarialAttack):
    """Base class for a common-corruption evaluation condition.

    A corruption is NOT an eps-bounded perturbation. An adversarial attack
    is defined by a budget -- the threat model is "anything inside this
    L_inf ball" -- and its result is only meaningful read against that
    radius. A corruption is a distributional-shift condition: JPEG q40 is a
    named transformation of the image, not a point inside a ball around it,
    and it routinely moves pixels much further than any epsilon a
    robustness table would report. `eps` is therefore `float("inf")` here,
    which is the honest value and not a placeholder, and corruptions are
    excluded from the eps-ball contract battery in
    `tests/attacks/test_attack_contracts.py` on purpose -- they have their
    own tests, because they make different promises.

    `AttackResult.effective_eps` is still filled in, as a *diagnostic*
    magnitude: it says how far the condition actually moved the batch, so
    a corruption row can be placed next to an adversarial row of known
    radius. It is a measurement, never a budget that was respected.

    Subclasses implement `name` -- which must encode the parameter, so
    `jpeg_q40` and `jpeg_q90` cannot collide in a metric prefix or a log
    directory -- and `corrupt`, which does the work. `run` and `__call__`
    are provided.

    Args:
        clip_min (float): Minimum valid value for a corrupted input.
        clip_max (float): Maximum valid value for a corrupted input.
    """

    family = AttackFamily.CORRUPTION
    direction = AttackDirection.NONE
    #: A corruption never consumes ground truth: it is applied to the image
    #: and is the same transformation whatever the label says.
    uses_labels = False
    #: No budget, so no norm to express one in.
    norm = "none"
    minimum_norm = False

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        super().__init__(eps=float("inf"), clip_min=clip_min, clip_max=clip_max)

    @abstractmethod
    def corrupt(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply the corruption to a batch of model inputs.

        Args:
            images (torch.Tensor): Model inputs, shape (B, C, H, W), in
                [clip_min, clip_max].

        Returns:
            torch.Tensor: The corrupted batch, same shape. Implementations
                may return it on another device or in another dtype (the
                codec round-trips work on CPU uint8); `run` moves it back.
        """
        ...

    def run(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AttackResult:
        """
        Corrupt `inputs`. `model` and `targets` are accepted for interface
        compatibility with `AdversarialAttack` and deliberately unused: a
        corruption is model-independent, which is exactly what makes it a
        transferable condition rather than an attack on one detector.
        """
        with torch.no_grad():
            corrupted = self.corrupt(inputs.detach())
            corrupted = self._clamp(
                corrupted.to(device=inputs.device, dtype=inputs.dtype)
            )
            return AttackResult(
                perturbed=corrupted,
                effective_eps=(corrupted - inputs).abs().flatten(1).amax(dim=1),
            )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
