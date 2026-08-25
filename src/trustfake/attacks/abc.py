from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["AdversarialAttack", "AttackResult"]


@dataclass
class AttackResult:
    """What an attack produced, beyond the perturbed batch itself.

    The metadata fields exist because some attacks make guarantees that a
    re-forward cannot verify. ACE accepts a perturbation only when the
    argmax is unchanged, so its label preservation is exact by construction
    -- but only as measured on the accept-check forward. Re-running the
    model on `perturbed` afterwards may use a different batch shape, and
    some backends (e.g. cuDNN convolutions) are not bit-identical across
    batch shapes, so a boundary sample can flip and preservation read 0.998
    for reasons unrelated to the attack. When `accepted_logits` is present,
    it is the reported forward.

    Attributes:
        perturbed: The perturbed inputs, same shape and range as the clean
            inputs. Always present.
        effective_eps: Per-sample L_inf perturbation actually applied,
            shape (B,). None when the attack does not track it.
        clean_preds: The model's predictions on the clean inputs, shape
            (B,). None when the attack does not track them.
        accepted_logits: Logits from the forward pass that accepted the
            perturbation, shape (B, C). None when the attack has no accept
            test.
    """

    perturbed: torch.Tensor
    effective_eps: torch.Tensor | None = None
    clean_preds: torch.Tensor | None = None
    accepted_logits: torch.Tensor | None = None


class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks used in training/evaluation pipelines.

    Attacks are called against a `TrustFakeWrapper`, whose `forward` applies
    normalization internally (see `TrustFakeWrapper` docstring) so that the
    inputs seen here stay in their original, unnormalized range. Subclasses
    should craft perturbations in that same range and keep the output inside
    `[clip_min, clip_max]`.

    Implement `__call__` for a plain-tensor attack. An attack that produces
    metadata (per-sample epsilon, an accept-check forward) overrides `run`
    instead and implements `__call__` as `self.run(...).perturbed`; callers
    that can use the metadata call `run`, everyone else keeps calling the
    attack directly.

    Args:
        eps (float): Maximum perturbation radius
        clip_min (float): Minimum valid value for a perturbed input.
        clip_max (float): Maximum valid value for a perturbed input.
    """

    def __init__(
        self, eps: float = 0.032, clip_min: float = 0.0, clip_max: float = 1.0
    ):
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier for the attack, used to prefix metrics and storage keys.
        """
        ...

    @abstractmethod
    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Craft adversarial examples from `inputs` against `model`.

        Args:
            model (TrustFakeWrapper): Model under attack.
            inputs (torch.Tensor): Clean, unnormalized inputs, shape (B, ...).
            targets (torch.Tensor | None): Ground-truth labels. When omitted,
                implementations should fall back to the model's own predictions
                to avoid label leaking.

        Returns:
            torch.Tensor: Perturbed inputs, same shape and range as `inputs`.
        """
        ...

    def run(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AttackResult:
        """
        Rich-result entry point. The default wraps `__call__` with no
        metadata; attacks that track metadata override this instead and
        implement `__call__` as `self.run(...).perturbed`.
        """
        return AttackResult(perturbed=self(model, inputs, targets))

    def _clamp(self, perturbed: torch.Tensor) -> torch.Tensor:
        return perturbed.clamp(self.clip_min, self.clip_max)
