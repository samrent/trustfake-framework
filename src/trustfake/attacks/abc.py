from abc import ABC, abstractmethod

import torch

from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["AdversarialAttack"]


class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks used in training/evaluation pipelines.

    Attacks are called against a `TrustFakeWrapper`, whose `forward` applies
    normalization internally (see `TrustFakeWrapper` docstring) so that the
    inputs seen here stay in their original, unnormalized range. Subclasses
    should craft perturbations in that same range and keep the output inside
    `[clip_min, clip_max]`.

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

    def _clamp(self, perturbed: torch.Tensor) -> torch.Tensor:
        return perturbed.clamp(self.clip_min, self.clip_max)
