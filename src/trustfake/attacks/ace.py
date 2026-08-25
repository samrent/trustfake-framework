import torch

from trustfake.attacks.abc import AdversarialAttack
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["ACE", "softmax_response"]


def softmax_response(logits: torch.Tensor) -> torch.Tensor:
    """Per-class softmax probabilities of `logits`, shape (B, C)."""
    return torch.softmax(logits, dim=1)


class ACE(AdversarialAttack):
    """
    Attack on Confidence Estimation (ACE).
    """

    def __init__(
        self,
        eps: float = 0.005,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)

    @property
    def name(self) -> str:
        return "ace"

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
