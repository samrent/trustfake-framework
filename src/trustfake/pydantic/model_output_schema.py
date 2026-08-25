import torch

from pydantic import BaseModel, ConfigDict, field_validator
from trustfake.logging import get_logger

logger = get_logger("output-validation")


class ClassificationModelOutput(BaseModel):
    """
    Pydantic model for the validation of classification module outputs.
    It checks that the module outputs four tensors:
    - logits: Raw model outputs (logits).
    - probs: Probabilities obtained from logits.
    - preds: Predicted class labels.
    - uncertainty: Uncertainty map.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor
    uncertainty: torch.Tensor

    @field_validator("preds")
    @classmethod
    def check_preds(cls, v: torch.Tensor) -> torch.Tensor:
        """The tensor `preds` must be a 1D tensor with shape (B,)
        with B being the batch size."""
        if not v.ndim == 1:
            raise ValueError(
                f"Preds must be a 1D tensor with shape (B,), got {v.shape}"
            )
        return v

    @field_validator("uncertainty")
    @classmethod
    def check_uncertainty(cls, v: torch.Tensor) -> torch.Tensor:
        """The tensor `uncertainty` must be a 1D tensor with shape (B,)
        with B being the batch size."""
        if not v.ndim == 1:
            raise ValueError(
                f"Uncertainty must be a 1D tensor with shape (B,), got {v.shape}"
            )
        return v

    @field_validator("logits", "probs")
    @classmethod
    def check_logits_probs(cls, v: torch.Tensor) -> torch.Tensor:
        """
        Logits for multi-class classification must be a 2D tensor with shape
        (B, C), where C is the number of classes and must be higher than 1.
        """
        if not v.ndim == 2 or v.shape[1] <= 1:
            raise ValueError(
                f"Logits must be a 2D tensor with shape (B, C), where"
                f" C is higher than 1, got {v.shape}"
            )
        return v
