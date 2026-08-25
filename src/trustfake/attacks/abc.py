from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import torch

from trustfake.models.wrapper import TrustFakeWrapper

__all__ = [
    "AdversarialAttack",
    "AttackResult",
    "AttackFamily",
    "AttackDirection",
    "attack_registry",
]


class AttackFamily(StrEnum):
    """What an attack is trying to break.

    The distinction is the reason this project exists, so it is data on the
    attack rather than prose in a table. A PREDICTION attack degrades
    selective risk as a *side effect* of destroying accuracy -- an accuracy
    monitor sees it immediately. A CONFIDENCE attack degrades selective risk
    while leaving accuracy bit-identical, so the same monitor sees a healthy
    system. Reporting both under one "robustness" heading is what makes the
    second one invisible.
    """

    PREDICTION = "prediction"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"
    EVIDENCE = "evidence"
    CORRUPTION = "corruption"


class AttackDirection(StrEnum):
    """Which way a confidence-targeted attack pushes confidence.

    OVER inflates it (the model becomes sure of its mistakes), UNDER
    deflates it, BOTH does each where it hurts most (down on correct
    predictions, up on incorrect ones -- ACE's rule). NONE is for attacks
    where the notion does not apply.
    """

    OVER = "over_confidence"
    UNDER = "under_confidence"
    BOTH = "both"
    NONE = "none"


def attack_registry() -> dict[str, dict]:
    """Taxonomy of every exported attack: family, direction, label use, norm.

    Built by instantiating each attack at its defaults, so it cannot drift
    from the implementations the way a hand-maintained table does. Used to
    group a results table by threat family and to flag which rows depend on
    ground-truth labels (an attacker who has them is a different, stronger
    threat model than one who does not).
    """
    from trustfake import attacks as _attacks

    registry: dict[str, dict] = {}
    for symbol in _attacks.__all__:
        obj = getattr(_attacks, symbol)
        if not (isinstance(obj, type) and issubclass(obj, AdversarialAttack)):
            continue
        if obj is AdversarialAttack:
            continue
        instance = obj()
        registry[instance.name] = {
            "class": symbol,
            "family": str(instance.family),
            "direction": str(instance.direction),
            "uses_labels": instance.uses_labels,
            "norm": instance.norm,
            "minimum_norm": instance.minimum_norm,
        }
    return registry


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
        l2_norm: Per-sample L2 norm of the perturbation, shape (B,). Set by
            the minimum-norm attacks (DeepFool, C&W, BB, PDPGD), for which
            this -- not `eps` -- is the quantity being minimised and the one
            a robustness curve should be read against. None for fixed-budget
            attacks, where it adds nothing to `effective_eps`.
        success: Per-sample flag, shape (B,), for whether the attack reached
            its own goal -- a changed prediction for the minimum-norm
            attacks. None when the attack does not define one. A minimum-norm
            attack that fails on a sample returns that sample unperturbed
            rather than a perturbation that does nothing, so this is the
            field that separates "robust" from "not attacked".
    """

    perturbed: torch.Tensor
    effective_eps: torch.Tensor | None = None
    clean_preds: torch.Tensor | None = None
    accepted_logits: torch.Tensor | None = None
    l2_norm: torch.Tensor | None = None
    success: torch.Tensor | None = None


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

    Subclasses declare their place in the taxonomy by overriding `family`,
    `direction`, `uses_labels`, `norm` and `minimum_norm`. The defaults
    describe the most common case -- a fixed-budget, label-free, L_inf
    prediction attack -- so only the attacks that differ have to say so.

    Args:
        eps (float): Maximum perturbation radius
        clip_min (float): Minimum valid value for a perturbed input.
        clip_max (float): Maximum valid value for a perturbed input.
    """

    #: What the attack tries to break. See `AttackFamily`.
    family: AttackFamily = AttackFamily.PREDICTION
    #: Which way it pushes confidence, for the confidence family.
    direction: AttackDirection = AttackDirection.NONE
    #: Whether the attack consumes ground-truth labels. An attacker holding
    #: them is a strictly stronger threat model, so rows produced with and
    #: without labels must not be compared as if they were the same attack.
    uses_labels: bool = False
    #: Norm the budget is expressed in.
    norm: str = "linf"
    #: True when `eps` is a cap on the result rather than the search budget:
    #: the attack minimises its norm and `AttackResult.l2_norm` -- not
    #: `eps` -- is the quantity to plot.
    minimum_norm: bool = False

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
