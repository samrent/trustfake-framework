from .abc import TrainingModule
from .adversarial_training import (
    HybridAdversarialTrainingModule,
    MARTTrainingModule,
    PGDAdversarialTrainingModule,
    TRADESTrainingModule,
)
from .awp import AWPMixin
from .confidence_training import (
    ConfidenceAdversarialTrainingModule,
    ConfidenceRegularisedTrainingModule,
)
from .depth_auxiliary import (
    DepthAuxiliaryMixin,
    DepthPGDAdversarialTrainingModule,
    DepthStandardTrainingModule,
    DepthTRADESTrainingModule,
)
from .evidential_adversarial_training import EvidentialAdversarialTrainingModule
from .standard_training import StandardTrainingModule

__all__ = [
    "TrainingModule",
    "AWPMixin",
    "StandardTrainingModule",
    # label-axis defences
    "PGDAdversarialTrainingModule",
    "TRADESTrainingModule",
    "HybridAdversarialTrainingModule",
    "MARTTrainingModule",
    # confidence-axis defences
    "ConfidenceAdversarialTrainingModule",
    "ConfidenceRegularisedTrainingModule",
    # evidential
    "EvidentialAdversarialTrainingModule",
    # Track C: auxiliary depth regularisation
    "DepthAuxiliaryMixin",
    "DepthStandardTrainingModule",
    "DepthPGDAdversarialTrainingModule",
    "DepthTRADESTrainingModule",
]
