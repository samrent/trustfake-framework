"""Fold a 3-class detector onto a binary benchmark's label space.

SID-Set is 3-class (0 real / 1 synthetic / 2 tampered); FakeClue is binary
(0 real / 1 fake, after :mod:`trustfake.data.fake_clue` remaps it to this
project's convention). To score a SID-Set checkpoint on FakeClue without
retraining, the model's three columns have to become two.

The fold is on probabilities, not logits: ``p_fake = P(synthetic) +
P(tampered)``, matching ``trustfake.metrics.moderation``'s repo-wide
definition of ``p_fake = 1 - P(real)``. Summing logits instead would be
meaningless -- logits are unnormalised and their sum is not the logit of the
union.

The result is returned as ``log`` of the folded probabilities, which is what
lets the rest of the harness stay unchanged: ``softmax(log p) == p`` exactly
when ``p`` sums to one, so `BaseWrapper` recovers the intended probabilities,
and every metric, attack and threshold downstream sees a well-formed 2-class
model.

**What this costs, stated because it is the opposite of the project's other
direction of travel.** The fold deliberately merges the two classes that
``detection_auroc_tampered`` and ``detection_auroc_synthetic`` exist to keep
apart. On FakeClue that is unavoidable -- its ground truth genuinely does not
distinguish a fully generated image from an edited one. It is a property of
the benchmark, not a modelling choice, and it is the reason So-Fake-OOD (whose
labels do carry the distinction) is the better shift condition where both are
available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["BinaryFoldClassifier", "REAL_CLASS"]

#: Index of the real class, shared with SID-Set and
#: `trustfake.metrics.evaluation.failure_detection.DetectionAUROC`.
REAL_CLASS = 0

#: Floor before the log, so a saturated P(real) == 0 or == 1 cannot produce
#: -inf logits and NaN gradients. A gradient attack against a folded model
#: would otherwise fail with no diagnostic.
_EPS = 1e-12


class BinaryFoldClassifier(nn.Module):
    """Wrap a multi-class ``x -> logits`` module as a binary one.

    Args:
        inner: The multi-class model. Must return raw logits over
            ``num_classes >= 2`` columns.
        real_class: Which column is "real". Every other column folds into
            "fake".

    The output columns are ``[real, fake]`` -- this project's convention,
    where real is class 0. Note that FakeClue's *own* json uses the opposite
    order; :mod:`trustfake.data.fake_clue` already remaps its labels, so the
    dataset and this wrapper agree.
    """

    def __init__(self, inner: nn.Module, real_class: int = REAL_CLASS) -> None:
        super().__init__()
        self.inner = inner
        self.real_class = real_class

    def forward(self, x: Tensor) -> Tensor:
        logits = self.inner(x)
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError(
                f"expected (B, C>=2) logits to fold, got {tuple(logits.shape)}"
            )
        if not 0 <= self.real_class < logits.shape[1]:
            raise ValueError(
                f"real_class {self.real_class} out of range for "
                f"{logits.shape[1]} columns"
            )
        probs = torch.softmax(logits, dim=1)
        p_real = probs[:, self.real_class]
        # 1 - P(real) rather than a sum over the other columns: identical in
        # exact arithmetic, and it keeps the two outputs summing to one under
        # floating point, which `softmax(log p) == p` depends on.
        p_fake = 1.0 - p_real
        folded = torch.stack((p_real, p_fake), dim=1).clamp_min(_EPS)
        return torch.log(folded)
