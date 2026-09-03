"""`DepthConsistencyWrapper`: the evaluation-time owner of the depth path.

Every uncertainty score in this repo receives only the class probabilities,
because that is all `BaseWrapper.forward` has. The depth-consistency score
needs two more things -- the student's depth map and a frozen teacher's map
of the SAME input -- and this wrapper is the one place that has all three.
The design follows the MC-dropout precedent: the wrapper drives the model,
the score stays pure tensor math.

What it does and does not change:

* With a probability-only score (`uncertainty_score=multiclass_max_probability`)
  it IS `BaseWrapper`, op for op; the teacher is never built. That is the
  first of the three scorings of a Track C checkpoint, and it costs nothing.
* With a `DepthAwareScore` it runs `forward_with_depth` (one backbone pass,
  both heads) and the teacher, and hands the score `(probs, head, teacher)`.
  `forward`'s return stays the 4-tuple every attack and pipe reads; only the
  4th element's meaning changes -- which is the point: the confidence-axis
  attacks (`ace`, `overconf`, `query_*`, `uncertainty_fgsm`) then attack the
  depth score itself, and gradient attacks see the gradient THROUGH the
  teacher (`teacher_grad=True`) rather than the student half only.
* `outputs_from_logits` returns None for a depth-aware score, so the eval
  pipe re-forwards the perturbed batch instead of deriving the uncertainty
  from an attack's accepted logits (which carry no depth) -- the teacher
  must see the attacked pixels, not the clean ones.

The teacher is deliberately kept OUT of the module tree (`object.__setattr__`)
so it never enters `state_dict()`, `parameters()` or a checkpoint: the
checkpoint is the model's, the teacher is the instrument's. It is moved to
the input's device on first use. The three scorings of one checkpoint are
therefore three ordinary evaluation runs: `wrapper=base` (or `depth`) with
`multiclass_max_probability`, `wrapper=depth` with `depth_consistency`, and
`wrapper=depth` with `depth_combined`.

What the ATTACK sees is an explicit, recorded choice (`attack_scoring`):

* ``white_box`` (default): the wrapper is the same object inside and outside
  the attack, so a confidence-axis attack (`ace`, `overconf`, `query_*`,
  `uncertainty_fgsm`) attacks the depth score itself, teacher included. The
  honest number for "how robust is this rejection score"; also the
  expensive one -- every attack iteration runs the teacher (PGD-10 is eleven
  teacher forwards per batch).
* ``transfer``: inside `attacking()` the wrapper returns `1 - max prob` as
  its 4th element and skips the teacher; the depth score is computed only
  on the final perturbed batch. That is "an attack crafted against the
  classifier's confidence, scored by depth": the cheap first-pass protocol,
  and for prediction-axis attacks (which never read the 4th element)
  identical in outcome to white-box at a fraction of the cost.

The evaluation pipe enters `attacking()` around `attack.run` (a no-op for
every other wrapper). Which mode produced a number is in that run's
`experiment_config.yaml`; a table must say which.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from trustfake.depth.teacher import (
    DEFAULT_TEACHER_INPUT_SIZE,
    DEPTH_ANYTHING_V2_SMALL,
    DEPTH_ANYTHING_V2_SMALL_REVISION,
    load_depth_teacher,
)
from trustfake.logging import get_logger
from trustfake.metrics.uncertainty.depth import DepthAwareScore
from trustfake.models.wrapper.base import BaseWrapper

__all__ = ["DepthConsistencyWrapper", "ATTACK_SCORING_MODES"]

logger = get_logger("depth-wrapper")

ATTACK_SCORING_MODES = ("white_box", "transfer")


class DepthConsistencyWrapper(BaseWrapper):
    """`BaseWrapper` that can feed a depth-aware uncertainty score.

    Args:
        teacher: A ready `DepthTeacher` (tests inject `FakeDepthTeacher`).
            None builds the real one lazily from `teacher_name` on first use.
        teacher_name: Hub id of the teacher checkpoint.
        teacher_revision: Pinned revision of that checkpoint.
        teacher_input_size: Target of the teacher's resize rule. Must equal
            the setting the training targets were precomputed with, or the
            residual compares two instruments.
        teacher_grad: Keep the autograd graph through the teacher. True is
            the honest default for gradient attacks on the score; False
            saves memory and makes those attacks see only the student half.
        attack_scoring: What an attack optimises against, see the module
            docstring: "white_box" (the depth score itself) or "transfer"
            (1 - max prob; the teacher runs only on the final batch).

    Raises:
        ValueError: a depth-aware score with a model that has no depth path,
            or an unknown `attack_scoring`.
    """

    def __init__(
        self,
        *args,
        teacher: nn.Module | None = None,
        teacher_name: str = DEPTH_ANYTHING_V2_SMALL,
        teacher_revision: str | None = DEPTH_ANYTHING_V2_SMALL_REVISION,
        teacher_input_size: int = DEFAULT_TEACHER_INPUT_SIZE,
        teacher_grad: bool = True,
        attack_scoring: str = "white_box",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # Not a submodule: never in state_dict / parameters / the checkpoint.
        object.__setattr__(self, "_teacher", teacher)
        self.teacher_name = teacher_name
        self.teacher_revision = teacher_revision
        self.teacher_input_size = int(teacher_input_size)
        self.teacher_grad = bool(teacher_grad)
        if attack_scoring not in ATTACK_SCORING_MODES:
            msg = (
                f"Unknown attack_scoring {attack_scoring!r}; "
                f"available: {list(ATTACK_SCORING_MODES)}"
            )
            raise ValueError(msg)
        self.attack_scoring = attack_scoring
        self._attacking = False
        self._probability_only = False
        if self.consumes_depth and not self.has_depth_head:
            msg = (
                f"uncertainty_score={type(self.uncertainty_score).__name__} needs "
                f"the depth path, but {type(self.model).__name__} has none. Use "
                "model=resnet18_depth with a depth-trained checkpoint, and no "
                "binary fold."
            )
            raise ValueError(msg)

    @property
    def consumes_depth(self) -> bool:
        return isinstance(self.uncertainty_score, DepthAwareScore)

    @contextmanager
    def attacking(self) -> Iterator[None]:
        """Mark the forwards an attack makes. The evaluation pipe wraps
        `attack.run` in this; under "transfer" scoring those forwards return
        1 - max prob and skip the teacher."""
        previous = self._attacking
        self._attacking = True
        try:
            yield
        finally:
            self._attacking = previous

    @contextmanager
    def probability_only(self) -> Iterator[None]:
        """Score by 1 - max prob regardless of the configured score. For the
        temperature fit in `src/test.py`, which needs logits only and runs
        BEFORE the combined score has its calib reference (an unfitted
        combined score raises, and a fitted one would be wasted teacher
        forwards)."""
        previous = self._probability_only
        self._probability_only = True
        try:
            yield
        finally:
            self._probability_only = previous

    @property
    def scoring_by_probability(self) -> bool:
        """True when this forward's 4th element is 1 - max prob."""
        return (
            not self.consumes_depth
            or self._probability_only
            or (self._attacking and self.attack_scoring == "transfer")
        )

    def _probability_forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """`BaseWrapper.forward` with 1 - max prob in the 4th slot, for the
        transfer mode (the configured score cannot take probabilities alone)."""
        x = self.normalization_layer(x)
        logits = self.model(x)
        probs = torch.softmax(logits / self.temperature, dim=1)
        preds = torch.argmax(probs, dim=1)
        return logits, probs, preds, 1.0 - torch.max(probs, dim=1).values

    @property
    def teacher(self) -> nn.Module:
        """The frozen teacher, built on first access."""
        if self._teacher is None:
            logger.info(
                f"Loading depth teacher {self.teacher_name}@{self.teacher_revision} "
                f"(input {self.teacher_input_size})"
            )
            object.__setattr__(
                self,
                "_teacher",
                load_depth_teacher(
                    self.teacher_name,
                    revision=self.teacher_revision,
                    input_size=self.teacher_input_size,
                    autocast=False,  # fp32 online: gradients must survive
                ),
            )
        return self._teacher

    def teacher_depth(self, x: Tensor) -> Tensor:
        """The teacher's map of the RAW input, on the input's device."""
        teacher = self.teacher
        first = next(iter(teacher.parameters()), None)
        device = (
            first.device if first is not None else next(iter(teacher.buffers())).device
        )
        if device != x.device:
            teacher.to(x.device)
        with nullcontext() if self.teacher_grad else torch.no_grad():
            depth = teacher(x)
        return depth if self.teacher_grad else depth.detach()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.consumes_depth:
            return super().forward(x)
        if self.scoring_by_probability:
            return self._probability_forward(x)
        self.uncertainty_score = self.uncertainty_score.to(x.device)
        logits, depth_pred = self.forward_with_depth(x)
        depth_target = self.teacher_depth(x)
        probs = torch.softmax(logits / self.temperature, dim=1)
        preds = torch.argmax(probs, dim=1)
        uncertainty = self.uncertainty_score(probs, depth_pred, depth_target)
        self.uncertainty_score.reset()
        return logits, probs, preds, uncertainty

    def outputs_from_logits(
        self, logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """None for a depth-aware score: the residual needs the pixels, so the
        eval pipe must re-forward the perturbed batch (the MC-dropout path)."""
        if self.consumes_depth:
            return None
        return super().outputs_from_logits(logits)

    @torch.no_grad()
    def score_components(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(1 - max prob, depth residual) on a batch, for fitting the combined
        score's calib reference. Uses the CURRENT temperature, so call it
        after temperature scaling."""
        logits, depth_pred = self.forward_with_depth(x)
        probs = torch.softmax(logits / self.temperature, dim=1)
        msp = 1.0 - torch.max(probs, dim=1).values
        from trustfake.metrics.uncertainty.depth import depth_consistency_residual

        residual = depth_consistency_residual(depth_pred, self.teacher_depth(x))
        return msp, residual
