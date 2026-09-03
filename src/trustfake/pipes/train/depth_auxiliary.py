"""Multi-task depth regularisation arms (Track C).

The hypothesis, from Mao et al. (ECCV 2020, "Multitask Learning Strengthens
Adversarial Robustness"): an attacker of a network trained on two objectives
must fool both through a shared backbone whose task gradients are not
aligned, so robust accuracy improves. Here the second task is monocular
depth from a frozen teacher (see `trustfake.depth`), and the loss is

    L = CE + depth_lambda * SSI_L1(head(x'), depth_target(x))

where `x'` is whatever input the arm trains its classifier on -- the CLEAN
image for `standard_depth`, the ADVERSARIAL one for `pgd_at_depth` and
`trades_depth` -- and the target is always the CLEAN image's precomputed
teacher map. Supervising the head on x_adv against the clean geometry is the
strongest form of the effect (the geometry must stay stable under the
perturbation); the inner maximisation stays CE-only, exactly the parent
arm's, so the only change to `pgd_at` is the outer loss. Attacking the joint
objective is a follow-up, not this pass.

Mechanically each arm is its parent with ONE difference: the train-mode
forward that feeds the loss goes through `forward_with_depth` (one backbone
pass, both heads) instead of `forward`, so the BatchNorm update count per
step is unchanged and pinned by the same test as the parents'. The
classification half of the output is rebuilt from those logits with
`outputs_from_logits`, which is why MC dropout (which cannot) is refused.

Everything is refused, never warned: a model without a head, a batch without
a depth target (the datamodule was not given `depth_targets_dir`), a
non-positive lambda (`standard_depth` at lambda 0 is `standard` with extra
parameters, and would read as a null result of the method).

AWP composes as with the parents, and -- following the TRADES/EV-AT
precedent in `awp.py` -- the weight adversary ascends the loss actually
trained: the parent's objective PLUS the depth term. Left at the parent's
objective, "AWP on pgd_at_depth" and "AWP on pgd_at" would be two different
modifiers wearing one name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import cross_entropy, kl_div, log_softmax

from trustfake.attacks._common import model_logits
from trustfake.losses.depth import ScaleShiftInvariantL1
from trustfake.models.wrapper.mc_dropout import MCDropoutWrapper
from trustfake.pipes.train._common import RobustValidationMixin, eval_mode
from trustfake.pipes.train.abc import TrainingModule
from trustfake.pipes.train.adversarial_training import (
    PGDAdversarialTrainingModule,
    TRADESTrainingModule,
)
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

__all__ = [
    "DepthAuxiliaryMixin",
    "DepthStandardTrainingModule",
    "DepthPGDAdversarialTrainingModule",
    "DepthTRADESTrainingModule",
    "check_depth_arm",
]


def check_depth_arm(pipe_cls: type, wrapper, datamodule, pipe_name: str) -> None:
    """Refuse the three quiet mismatches between arm, model and data.

    Called by `src/train.py` before the pipe is built. Each of these would
    otherwise run to completion and report a plausible number:

    * a depth arm on a datamodule with no depth targets -- dies on the
      first batch, but only after the checkpoint tree has been created;
    * a plain arm on a datamodule that carries depth targets -- the extra
      batch element is silently ignored, and the run reads as a depth run;
    * a plain arm on a model that HAS a depth head -- the head is never
      trained, but the checkpoint carries it and an evaluation with the
      depth score would compute a residual against random weights.
    """
    is_depth_arm = isinstance(pipe_cls, type) and issubclass(
        pipe_cls, DepthAuxiliaryMixin
    )
    has_targets = bool(getattr(datamodule, "has_depth_targets", False))
    if is_depth_arm and not has_targets:
        msg = (
            f"training_pipe={pipe_name} needs depth targets; set "
            "datamodule.datamodule.depth_targets_dir to a store written by "
            "src/precompute_depth.py."
        )
        raise ValueError(msg)
    if has_targets and not is_depth_arm:
        msg = (
            f"the datamodule carries depth targets but training_pipe={pipe_name} "
            "does not consume them; select standard_depth / pgd_at_depth / "
            "trades_depth, or unset depth_targets_dir."
        )
        raise ValueError(msg)
    if wrapper.has_depth_head and not is_depth_arm:
        msg = (
            f"the model has a depth head but training_pipe={pipe_name} does not "
            "train it; select a depth arm, or model=resnet18. An untrained head "
            "in a checkpoint would give the depth-consistency score a random "
            "reference."
        )
        raise ValueError(msg)


class DepthAuxiliaryMixin:
    """The depth term, shared by the three arms. Sits in front of the parent
    arm in the MRO; cooperative constructor.

    Args:
        depth_lambda: Weight of the depth term. Must be positive.
        depth_loss: The depth loss; `ScaleShiftInvariantL1()` by default.
    """

    def __init__(
        self,
        *args,
        depth_lambda: float = 1.0,
        depth_loss: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if depth_lambda <= 0:
            msg = (
                f"depth_lambda must be positive, got {depth_lambda}. A depth arm "
                "with no depth term is the parent arm with extra parameters; "
                "select the parent arm instead of an ablation that reads as a "
                "null result of the method."
            )
            raise ValueError(msg)
        self.depth_lambda = float(depth_lambda)
        self.depth_loss = (
            depth_loss if depth_loss is not None else ScaleShiftInvariantL1()
        )
        if not self.model.has_depth_head:
            msg = (
                f"{type(self).__name__} needs a model with a depth head "
                "(forward_with_depth); use model=resnet18_depth."
            )
            raise ValueError(msg)
        if isinstance(self.model, MCDropoutWrapper):
            msg = (
                f"{type(self).__name__} rebuilds the classification output from a "
                "single joint forward, which MC dropout cannot do; use "
                "wrapper=base or wrapper=evidential."
            )
            raise ValueError(msg)
        if getattr(self.model, "consumes_depth", False):
            msg = (
                f"{type(self).__name__} trains with a probability score; the "
                "depth-consistency score is an evaluation-time instrument. Use "
                "wrapper=base uncertainty_score=multiclass_max_probability for "
                "training and wrapper=depth only in src/test.py."
            )
            raise ValueError(msg)

    @staticmethod
    def _depth_target(batch) -> Tensor:
        if len(batch) < 3 or batch[2] is None:
            msg = (
                "a depth arm needs the depth target as the batch's third element; "
                "set datamodule.datamodule.depth_targets_dir to a precomputed store "
                "(src/precompute_depth.py)."
            )
            raise ValueError(msg)
        return batch[2]

    def _joint_forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, ClassificationModelOutput]:
        """ONE train-mode forward: (logits, depth map, classification output)."""
        logits, depth_pred = self.model.forward_with_depth(x)
        logits, probs, preds, uncertainty = self.model.outputs_from_logits(logits)
        output = ClassificationModelOutput(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )
        return logits, depth_pred, output

    def _depth_term(self, depth_pred: Tensor, depth_target: Tensor) -> Tensor:
        loss = self.depth_loss(depth_pred, depth_target.to(depth_pred.device))
        if self._trainer is not None:
            name = "train_depth_loss" if self.training else "val_depth_loss"
            self.log(name, loss.detach(), on_epoch=True, prog_bar=False)
        return loss

    def awp_objective(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> Tensor:
        """The parent's weight-space objective plus the depth term, so AWP
        ascends what is trained. The depth target is stashed by
        `compute_loss` before `apply_awp`, because AWP's signature has no
        slot for it."""
        target = getattr(self, "_awp_depth_target", None)
        if target is None:
            msg = "awp_objective called without a stashed depth target"
            raise ValueError(msg)
        _, depth_pred = self.model.forward_with_depth(x_adv)
        return super().awp_objective(inputs, x_adv, targets) + (
            self.depth_lambda
            * self.depth_loss(depth_pred, target.to(depth_pred.device))
        )

    def _apply_awp_with_depth(
        self, x: Tensor, x_adv: Tensor, y: Tensor, depth_target: Tensor
    ) -> None:
        self._awp_depth_target = depth_target
        try:
            self.apply_awp(x, x_adv, y)
        finally:
            self._awp_depth_target = None


class DepthStandardTrainingModule(
    DepthAuxiliaryMixin, RobustValidationMixin, TrainingModule
):
    """`standard` + the depth term on the clean image."""

    def compute_loss(self, batch) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        depth_target = self._depth_target(batch)
        logits, depth_pred, output = self._joint_forward(x)
        ce = self.model.loss_fn(self.model.loss_input(logits), y)
        loss = ce + self.depth_lambda * self._depth_term(depth_pred, depth_target)
        return loss, output


class DepthPGDAdversarialTrainingModule(
    DepthAuxiliaryMixin, PGDAdversarialTrainingModule
):
    """`pgd_at` + the depth term: the head sees the ADVERSARIAL input, the
    target is the CLEAN image's geometry. Inner maximisation unchanged
    (CE-only, eval mode); metrics on the clean forward, as in the parent."""

    def compute_loss(self, batch) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        depth_target = self._depth_target(batch)
        x_adv = self._pgd_ce(x, y, self.current_eps)
        self._apply_awp_with_depth(x, x_adv, y, depth_target)

        adv_logits, depth_pred = self.model.forward_with_depth(x_adv)
        ce = self.model.loss_fn(self.model.loss_input(adv_logits), y)
        loss = ce + self.depth_lambda * self._depth_term(depth_pred, depth_target)
        return loss, self._metrics_output(x)


class DepthTRADESTrainingModule(DepthAuxiliaryMixin, TRADESTrainingModule):
    """`trades` + the depth term on the ADVERSARIAL forward (the same forward
    the KL term uses), target the clean geometry. Two train-mode forwards,
    like the parent."""

    def compute_loss(self, batch) -> tuple[Tensor, ClassificationModelOutput]:
        x, y = batch[0], batch[1].long()
        depth_target = self._depth_target(batch)
        eps = self.current_eps

        with torch.no_grad(), eval_mode(self.model):
            reference_logp = log_softmax(model_logits(self.model, x), dim=1)

        def objective(x_adv: Tensor) -> Tensor:
            adv_logp = log_softmax(model_logits(self.model, x_adv), dim=1)
            return kl_div(
                adv_logp, reference_logp, log_target=True, reduction="batchmean"
            )

        x_adv = self._inner_pgd(x, objective, eps)
        self._apply_awp_with_depth(x, x_adv, y, depth_target)

        clean_logits, clean_output = self._output(x)
        natural = cross_entropy(clean_logits, y)
        adv_logits, depth_pred = self.model.forward_with_depth(x_adv)
        robust = kl_div(
            log_softmax(adv_logits, dim=1),
            log_softmax(clean_logits, dim=1),
            log_target=True,
            reduction="batchmean",
        )
        loss = (
            natural
            + self.beta * robust
            + self.depth_lambda * self._depth_term(depth_pred, depth_target)
        )
        return loss, clean_output
