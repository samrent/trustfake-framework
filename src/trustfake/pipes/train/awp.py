"""Adversarial Weight Perturbation (Wu, Xia & Wang, NeurIPS 2020).

AWP is a *modifier*, not a method: it composes with any adversarially-trained
arm rather than replacing one. That is why it lives in a mixin instead of a
`training_pipe` of its own -- the EV-AT ablation reports it as an additional,
non-substitutable gain on top of the evidential loss and REA, which is only a
meaningful claim if the same switch can be flipped on every arm.

The idea: adversarial training flattens the loss surface in *input* space but
can still land in a sharp minimum in *weight* space, and sharp weight minima
generalise badly -- the robust train/test gap. AWP adds an inner maximisation
over the weights themselves, so each step is taken at a deliberately
unfavourable point in weight space and the optimiser is pushed toward a flat
one.

Ordering matters and is easy to get wrong: the gradient is computed at the
perturbed weights ``w + v``, but it is applied to ``w``. So the perturbation
is added during `training_step` and removed in `on_train_batch_end`, which
Lightning runs *after* `optimizer.step()`. Restoring too early makes AWP a
no-op that still costs a forward pass, and nothing in the metrics would say so.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from trustfake.attacks._common import model_logits
from trustfake.pipes.train._common import eval_mode

__all__ = ["AWPMixin"]


class AWPMixin:
    """Weight-space inner maximisation, mixed into a training module.

    Disabled by default (`awp_gamma = 0.0`), so mixing it in changes nothing
    until it is asked for.

    The perturbation is one gradient-ascent step on the arm's own adversarial
    objective, scaled per layer to a fixed *relative* size:

        v = gamma * ||w|| / ||grad|| * grad

    The layer-wise relative scaling is the part that makes AWP stable -- an
    absolute weight budget means something different in every layer, and a
    single gamma tuned against one layer's scale silently does nothing in
    another. Only parameters with more than one dimension are perturbed
    (weight matrices and convolution kernels, not biases or normalisation
    scales), following the reference implementation: perturbing 1-D
    parameters mostly rescales activations and buys nothing.

    Args:
        awp_gamma: Relative weight-perturbation budget. 0 disables AWP.
            0.005-0.01 is the usual range.
        awp_warmup_epochs: Epochs to train before AWP switches on. AWP on a
            model that has not learned the task yet perturbs weights that
            carry no signal.
    """

    def __init__(
        self,
        *args,
        awp_gamma: float = 0.0,
        awp_warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.awp_gamma = awp_gamma
        self.awp_warmup_epochs = awp_warmup_epochs
        self._awp_diff: dict[str, Tensor] | None = None

    @property
    def awp_enabled(self) -> bool:
        return self.awp_gamma > 0.0

    def _awp_active(self) -> bool:
        """AWP applies during training only, and only after the warm-up.

        The `self.training` guard is load-bearing: `validation_step` calls the
        same `compute_loss`, and perturbing weights during validation would
        make the selection metric measure a model that is never saved.
        """
        return (
            self.awp_enabled
            and self.training
            and self.current_epoch >= self.awp_warmup_epochs
        )

    def awp_objective(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> Tensor:
        """The loss AWP ascends in weight space.

        Defaults to cross-entropy on the adversarial batch -- the objective
        the arm is already minimising. An arm whose objective is not
        cross-entropy (EV-AT) overrides this so the weight adversary attacks
        the loss actually being trained, not a proxy for it; `inputs` is the
        clean batch, which such an objective generally needs as its reference.
        """
        return cross_entropy(model_logits(self.model, x_adv), targets.long())

    def apply_awp(self, inputs: Tensor, x_adv: Tensor, targets: Tensor) -> None:
        """Perturb the weights toward the worst case, and remember by how much."""
        if not self._awp_active():
            return

        # A second call before the restore would overwrite the record of the
        # first, stranding that perturbation in the weights forever. Reachable:
        # any optimizer that re-evaluates the closure (LBFGS) calls
        # `compute_loss` several times per `on_train_batch_end`.
        self.restore_awp()

        named = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dim() > 1
        ]
        if not named:
            return

        # The weight adversary must see the deployed function, not a
        # batch-statistics one: in train mode this extra forward would fold
        # another batch into every BatchNorm running estimate.
        with torch.enable_grad(), eval_mode(self.model):
            loss = self.awp_objective(inputs, x_adv, targets)
            grads = torch.autograd.grad(
                loss, [param for _, param in named], allow_unused=True
            )

        diff: dict[str, Tensor] = {}
        with torch.no_grad():
            for (name, param), grad in zip(named, grads, strict=True):
                if grad is None:
                    continue
                scale = self.awp_gamma * param.norm() / grad.norm().clamp_min(1e-20)
                perturbation = scale * grad
                param.add_(perturbation)
                diff[name] = perturbation
        self._awp_diff = diff

    def restore_awp(self) -> None:
        """Undo the weight perturbation. Idempotent."""
        if self._awp_diff is None:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, perturbation in self._awp_diff.items():
                params[name].sub_(perturbation)
        self._awp_diff = None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # After `optimizer.step()`: the gradient was taken at w + v, and is
        # applied to w. See the module docstring.
        self.restore_awp()
        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        # Belt and braces: nothing should reach here with a perturbation
        # outstanding, but if a batch died between perturb and restore the
        # weights are currently w + v, and anything that reads them next --
        # validation, a checkpoint write -- would see the perturbed model.
        self.restore_awp()
        super().on_train_epoch_end()

    def on_exception(self, exception: BaseException) -> None:
        # `ModelCheckpoint(save_on_exception=True)` writes on the way out, so
        # without this the file on disk is the perturbed model rather than the
        # trained one -- a corrupted checkpoint that loads and runs.
        self.restore_awp()
        parent = getattr(super(), "on_exception", None)
        if parent is not None:
            parent(exception)
