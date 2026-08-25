"""Shared helpers for the training pipes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn

__all__ = ["eval_mode", "RobustValidationMixin"]


@contextmanager
def eval_mode(module: nn.Module) -> Iterator[nn.Module]:
    """Run a block with `module` in eval mode, restoring the previous mode.

    Every attack in `trustfake.attacks` does this around its inner loop, and
    the training pipes have to as well. An inner PGD run in train mode folds
    each of its k intermediate iterates into every BatchNorm running estimate,
    so a k-step adversary makes k+1 times as many updates to `running_mean`
    and `running_var` as there were batches -- and each of the attack's own
    forwards then normalises by the statistics of a perturbed batch rather
    than by the deployed ones. The model that gets evaluated is not the model
    that was trained against.

    It is silent: training converges, the loss curve looks ordinary, and only
    `num_batches_tracked` (or a careful eval-mode comparison) shows it. This
    is the framework's default backbone -- ResNet-18 -- so it is live, not
    theoretical.
    """
    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        module.train(was_training)


class RobustValidationMixin:
    """Log `val_robust_accuracy`, so a defence arm can be SELECTED on it.

    Selecting a defence on clean macro-F1 selects it on exactly what it
    trades away. This mixin sits on every pipe rather than only on the
    adversarially-trained ones, because the comparison is the point: if
    `standard`, `conf_reg` and `evidential_adversarial` cannot log the metric
    while `pgd_at` and `trades` can, then those arms get selected on clean F1
    while their baselines get selected on robustness, and the resulting table
    compares selection protocols rather than methods. (Before this existed
    they accepted `robust_val_steps` from the config, dropped it silently, and
    the run then died on `Early stopping conditioned on metric
    val_robust_accuracy which is not available`.)

    The attack is `trustfake.attacks.PGD`, not a bespoke loop: it already
    runs the model in eval mode, seeds its random start, and derives its step
    size from the step count it is actually given.

    `robust_val_eps` is FIXED, never the warm-up ramp. Under a ramp the attack
    strengthens each epoch while the metric keeps one name, and a
    `mode: max` checkpoint comparing those numbers across epochs reliably
    keeps the earliest one -- on a frozen model (lr=0) the metric fell
    1.0000 -> 0.0352 across five epochs purely from eps ramping. A selection
    metric has to be comparable across the epochs it selects between.

    Args:
        robust_val_steps: PGD steps per validation batch. 0 disables it,
            which is the default because it costs a full attack per batch.
        robust_val_eps: L_inf budget. None falls back to the arm's own `eps`
            when it has one, else 8/255.
    """

    def __init__(
        self,
        *args,
        robust_val_steps: int = 0,
        robust_val_eps: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.robust_val_steps = robust_val_steps
        self._robust_val_eps = robust_val_eps

    @property
    def robust_val_eps(self) -> float:
        if self._robust_val_eps is not None:
            return self._robust_val_eps
        return float(getattr(self, "eps", 8 / 255))

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        if self.robust_val_steps <= 0:
            return loss

        import torch

        from trustfake.attacks import PGD

        x, y = batch[0], batch[1].long()
        attack = PGD(
            eps=self.robust_val_eps, steps=self.robust_val_steps, use_labels=True
        )
        with torch.enable_grad():
            x_adv = attack(self.model, x, y)
        with torch.no_grad():
            preds = self.model(x_adv)[2]
        self.log(
            "val_robust_accuracy",
            (preds.long() == y).float().mean(),
            on_epoch=True,
            prog_bar=False,
        )
        return loss
