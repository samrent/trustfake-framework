"""Tests for Adversarial Weight Perturbation as a composable modifier.

The failure mode AWP invites is silence: perturb the weights, restore them at
the wrong moment, and the run still trains, still converges, and produces
numbers that look like AWP without any AWP in them. Every test here pins one
of the properties that would break in that case.
"""

import lightning as L  # noqa: N812
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes.train import (
    ConfidenceAdversarialTrainingModule,
    MARTTrainingModule,
    PGDAdversarialTrainingModule,
    TRADESTrainingModule,
)

NUM_CLASSES = 3
ARMS = {
    "pgd_at": PGDAdversarialTrainingModule,
    "trades": TRADESTrainingModule,
    "mart": MARTTrainingModule,
    "at_conf": ConfidenceAdversarialTrainingModule,
}


def _model(seed: int = 0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, NUM_CLASSES))
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=net,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


def _batch(n=16):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    return x, x.flatten(1)[:, :NUM_CLASSES].argmax(1).long()


def _module(cls, model, **kwargs):
    kwargs.setdefault("eps", 0.05)
    kwargs.setdefault("steps", 3)
    return cls(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        **kwargs,
    )


@pytest.mark.parametrize("name", list(ARMS), ids=list(ARMS))
def test_awp_composes_with_every_adversarial_arm(name):
    """AWP is a modifier, not a method: it has to attach to all of them."""
    model = _model()
    module = _module(ARMS[name], model, awp_gamma=0.01)
    module.train()
    x, y = _batch()

    module.compute_loss([x, y])

    assert module._awp_diff is not None
    assert module._awp_diff, "no parameter was perturbed"


def test_gamma_zero_is_a_true_no_op():
    """The default. Not 'a very small perturbation' -- none at all, so a run
    with AWP off is bit-identical to one built before AWP existed."""
    model = _model()
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=0.0)
    module.train()
    x, y = _batch()

    before = [p.clone() for p in model.parameters()]
    module.compute_loss([x, y])

    assert module._awp_diff is None
    for a, b in zip(before, model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_weights_are_perturbed_during_the_step_and_restored_after():
    """The ordering the whole method depends on: the gradient is taken at
    w + v, and applied to w. Restoring early makes AWP a no-op that still
    costs a forward pass; never restoring corrupts the weights permanently."""
    model = _model()
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=0.05)
    module.train()
    x, y = _batch()

    before = [p.clone() for p in model.parameters()]
    module.compute_loss([x, y])

    # Mid-step: weights are somewhere else.
    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )

    module.on_train_batch_end(outputs=None, batch=[x, y], batch_idx=0)

    # After the (simulated) optimizer step: exactly back where we started.
    for a, b in zip(before, model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_restore_is_idempotent():
    """`on_train_batch_end` can fire when no perturbation is outstanding
    (gamma=0, or a validation batch). A second restore must not subtract the
    perturbation twice."""
    model = _model()
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=0.05)
    module.train()
    x, y = _batch()

    before = [p.clone() for p in model.parameters()]
    module.compute_loss([x, y])
    module.restore_awp()
    module.restore_awp()

    for a, b in zip(before, model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_awp_does_not_fire_during_validation():
    """`validation_step` runs the same `compute_loss`. Perturbing there would
    make the selection metric describe a model that is never saved."""
    model = _model()
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=0.05)
    module.eval()
    x, y = _batch()

    before = [p.clone() for p in model.parameters()]
    module.compute_loss([x, y])

    assert module._awp_diff is None
    for a, b in zip(before, model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_awp_warmup_delays_the_perturbation():
    """AWP on a model that has not learned the task yet perturbs weights that
    carry no signal."""
    model = _model()
    module = _module(
        PGDAdversarialTrainingModule, model, awp_gamma=0.05, awp_warmup_epochs=3
    )
    module.train()
    module.trainer = type("T", (), {"current_epoch": 0})()
    x, y = _batch()

    module.compute_loss([x, y])
    assert module._awp_diff is None

    module.trainer.current_epoch = 3
    module.compute_loss([x, y])
    assert module._awp_diff is not None


def test_only_multi_dimensional_parameters_are_perturbed():
    """Biases and normalisation scales are left alone, per the reference
    implementation -- perturbing them mostly rescales activations."""
    model = _model()
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=0.05)
    module.train()
    x, y = _batch()

    module.compute_loss([x, y])

    perturbed = set(module._awp_diff)
    for name, param in model.named_parameters():
        if param.dim() > 1:
            assert name in perturbed
        else:
            assert name not in perturbed


def test_awp_perturbation_respects_the_relative_budget():
    """The layer-wise relative scaling is what makes gamma mean the same
    thing in every layer. ||v|| should be gamma * ||w||, per parameter."""
    model = _model()
    gamma = 0.02
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=gamma)
    module.train()
    x, y = _batch()

    params = dict(model.named_parameters())
    before = {n: p.clone() for n, p in params.items()}
    module.compute_loss([x, y])

    for name, perturbation in module._awp_diff.items():
        expected = gamma * before[name].norm()
        assert perturbation.norm().item() == pytest.approx(expected.item(), rel=1e-4)


def test_awp_training_still_converges():
    """A modifier that breaks training is not a modifier."""
    model = _model()
    module = _module(PGDAdversarialTrainingModule, model, awp_gamma=0.01)
    x, y = _batch(32)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        inference_mode=False,
    )
    before = [p.clone() for p in model.parameters()]
    trainer.fit(module, train_dataloaders=loader)

    assert any(
        not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )
    # And no perturbation is left outstanding at the end of the run.
    assert module._awp_diff is None
