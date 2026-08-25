"""Tests for the PGD-AT and TRADES adversarial-training modules. Tiny real
Lightning fits on synthetic tensors, CPU only."""

import lightning as L  # noqa: N812
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.attacks import PGD
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes.train import (
    PGDAdversarialTrainingModule,
    TRADESTrainingModule,
)

NUM_CLASSES = 3


def _model():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, NUM_CLASSES))
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=net,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


def _loader(n=32, bs=8):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1)
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _fit(tm):
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        inference_mode=False,
    )
    before = [p.clone() for p in tm.model.parameters()]
    trainer.fit(tm, train_dataloaders=_loader())
    after = list(tm.model.parameters())
    return any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.parametrize(
    "make",
    [
        lambda m, o: PGDAdversarialTrainingModule(
            model=m, num_classes=NUM_CLASSES, optimizer=o, eps=0.05, steps=3
        ),
        lambda m, o: TRADESTrainingModule(
            model=m, num_classes=NUM_CLASSES, optimizer=o, eps=0.05, steps=3, beta=6.0
        ),
    ],
    ids=["pgd_at", "trades"],
)
def test_at_training_updates_weights(make):
    m = _model()
    tm = make(m, torch.optim.Adam(m.parameters(), lr=0.01))
    assert _fit(tm)


def test_eps_warmup_ramps():
    m = _model()
    tm = PGDAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.1,
        steps=3,
        eps_warmup_epochs=10,
    )
    tm._current_epoch = 0
    # LightningModule.current_epoch reads the trainer; emulate via monkeypatch
    tm.trainer = type("T", (), {"current_epoch": 0})()
    assert tm.current_eps == pytest.approx(0.1 * 1 / 10)
    tm.trainer.current_epoch = 4
    assert tm.current_eps == pytest.approx(0.1 * 5 / 10)
    tm.trainer.current_epoch = 50
    assert tm.current_eps == pytest.approx(0.1)  # capped


def test_pgd_at_improves_robust_loss():
    """Overfitting a batch with PGD-AT should reduce the loss on PGD-perturbed
    inputs -- the point of adversarial training."""
    torch.manual_seed(0)
    m = _model()
    x = torch.rand(16, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1)
    tm = PGDAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.05,
        steps=5,
    )
    attack = PGD(eps=0.05, steps=10)

    def robust_loss():
        with torch.no_grad():
            adv = attack(m, x, y)
            return nn.functional.cross_entropy(m(adv)[0], y).item()

    first = robust_loss()
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(40):
        opt.zero_grad()
        loss, _ = tm.compute_loss([x, y])
        loss.backward()
        opt.step()
    assert robust_loss() < first
