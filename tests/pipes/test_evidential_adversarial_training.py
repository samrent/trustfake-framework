"""End-to-end test of the EV-AT training module on a real (tiny) Lightning
Trainer over synthetic tensors. No dataset, no network, CPU only."""

import lightning as L  # noqa: N812
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.losses import EvidentialLoss
from trustfake.metrics.uncertainty import EvidentialPredictiveEntropy
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper, EvidentialWrapper
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pipes.train import EvidentialAdversarialTrainingModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

NUM_CLASSES = 3


def _evidential_module():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, NUM_CLASSES))
    return EvidentialWrapper(
        normalization_layer=nn.Identity(),
        model=net,
        loss_fn=EvidentialLoss(NUM_CLASSES),
        uncertainty_score=EvidentialPredictiveEntropy(),
    )


def _loader(n=32, bs=8):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1)
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _module(**kwargs):
    m = _evidential_module()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    return EvidentialAdversarialTrainingModule(
        model=m, num_classes=NUM_CLASSES, optimizer=opt, **kwargs
    )


@pytest.mark.parametrize("mode", ["ikl", "kl", "l2"])
def test_evat_training_runs_and_updates_weights(mode):
    tm = _module(divergence_mode=mode, adv_steps=3, beta=1.0)
    before = [p.clone() for p in tm.model.parameters()]
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        inference_mode=False,  # attacks need grad
    )
    trainer.fit(tm, train_dataloaders=_loader())
    after = list(tm.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_evat_updates_ikl_global_stats():
    tm = _module(divergence_mode="ikl", adv_steps=2)
    assert not bool(tm.divergence.stats_ready)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        inference_mode=False,
    )
    trainer.fit(tm, train_dataloaders=_loader())
    assert bool(tm.divergence.stats_ready)
    assert torch.allclose(
        tm.divergence.class_means.sum(1), torch.ones(NUM_CLASSES), atol=1e-4
    )


def test_evat_loss_decreases_over_steps():
    """Overfitting a tiny batch, the EV-AT objective should fall."""
    tm = _module(divergence_mode="ikl", adv_steps=3, beta=1.0)
    tm.model.loss_fn.set_epoch(5)
    x = torch.rand(8, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1)
    opt = torch.optim.Adam(tm.model.parameters(), lr=0.02)
    first, last = None, None
    for step in range(30):
        opt.zero_grad()
        loss, _ = tm.compute_loss([x, y])
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_evat_rejects_non_evidential_model():
    m = BaseWrapper(
        normalization_layer=nn.Identity(),
        model=nn.Linear(4, 3),
        loss_fn=EvidentialLoss(3),
        uncertainty_score=MultiClassMaxProbability(),
    )
    with pytest.raises(TypeError, match="not an evidential wrapper"):
        EvidentialAdversarialTrainingModule(
            model=m,
            num_classes=3,
            optimizer=torch.optim.Adam(m.parameters()),
        )


def test_evat_rejects_non_evidential_loss():
    m = EvidentialWrapper(
        normalization_layer=nn.Identity(),
        model=nn.Linear(4, 3),
        loss_fn=nn.CrossEntropyLoss(),  # not EvidentialLoss
        uncertainty_score=EvidentialPredictiveEntropy(),
    )
    with pytest.raises(TypeError, match="EvidentialLoss"):
        EvidentialAdversarialTrainingModule(
            model=m,
            num_classes=3,
            optimizer=torch.optim.Adam(m.parameters()),
        )


def test_evat_checkpoint_loads_into_eval_module(tmp_path):
    """Regression: the EV-AT training checkpoint must load into the evaluation
    module. The divergence's IKL global-stat buffers are non-persistent, so
    they stay out of the checkpoint and do not break the strict state_dict
    load (the classifier weights are all that eval needs)."""
    tm = _module(divergence_mode="ikl", adv_steps=2)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        inference_mode=False,
    )
    trainer.fit(tm, train_dataloaders=_loader())
    ckpts = list(tmp_path.rglob("*.ckpt"))
    assert ckpts, "no checkpoint written"

    fresh_wrapper = _evidential_module()
    eval_module = ClassificationEvaluationModule.load_from_checkpoint(
        ckpts[0],
        model=fresh_wrapper,
        model_output_schema_cls=ClassificationModelOutput,
        num_classes=NUM_CLASSES,
        attack=None,
    )
    # the loaded model runs
    x = torch.rand(4, 3, 8, 8)
    logits, probs, preds, unc = eval_module.model(x)
    assert probs.shape == (4, NUM_CLASSES)
