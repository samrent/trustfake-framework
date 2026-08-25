"""Tests for the confidence-targeted defences (at_conf and conf_reg).

These arms exist because the label-axis defences do not address the failure
this harness measures, so the tests check the confidence-axis property
directly rather than checking that training runs. A defence that trains
happily and leaves the confidence ranking untouched is exactly the outcome
these are here to catch.
"""

import lightning as L  # noqa: N812
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.attacks import OverConfidence
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes.train import (
    ConfidenceAdversarialTrainingModule,
    ConfidenceRegularisedTrainingModule,
)

NUM_CLASSES = 3


def _model(seed: int = 0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, NUM_CLASSES))
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=net,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


def _batch(n=32):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    # A label rule the linear model can only partly fit, so errors exist --
    # without errors the confidence penalty has nothing to act on.
    y = (x.flatten(1)[:, :NUM_CLASSES].argmax(1) + (x.mean((1, 2, 3)) > 0.5)) % (
        NUM_CLASSES
    )
    return x, y.long()


def _loader(bs=8):
    x, y = _batch()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _fit(module):
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        inference_mode=False,
    )
    before = [p.clone() for p in module.model.parameters()]
    trainer.fit(module, train_dataloaders=_loader())
    after = list(module.model.parameters())
    return any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.parametrize(
    "make",
    [
        lambda m, o: ConfidenceAdversarialTrainingModule(
            model=m, num_classes=NUM_CLASSES, optimizer=o, eps=0.05, steps=3
        ),
        lambda m, o: ConfidenceRegularisedTrainingModule(
            model=m, num_classes=NUM_CLASSES, optimizer=o, lambda_reg=1.0
        ),
    ],
    ids=["at_conf", "conf_reg"],
)
def test_confidence_arms_train(make):
    m = _model()
    assert _fit(make(m, torch.optim.Adam(m.parameters(), lr=0.01)))


def test_overconfidence_inner_max_raises_confidence_without_moving_the_label():
    """The defining property of the at_conf inner adversary. If the sign is
    inverted it becomes an ordinary (weak) prediction attack, which still
    trains and still produces plausible curves -- so the sign is asserted."""
    m = _model()
    x, y = _batch()
    tm = ConfidenceAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.05,
        steps=10,
    )

    with torch.no_grad():
        _, clean_probs, clean_preds, _ = m(x)
    x_adv = tm._pgd_overconfidence(x, eps=0.05)
    with torch.no_grad():
        _, adv_probs, adv_preds, _ = m(x_adv)

    clean_conf = clean_probs.gather(1, clean_preds[:, None]).squeeze(1)
    adv_conf = adv_probs.gather(1, clean_preds[:, None]).squeeze(1)

    # Confidence in the ORIGINAL prediction goes up, on average and mostly
    # per-sample; and the prediction itself does not move.
    assert adv_conf.mean() > clean_conf.mean()
    assert (adv_conf >= clean_conf - 1e-6).float().mean() > 0.9
    assert torch.equal(adv_preds, clean_preds)


def test_conf_reg_penalises_confidence_on_errors_only():
    """The penalty must be zero when the model is perfect: it is a penalty on
    confident MISTAKES, not a general confidence penalty. A model that is
    right and sure is exactly what we want, and an arm that punishes it would
    trade away accuracy for nothing."""
    m = _model()
    tm = ConfidenceRegularisedTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        lambda_reg=1.0,
    )
    x, _ = _batch()
    with torch.no_grad():
        perfect_labels = m(x)[2].long()  # labels the model already predicts

    loss_perfect, _ = tm.compute_loss([x, perfect_labels])
    ce_only = nn.functional.cross_entropy(m(x)[0], perfect_labels)
    assert loss_perfect.item() == pytest.approx(ce_only.item(), abs=1e-5)


def test_conf_reg_lambda_zero_is_standard_training():
    """The control the arm has to be read against."""
    m = _model()
    x, y = _batch()
    zero = ConfidenceRegularisedTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        lambda_reg=0.0,
    )
    loss, _ = zero.compute_loss([x, y])
    assert loss.item() == pytest.approx(
        nn.functional.cross_entropy(m(x)[0], y).item(), abs=1e-6
    )


def test_conf_reg_lowers_confidence_on_the_errors_it_keeps():
    """The point of the arm: after training, mistakes should carry less
    confidence than they did -- which is what makes them rank below correct
    answers, which is what selective risk reads."""
    x, y = _batch()
    m = _model()
    tm = ConfidenceRegularisedTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        lambda_reg=5.0,
    )

    def error_confidence():
        with torch.no_grad():
            _, probs, preds, _ = m(x)
        wrong = preds.long() != y
        if not wrong.any():
            return None
        return probs.amax(dim=1)[wrong].mean().item()

    before = error_confidence()
    assert before is not None, "model starts perfect; test is vacuous"

    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(60):
        opt.zero_grad()
        loss, _ = tm.compute_loss([x, y])
        loss.backward()
        opt.step()

    after = error_confidence()
    if after is None:
        return  # all errors eliminated, which is also a pass
    assert after < before


def test_at_conf_reduces_the_confidence_attack_s_effect():
    """End-to-end: after at_conf training, the over-confidence attack should
    inflate confidence on wrong answers less than it did before. This is the
    non-adaptive claim -- the attack is fixed, not re-optimised against the
    defended model -- and the docstring says so."""
    x, y = _batch()
    m = _model()
    attack = OverConfidence(eps=0.05, steps=10)

    def confidence_on_errors():
        with torch.no_grad():
            adv = attack(m, x, y)
            _, probs, preds, _ = m(adv)
        wrong = preds.long() != y
        if not wrong.any():
            return None
        return probs.amax(dim=1)[wrong].mean().item()

    before = confidence_on_errors()
    assert before is not None, "no errors to inflate; test is vacuous"

    tm = ConfidenceAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.05,
        steps=5,
    )
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(40):
        opt.zero_grad()
        loss, _ = tm.compute_loss([x, y])
        loss.backward()
        opt.step()

    after = confidence_on_errors()
    if after is None:
        return  # no errors left at all: strictly better
    assert after <= before
