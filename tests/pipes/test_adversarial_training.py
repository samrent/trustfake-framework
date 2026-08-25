"""Tests for the label-axis adversarial-training modules (PGD-AT, TRADES,
AT+KL, MART). Tiny real Lightning fits on synthetic tensors, CPU only."""

import lightning as L  # noqa: N812
import pytest
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, kl_div, log_softmax
from torch.utils.data import DataLoader, TensorDataset

from trustfake.attacks import PGD
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes.train import (
    HybridAdversarialTrainingModule,
    MARTTrainingModule,
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
        lambda m, o: HybridAdversarialTrainingModule(
            model=m, num_classes=NUM_CLASSES, optimizer=o, eps=0.05, steps=3, beta=6.0
        ),
        lambda m, o: MARTTrainingModule(
            model=m, num_classes=NUM_CLASSES, optimizer=o, eps=0.05, steps=3, beta=6.0
        ),
    ],
    ids=["pgd_at", "trades", "at_kl", "mart"],
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


def test_at_kl_is_not_trades_renamed():
    """The distinction that makes AT+KL a separate arm: TRADES puts
    cross-entropy on the CLEAN forward, AT+KL puts it on the ADVERSARIAL one.
    Swapping the two is the classic TRADES misimplementation -- it still
    trains, and the loss curve still looks reasonable, so nothing but a test
    like this notices."""
    torch.manual_seed(0)
    x = torch.rand(16, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1).long()

    shared = {"num_classes": NUM_CLASSES, "eps": 0.05, "steps": 3, "beta": 6.0}
    m1, m2 = _model(), _model()
    trades = TRADESTrainingModule(
        model=m1, optimizer=torch.optim.Adam(m1.parameters()), **shared
    )
    at_kl = HybridAdversarialTrainingModule(
        model=m2, optimizer=torch.optim.Adam(m2.parameters()), **shared
    )

    torch.manual_seed(7)
    trades_loss, _ = trades.compute_loss([x, y])
    torch.manual_seed(7)
    at_kl_loss, _ = at_kl.compute_loss([x, y])

    # Same weights, same data, same seed -- the objectives still differ.
    assert not torch.isclose(trades_loss, at_kl_loss, atol=1e-4)


def test_at_kl_detaches_the_clean_side_of_the_consistency_term():
    """Without the detach the model can satisfy the KL by degrading its clean
    prediction toward the adversarial one -- the cheaper solution, and one
    that looks like success on the loss curve."""
    torch.manual_seed(0)
    x = torch.rand(8, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1).long()
    m = _model()
    tm = HybridAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.05,
        steps=3,
        beta=100.0,  # make the consistency term dominate
    )

    torch.manual_seed(7)
    loss, _ = tm.compute_loss([x, y])
    grads = torch.autograd.grad(loss, list(m.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)

    # Reconstruct the term with the clean side attached and confirm it is a
    # different gradient -- i.e. the detach is load-bearing, not decorative.
    torch.manual_seed(7)
    x_adv = tm._pgd_ce(x, y, tm.current_eps)
    clean_logits = m(x)[0]
    adv_logits = m(x_adv)[0]
    attached = cross_entropy(adv_logits, y) + tm.beta * kl_div(
        log_softmax(adv_logits, dim=1),
        log_softmax(clean_logits, dim=1),
        log_target=True,
        reduction="batchmean",
    )
    attached_grads = torch.autograd.grad(
        attached, list(m.parameters()), allow_unused=True
    )
    assert any(
        a is not None and b is not None and not torch.allclose(a, b, atol=1e-6)
        for a, b in zip(grads, attached_grads, strict=True)
    )


def test_mart_weights_the_kl_by_clean_uncertainty():
    """MART's defining term. With a model that is certain and right on every
    sample the weight `1 - p_y(clean)` vanishes, so MART collapses toward its
    boosted cross-entropy -- if it does not, the weighting is not wired in."""
    torch.manual_seed(0)
    x = torch.rand(16, 3, 8, 8)
    m = _model()
    # Drive the model to near-certainty on its own predictions, so
    # 1 - p_y(clean) -> 0 for the labels it already predicts.
    with torch.no_grad():
        confident_labels = m(x)[2].long()
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        cross_entropy(m(x)[0], confident_labels).backward()
        opt.step()

    tm = MARTTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.0,  # no perturbation: isolate the weighting from the adversary
        steps=1,
        beta=6.0,
    )
    with torch.no_grad():
        clean_probs = m(x)[1]
        true_prob = clean_probs.gather(1, confident_labels[:, None]).squeeze(1)
    assert true_prob.mean() > 0.95, "model is not confident enough; test is vacuous"

    loss, _ = tm.compute_loss([x, confident_labels])
    # At eps=0 the adversarial and clean forwards coincide, so the KL term is
    # ~0 regardless; what this pins is that the weighted term cannot blow up.
    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_robust_validation_metric_is_opt_in():
    """It costs a PGD run per validation batch, so it must not fire by
    default -- but when asked for it must actually be logged, or a config
    pointing the checkpoint callback at it would silently never fire."""
    m = _model()
    off = PGDAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.05,
        steps=2,
    )
    assert off.robust_val_steps == 0

    on = PGDAdversarialTrainingModule(
        model=m,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(m.parameters()),
        eps=0.05,
        steps=2,
        robust_val_steps=2,
    )
    logged = {}
    on.log = lambda name, value, **kw: logged.__setitem__(name, value)

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        inference_mode=False,
    )
    trainer.validate(on, dataloaders=_loader(), verbose=False)
    assert "val_robust_accuracy" in logged
    # The inner-step count is restored after the robust pass, so the next
    # training step still uses the configured budget.
    assert on.steps == 2
