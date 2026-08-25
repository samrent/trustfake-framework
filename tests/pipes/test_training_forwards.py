"""Which forwards are allowed to update BatchNorm, and at what step size.

Both properties here are invisible in a loss curve. An inner PGD run in train
mode still converges; a robust-validation attack using the wrong step size
still logs a number. They only show up as a model that behaves differently at
eval time than the one that was trained, or as a checkpoint selected on the
wrong epoch -- which is why they get their own tests.
"""

import pytest
import torch
import torch.nn as nn

from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes.train import (
    ConfidenceAdversarialTrainingModule,
    ConfidenceRegularisedTrainingModule,
    HybridAdversarialTrainingModule,
    MARTTrainingModule,
    PGDAdversarialTrainingModule,
    StandardTrainingModule,
    TRADESTrainingModule,
)

NUM_CLASSES = 3
STEPS = 10


def _model():
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, NUM_CLASSES),
    )
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


def _bn(model):
    return next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))


# Expected BatchNorm updates per training step == the number of forwards whose
# output actually feeds the loss. Everything else (the k inner-PGD iterates,
# the clean metrics forward, the frozen-prediction read, the attack's
# reference forward) must run in eval mode.
ARMS = [
    ("pgd_at", PGDAdversarialTrainingModule, {"eps": 0.05, "steps": STEPS}, 1),
    ("at_conf", ConfidenceAdversarialTrainingModule, {"eps": 0.05, "steps": STEPS}, 1),
    ("trades", TRADESTrainingModule, {"eps": 0.05, "steps": STEPS, "beta": 6.0}, 2),
    ("mart", MARTTrainingModule, {"eps": 0.05, "steps": STEPS, "beta": 6.0}, 2),
    (
        "at_kl",
        HybridAdversarialTrainingModule,
        {"eps": 0.05, "steps": STEPS, "beta": 6.0},
        2,
    ),
    ("conf_reg", ConfidenceRegularisedTrainingModule, {"lambda_reg": 1.0}, 1),
]


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [(c, k, e) for _, c, k, e in ARMS],
    ids=[name for name, *_ in ARMS],
)
def test_only_training_forwards_update_batchnorm(cls, kwargs, expected):
    """The inner PGD used to run in train mode, so a 10-step adversary made
    12 BatchNorm updates per batch instead of 1 -- every intermediate iterate
    folded into `running_mean`/`running_var`, and every attack forward
    normalised by a perturbed batch's statistics rather than the deployed
    ones. Training converged the whole time."""
    model = _model()
    module = cls(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        **kwargs,
    )
    module.train()
    x, y = _batch()

    module.compute_loss([x, y])

    assert int(_bn(model).num_batches_tracked) == expected


def test_inner_pgd_restores_training_mode():
    model = _model()
    module = PGDAdversarialTrainingModule(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        eps=0.05,
        steps=3,
    )
    module.train()
    x, y = _batch()

    module.compute_loss([x, y])

    assert model.training is True


def test_step_size_follows_the_current_step_count():
    """`alpha` used to be frozen against the constructor's step count, so
    robust validation -- which runs fewer steps -- attacked with the training
    stride. With robust_val_steps=3 against steps=10 the attack could not
    even reach the ball boundary (3 x 0.25 eps), and the logged robust
    accuracy was of a weaker attack than the one being trained against."""
    model = _model()
    module = PGDAdversarialTrainingModule(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        eps=0.1,
        steps=10,
    )
    assert module.alpha == pytest.approx(2.5 * 0.1 / 10)

    module.steps = 3
    assert module.alpha == pytest.approx(2.5 * 0.1 / 3)


def test_explicit_step_size_is_honoured_under_eps_warmup():
    """An explicit alpha= used to be silently discarded whenever eps warm-up
    was on."""
    model = _model()
    module = PGDAdversarialTrainingModule(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        eps=0.1,
        steps=10,
        alpha=0.0001,
        eps_warmup_epochs=5,
    )
    module.trainer = type("T", (), {"current_epoch": 0})()
    assert module.alpha == pytest.approx(0.0001)


def test_robust_validation_attacks_at_a_fixed_epsilon():
    """Under warm-up the attack would otherwise strengthen every epoch while
    the metric kept one name, and a max-mode checkpoint comparing those
    numbers across epochs reliably keeps the earliest, weakest-attack epoch."""
    model = _model()
    module = PGDAdversarialTrainingModule(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        eps=0.1,
        steps=5,
        eps_warmup_epochs=10,
        robust_val_steps=3,
    )
    module.trainer = type("T", (), {"current_epoch": 0})()

    # The ramp moves the TRAINING eps but not the validation attack's.
    assert module.current_eps < module.eps
    assert module.robust_val_eps == pytest.approx(module.eps)
    module.trainer.current_epoch = 9
    assert module.robust_val_eps == pytest.approx(module.eps)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (StandardTrainingModule, {}),
        (ConfidenceRegularisedTrainingModule, {}),
        (PGDAdversarialTrainingModule, {"eps": 0.05, "steps": 3}),
    ],
    ids=["standard", "conf_reg", "pgd_at"],
)
def test_every_arm_can_log_the_robust_selection_metric(cls, kwargs):
    """`standard`, `conf_reg` and `evidential_adversarial` used to accept
    robust_val_steps from the config, drop it silently, and then die on
    "Early stopping conditioned on metric val_robust_accuracy which is not
    available". The classical baselines could be selected on robustness and
    the flagship arms could not, so a comparison between them would have been
    a comparison of selection protocols."""
    model = _model()
    module = cls(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        robust_val_steps=2,
        **kwargs,
    )
    logged = {}
    module.log = lambda name, value, **kw: logged.__setitem__(name, value)
    module.eval()

    module.validation_step(list(_batch()), 0)

    assert "val_robust_accuracy" in logged
