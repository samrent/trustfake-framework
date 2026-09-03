"""The multi-task depth arms: what they add, and what they leave alone.

Pinned here because a loss curve cannot show it: the depth term is really
in the loss and really computed on the adversarial input against the clean
target; each arm makes exactly as many BatchNorm updates as its parent; and
the plain arms are untouched by a batch that happens to carry a third
element. Every refusal is a test. CPU only, synthetic tensors.
"""

from __future__ import annotations

import lightning as L  # noqa: N812
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.losses import ssi_l1_per_image
from trustfake.metrics.uncertainty import (
    MCDropoutPredictiveEntropy,
    MultiClassMaxProbability,
)
from trustfake.models.torch import resnet18
from trustfake.models.wrapper import BaseWrapper, MCDropoutWrapper
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pipes.train import (
    DepthPGDAdversarialTrainingModule,
    DepthStandardTrainingModule,
    DepthTRADESTrainingModule,
    PGDAdversarialTrainingModule,
    StandardTrainingModule,
)
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

NUM_CLASSES = 3
STEPS = 3


class _DepthNet(nn.Module):
    """Tiny two-headed net with one BatchNorm in the shared trunk."""

    def __init__(self, with_head=True):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        self.fc = nn.Linear(4 * 8 * 8, NUM_CLASSES)
        self.depth_head = nn.Conv2d(4, 1, 1) if with_head else None

    def forward(self, x):
        return self.fc(self.trunk(x).flatten(1))

    def forward_with_depth(self, x):
        if self.depth_head is None:
            raise ValueError("no head")
        f = self.trunk(x)
        return self.fc(f.flatten(1)), self.depth_head(f)


def _model(with_head=True, wrapper=BaseWrapper, score=None):
    torch.manual_seed(0)
    return wrapper(
        normalization_layer=nn.Identity(),
        model=_DepthNet(with_head),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=score if score is not None else MultiClassMaxProbability(),
    )


def _batch(n=16):
    torch.manual_seed(1)
    x = torch.rand(n, 3, 8, 8)
    y = x.flatten(1)[:, :NUM_CLASSES].argmax(1).long()
    d = torch.rand(n, 1, 8, 8)
    return x, y, d


def _bn(model):
    return next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))


def _module(cls, model=None, **kw):
    model = model if model is not None else _model()
    return cls(
        model=model,
        num_classes=NUM_CLASSES,
        optimizer=torch.optim.Adam(model.parameters()),
        **kw,
    )


ARMS = [
    ("standard_depth", DepthStandardTrainingModule, {"depth_lambda": 0.5}, 1),
    (
        "pgd_at_depth",
        DepthPGDAdversarialTrainingModule,
        {"eps": 0.05, "steps": STEPS, "depth_lambda": 0.5},
        1,
    ),
    (
        "trades_depth",
        DepthTRADESTrainingModule,
        {"eps": 0.05, "steps": STEPS, "beta": 6.0, "depth_lambda": 0.5},
        2,
    ),
]


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [(c, k, e) for _, c, k, e in ARMS],
    ids=[name for name, *_ in ARMS],
)
def test_depth_arms_update_batchnorm_exactly_like_their_parents(cls, kwargs, expected):
    model = _model()
    module = _module(cls, model, **kwargs)
    module.train()
    module.compute_loss(list(_batch()))
    assert int(_bn(model).num_batches_tracked) == expected
    assert model.training is True


def test_standard_depth_loss_is_ce_plus_lambda_times_ssi():
    model = _model()
    module = _module(DepthStandardTrainingModule, model, depth_lambda=0.7)
    module.eval()
    x, y, d = _batch()
    loss, output = module.compute_loss([x, y, d])

    logits, depth = model.forward_with_depth(x)
    expected = (
        nn.functional.cross_entropy(logits, y) + 0.7 * ssi_l1_per_image(depth, d).mean()
    )
    assert torch.allclose(loss, expected, atol=1e-6)
    assert isinstance(output, ClassificationModelOutput)
    assert torch.equal(output.logits, logits)


def test_pgd_depth_supervises_the_adversarial_input_against_the_clean_target(
    monkeypatch,
):
    """THE property of the adversarial variant: the head sees x_adv, the
    target is the clean geometry, and the inner maximisation is untouched."""
    model = _model()
    module = _module(
        DepthPGDAdversarialTrainingModule,
        model,
        eps=0.05,
        steps=STEPS,
        depth_lambda=0.7,
    )
    module.eval()
    x, y, d = _batch()
    torch.manual_seed(5)
    x_adv = (x + 0.05 * torch.randn_like(x).sign()).clamp(0, 1)
    seen = {}

    def fake_inner(inputs, objective, eps):
        seen["eps"] = eps
        return x_adv

    monkeypatch.setattr(module, "_inner_pgd", fake_inner)
    loss, output = module.compute_loss([x, y, d])

    adv_logits, adv_depth = model.forward_with_depth(x_adv)
    expected = (
        nn.functional.cross_entropy(adv_logits, y)
        + 0.7 * ssi_l1_per_image(adv_depth, d).mean()
    )
    assert torch.allclose(loss, expected, atol=1e-6)
    assert seen["eps"] == pytest.approx(0.05)
    # metrics on the CLEAN forward, as in pgd_at
    assert torch.equal(output.logits, model(x)[0])
    # not the clean-image depth term
    _, clean_depth = model.forward_with_depth(x)
    wrong = (
        nn.functional.cross_entropy(adv_logits, y)
        + 0.7 * ssi_l1_per_image(clean_depth, d).mean()
    )
    assert not torch.allclose(loss, wrong)


def test_trades_depth_puts_the_depth_term_on_the_adversarial_forward(monkeypatch):
    model = _model()
    module = _module(
        DepthTRADESTrainingModule,
        model,
        eps=0.05,
        steps=STEPS,
        beta=2.0,
        depth_lambda=0.3,
    )
    module.eval()
    x, y, d = _batch()
    x_adv = (x + 0.05).clamp(0, 1)
    monkeypatch.setattr(module, "_inner_pgd", lambda inputs, objective, eps: x_adv)
    loss, _ = module.compute_loss([x, y, d])

    clean = model(x)[0]
    adv_logits, adv_depth = model.forward_with_depth(x_adv)
    kl = nn.functional.kl_div(
        adv_logits.log_softmax(1),
        clean.log_softmax(1),
        log_target=True,
        reduction="batchmean",
    )
    expected = (
        nn.functional.cross_entropy(clean, y)
        + 2.0 * kl
        + 0.3 * ssi_l1_per_image(adv_depth, d).mean()
    )
    assert torch.allclose(loss, expected, atol=1e-5)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_batch_without_a_depth_target_is_refused():
    module = _module(DepthStandardTrainingModule)
    x, y, _ = _batch()
    with pytest.raises(ValueError, match="depth_targets_dir"):
        module.compute_loss([x, y])


def test_a_model_without_a_head_is_refused_at_construction():
    with pytest.raises(ValueError, match="depth head"):
        _module(DepthStandardTrainingModule, _model(with_head=False))


def test_mc_dropout_is_refused():
    model = _model(wrapper=MCDropoutWrapper, score=MCDropoutPredictiveEntropy())
    with pytest.raises(ValueError, match="MC dropout"):
        _module(DepthStandardTrainingModule, model)


def test_a_depth_scoring_wrapper_is_refused_for_training():
    """Every guard passed and the first batch crashed on outputs_from_logits
    returning None; now it is refused at construction."""
    from trustfake.depth import FakeDepthTeacher
    from trustfake.metrics.uncertainty import DepthConsistencyScore
    from trustfake.models.wrapper import DepthConsistencyWrapper

    torch.manual_seed(0)
    model = DepthConsistencyWrapper(
        normalization_layer=nn.Identity(),
        model=_DepthNet(),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=DepthConsistencyScore(),
        teacher=FakeDepthTeacher(output_size=8, input_size=8, multiple=1),
    )
    with pytest.raises(ValueError, match="evaluation-time instrument"):
        _module(DepthStandardTrainingModule, model)


def test_non_positive_lambda_is_refused():
    with pytest.raises(ValueError, match="depth_lambda"):
        _module(DepthStandardTrainingModule, depth_lambda=0.0)
    with pytest.raises(ValueError, match="depth_lambda"):
        _module(DepthPGDAdversarialTrainingModule, eps=0.05, steps=2, depth_lambda=-1)


# --------------------------------------------------------------------------
# The parents are untouched
# --------------------------------------------------------------------------


def test_plain_arms_ignore_a_third_batch_element(monkeypatch):
    x, y, d = _batch()
    model = _model()
    std = _module(StandardTrainingModule, model)
    std.eval()
    a, _ = std.compute_loss([x, y])
    b, _ = std.compute_loss([x, y, d])
    assert torch.equal(a, b)

    at = _module(PGDAdversarialTrainingModule, model, eps=0.05, steps=2)
    at.eval()
    monkeypatch.setattr(at, "_inner_pgd", lambda inputs, objective, eps: x)
    a, _ = at.compute_loss([x, y])
    b, _ = at.compute_loss([x, y, d])
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# AWP ascends the trained loss
# --------------------------------------------------------------------------


def test_awp_perturbs_the_head_and_gamma_zero_is_a_no_op():
    """With awp_gamma > 0 the weight adversary's objective includes the depth
    term, so the head's weights are in the perturbation; at gamma 0 every
    weight is bit-identical after compute_loss."""
    for gamma, expect_head in ((0.01, True), (0.0, False)):
        model = _model()
        module = _module(
            DepthPGDAdversarialTrainingModule,
            model,
            eps=0.05,
            steps=2,
            depth_lambda=0.5,
            awp_gamma=gamma,
        )
        module.trainer = type("T", (), {"current_epoch": 0})()
        module.log = lambda *args, **kwargs: None
        module.train()
        # parameters only: a train-mode forward updates BatchNorm buffers
        before = {k: v.clone() for k, v in model.named_parameters()}
        module.compute_loss(list(_batch()))
        diff = module._awp_diff or {}
        assert any("depth_head" in k for k in diff) is expect_head
        if not expect_head:
            after = dict(model.named_parameters())
            assert all(torch.equal(before[k], after[k]) for k in before)
        module.restore_awp()


# --------------------------------------------------------------------------
# Logging and a real fit
# --------------------------------------------------------------------------


def test_depth_term_is_logged_under_the_right_split():
    module = _module(DepthStandardTrainingModule)
    module.trainer = type("T", (), {"current_epoch": 0})()
    logged = {}
    module.log = lambda name, value, **kw: logged.__setitem__(name, value)
    module.train()
    module.compute_loss(list(_batch()))
    assert "train_depth_loss" in logged
    module.eval()
    module.validation_step(list(_batch()), 0)
    assert "val_depth_loss" in logged and "val_loss" in logged


def _loader(n=32, bs=8):
    x, y, d = _batch(n)
    return DataLoader(TensorDataset(x, y, d), batch_size=bs)


def _fit(tm, loader=None):
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
    trainer.fit(tm, train_dataloaders=loader if loader is not None else _loader())
    after = list(tm.model.parameters())
    return any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [(c, k) for _, c, k, _ in ARMS],
    ids=[name for name, *_ in ARMS],
)
def test_depth_arms_train_the_head(cls, kwargs):
    model = _model()
    head_before = model.model.depth_head.weight.clone()
    assert _fit(_module(cls, model, **kwargs))
    assert not torch.equal(head_before, model.model.depth_head.weight)


def test_depth_checkpoint_loads_only_into_a_depth_module(tmp_path):
    """End to end with the real backbone: a checkpoint from standard_depth
    loads strictly into an evaluation module built with model=resnet18_depth
    and refuses the plain resnet18 -- the contract src/test.py relies on."""

    def wrapper(depth_head):
        torch.manual_seed(0)
        return BaseWrapper(
            normalization_layer=nn.Identity(),
            model=resnet18(
                num_classes=NUM_CLASSES, depth_head=depth_head, depth_head_width=8
            ),
            loss_fn=nn.CrossEntropyLoss(),
            uncertainty_score=MultiClassMaxProbability(),
        )

    torch.manual_seed(1)
    x = torch.rand(16, 3, 32, 32)
    y = torch.randint(0, NUM_CLASSES, (16,))
    d = torch.rand(16, 1, 16, 16)
    loader = DataLoader(TensorDataset(x, y, d), batch_size=8)
    tm = _module(DepthStandardTrainingModule, wrapper(True), depth_lambda=0.5)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        inference_mode=False,
    )
    trainer.fit(tm, train_dataloaders=loader)
    ckpt = next(p for p in tmp_path.rglob("*.ckpt"))

    common = {
        "model_output_schema_cls": ClassificationModelOutput,
        "num_classes": NUM_CLASSES,
        "attack": None,
    }
    loaded = ClassificationEvaluationModule.load_from_checkpoint(
        ckpt, model=wrapper(True), **common
    )
    assert torch.equal(loaded.model.model.fc.weight, tm.model.model.fc.weight)
    with pytest.raises(RuntimeError, match="depth_head"):
        ClassificationEvaluationModule.load_from_checkpoint(
            ckpt, model=wrapper(False), **common
        )
