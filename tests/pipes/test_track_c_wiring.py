"""Track C's configuration surface: the pairing table, the config files, and
the arm/model/data guard in src/train.py. Nothing here touches data.

The pairing validator is the first line: `wrapper=depth` admits exactly the
three scorings of a checkpoint, and every historical (wrapper, score) refusal
is unchanged. The guard is the second: the three quiet mismatches between a
depth arm, a depth model and a depth datamodule each raise before a
checkpoint tree is created.
"""

from __future__ import annotations

import importlib
import math
import pathlib

import pytest
import torch
import torch.nn as nn
import yaml

from trustfake.instantiator import WRAPPER_UNCERTAINTY_SCORE_TARGETS, ExperimentConfig
from trustfake.metrics.uncertainty import MultiClassMaxProbability
from trustfake.models.torch import resnet18
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes.train import (
    DepthPGDAdversarialTrainingModule,
    DepthStandardTrainingModule,
    PGDAdversarialTrainingModule,
    StandardTrainingModule,
)
from trustfake.pipes.train.depth_auxiliary import check_depth_arm

CONFIG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "configs" / "training"
MSP = "trustfake.metrics.uncertainty.probs.MultiClassMaxProbability"
DEPTH = "trustfake.metrics.uncertainty.depth.DepthConsistencyScore"
COMBINED = "trustfake.metrics.uncertainty.depth.CombinedDepthScore"


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target", [MSP, DEPTH, COMBINED])
def test_depth_wrapper_admits_its_three_scorings(target):
    ExperimentConfig(seed=1, name="x", wrapper="depth", uncertainty_score=target)


def test_depth_wrapper_refuses_other_scores():
    with pytest.raises(ValueError, match="one of"):
        ExperimentConfig(
            seed=1,
            name="x",
            wrapper="depth",
            uncertainty_score="trustfake.metrics.uncertainty.mc_dropout.MCDropoutPredictiveEntropy",
        )


def test_historical_pairings_are_unchanged():
    ExperimentConfig(seed=1, name="x", wrapper="base", uncertainty_score=MSP)
    with pytest.raises(ValueError, match="requires uncertainty_score"):
        ExperimentConfig(seed=1, name="x", wrapper="base", uncertainty_score=DEPTH)
    with pytest.raises(ValueError, match="Unknown wrapper"):
        ExperimentConfig(seed=1, name="x", wrapper="nope", uncertainty_score=MSP)
    assert isinstance(WRAPPER_UNCERTAINTY_SCORE_TARGETS["base"], str)


# --------------------------------------------------------------------------
# Config files
# --------------------------------------------------------------------------


def _instantiate(path: pathlib.Path, key: str, **override):
    spec = yaml.safe_load(path.read_text())[key]
    module_path, _, name = spec["_target_"].rpartition(".")
    cls = getattr(importlib.import_module(module_path), name)
    kwargs = {k: v for k, v in spec.items() if k != "_target_"}
    kwargs.update(override)
    return cls(**kwargs)


@pytest.mark.parametrize("stem", ["depth_consistency", "depth_combined"])
def test_uncertainty_score_configs_instantiate(stem):
    score = _instantiate(
        CONFIG_ROOT / "uncertainty_score" / f"{stem}.yaml", "uncertainty_score"
    )
    assert score is not None


def test_every_uncertainty_score_config_is_admitted_by_some_wrapper():
    """A score yaml nobody can select is dead weight a reader assumes ran."""
    admitted = set()
    for value in WRAPPER_UNCERTAINTY_SCORE_TARGETS.values():
        admitted.update(value if isinstance(value, tuple) else (value,))
    for path in sorted((CONFIG_ROOT / "uncertainty_score").glob("*.yaml")):
        target = yaml.safe_load(path.read_text())["uncertainty_score"]["_target_"]
        assert target in admitted, path.name


def test_depth_model_config_builds_a_head_and_keeps_a_distinct_name():
    spec = yaml.safe_load((CONFIG_ROOT / "model" / "resnet18_depth.yaml").read_text())
    assert spec["name"] == "resnet18_depth"
    model = _instantiate(
        CONFIG_ROOT / "model" / "resnet18_depth.yaml", "model", pretrained=False
    )
    assert model.depth_head is not None
    plain = yaml.safe_load((CONFIG_ROOT / "model" / "resnet18.yaml").read_text())
    assert "depth_head" not in plain["model"]


def test_train_config_defaults_leave_existing_arms_untouched():
    cfg = yaml.safe_load((CONFIG_ROOT / "train_config.yaml").read_text())
    assert cfg["wrapper"] == "base"
    assert cfg["experiment"]["training_pipe"] == "standard"
    assert "depth_lambda" in cfg
    dm = yaml.safe_load((CONFIG_ROOT / "datamodule" / "sid_set.yaml").read_text())
    assert dm["datamodule"]["depth_targets_dir"] is None
    ev = yaml.safe_load((CONFIG_ROOT / "eval_config.yaml").read_text())
    assert ev["wrapper"] == "base" and ev["depth_attack_scoring"] == "white_box"


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


class _DM:
    def __init__(self, depth):
        self.has_depth_targets = depth


def _wrapper(depth_head):
    torch.manual_seed(0)
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=resnet18(num_classes=3, depth_head=depth_head, depth_head_width=8),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


def test_guard_accepts_the_matched_combinations():
    check_depth_arm(
        DepthStandardTrainingModule, _wrapper(True), _DM(True), "standard_depth"
    )
    check_depth_arm(StandardTrainingModule, _wrapper(False), _DM(False), "standard")
    check_depth_arm(PGDAdversarialTrainingModule, _wrapper(False), _DM(False), "pgd_at")


def test_guard_refuses_a_depth_arm_without_targets():
    with pytest.raises(ValueError, match="depth_targets_dir"):
        check_depth_arm(
            DepthPGDAdversarialTrainingModule, _wrapper(True), _DM(False), "x"
        )


def test_guard_refuses_targets_nobody_consumes():
    with pytest.raises(ValueError, match="does not consume"):
        check_depth_arm(StandardTrainingModule, _wrapper(False), _DM(True), "standard")


def test_guard_refuses_an_untrained_head():
    with pytest.raises(ValueError, match="does not train it"):
        check_depth_arm(
            PGDAdversarialTrainingModule, _wrapper(True), _DM(False), "pgd_at"
        )


def _compose(config_name, overrides, monkeypatch, tmp_path):
    from hydra import compose, initialize_config_dir

    for key in ("DATA_PATH", "OUTPUT_PATH", "LOGS_PATH"):
        monkeypatch.setenv(key, str(tmp_path))
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(config_name=config_name, overrides=overrides)


def test_hydra_composes_the_chain_override_lists(monkeypatch, tmp_path):
    """The exact override shapes jobs/track_c_depth.sh uses. A key that is
    not in the config needs a leading plus; getting it wrong is a
    ConfigCompositionException on the box at step zero -- which is where the
    smoke step's limit_train_batches was caught."""
    cfg = _compose(
        "train_config",
        [
            "experiment.name=x",
            "experiment.training_pipe=pgd_at_depth",
            "model=resnet18_depth",
            "datamodule.datamodule.depth_targets_dir=/tmp/store",
            "datamodule.datamodule.profile=smoke",
            "trainer.trainer.max_epochs=1",
            "+trainer.trainer.limit_train_batches=2",
            "+trainer.trainer.limit_val_batches=1",
            "adv_eps=0.03137",
            "adv_steps=7",
            "adv_warmup_epochs=2",
            "depth_lambda=1.0",
            "resume=false",
            "robust_val_steps=3",
        ],
        monkeypatch,
        tmp_path,
    )
    assert cfg.model.name == "resnet18_depth" and cfg.depth_lambda == 1.0
    assert cfg.trainer.trainer.limit_train_batches == 2
    for overrides in (
        [
            "experiment.name=x",
            "model=resnet18_depth",
            "wrapper=depth",
            "uncertainty_score=depth_combined",
            "datamodule.datamodule.profile=smoke",
            "datamodule.datamodule.limit_test=64",
            "depth_teacher_input_size=518",
            "depth_attack_scoring=white_box",
            "+attack=fgsm",
        ],
        [
            "experiment.name=x",
            "model=resnet18_depth",
            "wrapper=depth",
            "uncertainty_score=depth_consistency",
            "datamodule=so_fake_ood",
            "calib_datamodule=sid_set",
            "datamodule.datamodule.limit_test=1000",
            "depth_attack_scoring=transfer",
            "+corruption=jpeg",
        ],
    ):
        cfg = _compose("eval_config", overrides, monkeypatch, tmp_path)
        assert cfg.wrapper == "depth"


def test_calib_depth_pass_fits_the_combined_score_and_writes_the_gate(tmp_path):
    """The helper src/test.py runs between temperature and moderation: the
    combined score comes out fitted, and the sigma-seam gate lands beside the
    metrics as JSON."""
    import json
    import sys

    import lightning as L  # noqa: N812

    from trustfake.depth import FakeDepthTeacher
    from trustfake.metrics.uncertainty import CombinedDepthScore
    from trustfake.models.wrapper import DepthConsistencyWrapper

    sys.argv = ["x"]
    test_module = importlib.import_module("test")
    torch.manual_seed(0)
    wrapper = DepthConsistencyWrapper(
        normalization_layer=nn.Identity(),
        model=resnet18(num_classes=3, depth_head=True, depth_head_width=8),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=CombinedDepthScore(),
        teacher=FakeDepthTeacher(output_size=16, input_size=32, multiple=1),
    )
    eval_module = type("E", (), {"model": wrapper})()
    trainer = type(
        "T", (), {"loggers": [L.pytorch.loggers.CSVLogger(save_dir=str(tmp_path))]}
    )()
    x = torch.rand(12, 3, 32, 32)
    y = torch.randint(0, 3, (12,))
    gate = test_module.calib_depth_pass(
        eval_module, [(x[:6], y[:6]), (x[6:], y[6:])], "cpu", trainer
    )
    assert wrapper.uncertainty_score.fitted
    assert gate["n_calib"] == 12 and gate["residual_finite"]
    written = json.loads(
        (
            tmp_path / "lightning_logs" / "version_0" / "depth_calib_gate.json"
        ).read_text()
    )
    assert written["n_calib"] == 12


def test_spearman_helper():
    import sys

    sys.argv = ["x"]
    test_module = importlib.import_module("test")
    a = torch.arange(10.0)
    assert test_module.spearman_abs(a, a) == pytest.approx(1.0)
    assert test_module.spearman_abs(a, -a) == pytest.approx(1.0)
    torch.manual_seed(0)
    assert test_module.spearman_abs(a, torch.randperm(10).float()) < 1.0
    # a constant score correlates with nothing: NaN, never 0.0 or 1.0
    assert math.isnan(test_module.spearman_abs(a, torch.zeros(10)))
    # ties get their mean rank: two identical vectors with ties still give 1
    tied = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])
    assert test_module.spearman_abs(tied, tied) == pytest.approx(1.0)
