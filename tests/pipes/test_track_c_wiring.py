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
