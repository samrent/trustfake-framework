"""The eval config wiring: the corruption group is reachable from Hydra, a
run declares exactly one evaluation condition, and every knob `src/test.py`
reads is actually declared.

Config files fail silently in the worst possible way: a wrong `_target_`, a
renamed argument or a key nothing sets surfaces only when a real evaluation
job starts, after the checkpoint has been loaded. Every shipped
`configs/training/corruption/*.yaml` is instantiated here instead, and
`eval_config.yaml` is read for the keys the pipe depends on.
"""

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from trustfake.attacks import FGSM
from trustfake.corruptions import ImageCorruption, JPEGCompression
from trustfake.instantiator import Instantiator, select_evaluation_condition
from trustfake.metrics.moderation import fit_thresholds

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "training"
CORRUPTION_CONFIGS = sorted((CONFIG_ROOT / "corruption").glob("*.yaml"))


def test_the_config_group_is_not_empty():
    assert CORRUPTION_CONFIGS, "no corruption configs found to test"


@pytest.mark.parametrize("config_path", CORRUPTION_CONFIGS, ids=lambda p: p.stem)
def test_shipped_config_instantiates(config_path):
    # Hydra merges a config-group file under a node named after the group,
    # so `cfg.corruption` is the file's content -- which is what
    # `instantiate_corruption` unwraps. Reproduce that nesting rather than
    # testing a shape Hydra never produces.
    cfg = OmegaConf.create({"corruption": OmegaConf.load(config_path)})
    corruption = Instantiator(cfg).instantiate_corruption()

    assert isinstance(corruption, ImageCorruption)
    # The parameter reached the object: a name that lost it would collide
    # with every other rung of the same ladder.
    assert any(character.isdigit() for character in corruption.name)


def test_instantiate_corruption_rejects_a_config_without_the_key():
    from omegaconf.errors import ConfigAttributeError

    cfg = OmegaConf.create({"corruption": {"wrong_key": {"_target_": "builtins.dict"}}})
    with pytest.raises(ConfigAttributeError, match="'corruption' key is missing"):
        Instantiator(cfg).instantiate_corruption()


def test_condition_selection_picks_the_one_that_is_set():
    attack = FGSM()
    corruption = JPEGCompression(quality=40)

    assert select_evaluation_condition({"attack": None, "corruption": None}) is None
    assert select_evaluation_condition({"attack": attack, "corruption": None}) is attack
    selected = select_evaluation_condition({"attack": None, "corruption": corruption})
    assert selected is corruption


def test_condition_selection_refuses_both():
    """One run, one condition: the pipe keys its metrics by the condition's
    name, so configuring both would report one and silently drop the other."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_evaluation_condition(
            {"attack": FGSM(), "corruption": JPEGCompression(quality=40)}
        )


def test_eval_config_declares_the_missed_fake_sla():
    """`fit_thresholds` has supported `sla_missed_fake` all along and nothing
    set it, so the second (asymmetric) SLA was unreachable from a run.
    Residual risk counts both error directions, so a policy can satisfy it
    while auto-allowing most of the fakes -- for a deepfake queue that is the
    error that matters, and it must be settable from the config.
    """
    cfg = OmegaConf.load(CONFIG_ROOT / "eval_config.yaml")

    assert "moderation_sla_missed_fake" in cfg
    # null by default: no missed-fake commitment unless a deployment states
    # one, so existing runs keep their behaviour exactly.
    assert cfg.moderation_sla_missed_fake is None
    assert (
        inspect.signature(fit_thresholds).parameters["sla_missed_fake"].default is None
    )
