"""The configs and the attack registry must agree.

An attack's `name` keys its log directory and its metric prefixes, so two
configs that produce the same name are not two conditions — they are one
condition run twice, with the second silently overwriting the first. Nothing
in a run reports that; the metrics look normal and the table looks complete.
This module is the tripwire.

It also checks the reverse direction: an attack that is exported but has no
config cannot be selected with `+attack=`, so it is dead weight a reader will
assume was evaluated.
"""

from __future__ import annotations

import importlib
import pathlib
from collections import Counter

import pytest
import yaml

from trustfake.attacks import ACE, ParamACE, attack_registry, describe

CONFIG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "configs" / "training"
ATTACK_CONFIGS = sorted((CONFIG_ROOT / "attack").glob("*.yaml"))
CORRUPTION_CONFIGS = sorted((CONFIG_ROOT / "corruption").glob("*.yaml"))


def _instantiate(config_path: pathlib.Path, key: str):
    spec = yaml.safe_load(config_path.read_text())[key]
    module_path, _, class_name = spec["_target_"].rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**{k: v for k, v in spec.items() if k != "_target_"})


def test_config_files_exist():
    assert ATTACK_CONFIGS, "no attack configs found; the glob is wrong"
    assert CORRUPTION_CONFIGS, "no corruption configs found; the glob is wrong"


@pytest.mark.parametrize(
    "config_path", ATTACK_CONFIGS, ids=[p.stem for p in ATTACK_CONFIGS]
)
def test_attack_config_instantiates_and_is_registered(config_path):
    attack = _instantiate(config_path, "attack")
    assert type(attack).__name__ in attack_registry()


@pytest.mark.parametrize(
    "config_path", CORRUPTION_CONFIGS, ids=[p.stem for p in CORRUPTION_CONFIGS]
)
def test_corruption_config_instantiates(config_path):
    corruption = _instantiate(config_path, "corruption")
    assert corruption.name


def test_no_two_attack_configs_share_a_logged_name():
    """The one that bit us: `ace.yaml` and `ace_uint8.yaml` both reported
    `ace`, so the quantised run — a different threat model — would land in the
    same log directory as the unquantised one and overwrite it."""
    names = [_instantiate(p, "attack").name for p in ATTACK_CONFIGS]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"configs sharing a logged name: {duplicates}"


def test_no_two_corruption_configs_share_a_logged_name():
    names = [_instantiate(p, "corruption").name for p in CORRUPTION_CONFIGS]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert not duplicates, f"configs sharing a logged name: {duplicates}"


def test_every_registered_attack_has_a_config():
    """An exported attack with no config cannot be selected with `+attack=`,
    but a reader scanning the registry will assume it was evaluated."""
    configured = {type(_instantiate(p, "attack")).__name__ for p in ATTACK_CONFIGS}
    missing = sorted(set(attack_registry()) - configured)
    assert not missing, f"registered attacks with no config: {missing}"


def test_describe_reflects_the_instance_not_the_class_default():
    """The registry catalogues classes at their defaults; `describe` is what a
    results table should use, because for a parameterised attack the default
    and the instance can differ on exactly the fields that matter."""
    assert attack_registry()["ACE"]["default_name"] == "ace"
    assert describe(ACE(quantize=True))["default_name"] == "ace_uint8"
    assert attack_registry()["ParamACE"]["uses_labels"] is False
    assert describe(ParamACE(omega="true"))["uses_labels"] is True


def test_ace_name_encodes_the_quantisation_threat_model():
    assert ACE(quantize=False).name == "ace"
    assert ACE(quantize=True).name == "ace_uint8"


def test_param_ace_name_separates_the_family_members():
    """(eta, omega) IS the attack for this family, so a sweep over it must
    produce distinct names rather than N runs of `param_ace`."""
    names = {
        ParamACE(eta=eta, omega=omega).name
        for eta in (1, -1)
        for omega in ("true", "pred")
    }
    assert len(names) == 4


@pytest.mark.parametrize(("omega", "expected"), [("true", True), ("pred", False)])
def test_param_ace_reports_label_use_per_instance(omega, expected):
    """Only omega='true' consults the ground truth. Reporting the label-free
    default as label-using overstates the threat model the row was produced
    under -- which is exactly what the registry is for."""
    assert ParamACE(omega=omega).uses_labels is expected


@pytest.mark.parametrize(
    ("eta", "expected"), [(1, "under_confidence"), (-1, "over_confidence")]
)
def test_param_ace_direction_follows_eta(eta, expected):
    """eta fixes the direction across the batch, so it is not `both`."""
    assert str(ParamACE(eta=eta).direction) == expected
