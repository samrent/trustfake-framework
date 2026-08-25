"""The attack taxonomy: family, direction, label use, norm.

This is metadata on the attack rather than prose in a table because a results
table has to be groupable by threat family, and because "which rows used
ground-truth labels" is a threat-model question a reader cannot answer from a
number. A hand-maintained table drifts from the code; this one is built by
instantiating every exported attack, so it cannot.
"""

import pytest

from trustfake.attacks import (
    ACE,
    PGD,
    PGDL2,
    AdaptiveAutoAttack,
    AdversarialAttack,
    AttackDirection,
    AttackFamily,
    BrendelBethge,
    DeepFool,
    OverConfidence,
    UnderConfidence,
    attack_registry,
)

REGISTRY = attack_registry()


def test_every_exported_attack_is_registered():
    import trustfake.attacks as attacks

    exported = {
        getattr(attacks, name)
        for name in attacks.__all__
        if isinstance(getattr(attacks, name), type)
        and issubclass(getattr(attacks, name), AdversarialAttack)
        and getattr(attacks, name) is not AdversarialAttack
    }
    assert len(REGISTRY) == len(exported)


def test_default_attack_names_are_unique():
    """Names key the logged metric prefixes and the storage files; a
    collision silently merges two conditions into one row. Parameterised
    variants are covered in test_config_registry.py, which checks the names
    the CONFIGS actually produce."""
    import trustfake.attacks as attacks

    names = [
        getattr(attacks, name)().name
        for name in attacks.__all__
        if isinstance(getattr(attacks, name), type)
        and issubclass(getattr(attacks, name), AdversarialAttack)
        and getattr(attacks, name) is not AdversarialAttack
    ]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("class_name", "family"),
    [
        ("ACE", AttackFamily.CONFIDENCE),
        ("ParamACE", AttackFamily.CONFIDENCE),
        ("OverConfidence", AttackFamily.CONFIDENCE),
        ("UnderConfidence", AttackFamily.CONFIDENCE),
        ("UncertaintyFGSM", AttackFamily.UNCERTAINTY),
        ("EvidenceTargetedPGD", AttackFamily.EVIDENCE),
        ("FGSM", AttackFamily.PREDICTION),
        ("PGD", AttackFamily.PREDICTION),
        ("AdaptiveAutoAttack", AttackFamily.PREDICTION),
        ("BrendelBethge", AttackFamily.PREDICTION),
        ("PDPGD", AttackFamily.PREDICTION),
    ],
)
def test_family_assignment(class_name, family):
    assert REGISTRY[class_name]["family"] == str(family)


@pytest.mark.parametrize(
    ("attack", "direction"),
    [
        (OverConfidence(), AttackDirection.OVER),
        (UnderConfidence(), AttackDirection.UNDER),
        (ACE(), AttackDirection.BOTH),
        (PGD(), AttackDirection.NONE),
    ],
)
def test_confidence_direction(attack, direction):
    assert attack.direction == direction


def test_minimum_norm_attacks_are_flagged():
    """`eps` means a cap for these and a budget for the others. Reading a
    min-norm row as if eps were the search budget understates the attack."""
    min_norm = {n for n, meta in REGISTRY.items() if meta["minimum_norm"]}
    assert min_norm == {"DeepFool", "CarliniWagner", "BrendelBethge", "PDPGD", "FAB"}


def test_l2_attacks_are_flagged():
    """Robustness does not transfer between norms, so a table that does not
    carry the norm is not comparable row to row."""
    assert REGISTRY["PGDL2"]["norm"] == "l2"
    assert BrendelBethge.norm == "l2"
    assert DeepFool.norm == "l2"
    assert REGISTRY["PGD"]["norm"] == "linf"


def test_label_use_is_opt_in_for_the_prediction_attacks():
    """An attacker holding the ground truth is a strictly stronger threat
    model than one who does not. The default is the realisable one."""
    assert PGD().uses_labels is False
    assert PGD(use_labels=True).uses_labels is True
    assert PGDL2().uses_labels is False
    assert AdaptiveAutoAttack().uses_labels is False
    # ACE is the exception: ground truth is what selects its direction.
    assert ACE().uses_labels is True


def test_registry_reports_label_use_from_the_instance_defaults():
    """The registry describes the DEFAULT configuration of each attack, which
    is what a reader assumes when a table names an attack without a footnote."""
    assert REGISTRY["PGD"]["uses_labels"] is False
    assert REGISTRY["ACE"]["uses_labels"] is True


def test_use_labels_actually_changes_the_attack(model, inputs, targets):
    """The flag has to reach the gradient, not just the metadata."""
    import torch

    with torch.no_grad():
        clean_preds = model(inputs)[2]
    # Choose targets that disagree with the model, so the two modes differ.
    wrong = (clean_preds + 1) % 4

    free = PGD(eps=0.05, steps=5)(model, inputs, wrong)
    supervised = PGD(eps=0.05, steps=5, use_labels=True)(model, inputs, wrong)

    assert not torch.allclose(free, supervised)
