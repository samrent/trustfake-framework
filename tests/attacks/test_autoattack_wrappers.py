"""Contract tests for the autoattack-package wrappers.

These are separated from the main battery because the fra31 package is
slower and pulls a third-party dependency; if it cannot be imported the
whole module skips rather than failing. Iterations/queries are cut hard so
the suite stays fast -- these tests check the framework contract (eps-ball,
range, no input mutation, determinism, plumbing), not attack strength.
"""

import pytest
import torch

pytest.importorskip("autoattack")

from trustfake.attacks import APGD, FAB, AutoAttackLinf, SquareAttack  # noqa: E402

EPS = 0.05
CLIP_MIN, CLIP_MAX = 0.0, 1.0

ATTACKS = [
    APGD(eps=EPS, n_iter=5, seed=0),
    FAB(eps=EPS, n_iter=5, seed=0),
    SquareAttack(eps=EPS, n_queries=50, seed=0),
    AutoAttackLinf(eps=EPS, n_iter=5, n_queries=50, seed=0),
]
ATTACK_IDS = [a.name for a in ATTACKS]


@pytest.fixture(params=ATTACKS, ids=ATTACK_IDS)
def attack(request):
    return request.param


def test_stays_within_epsilon_ball(attack, model, inputs, targets):
    perturbed = attack(model, inputs, targets)
    assert (perturbed - inputs).abs().max().item() <= attack.eps + 1e-6


def test_stays_within_valid_range(attack, model, inputs, targets):
    perturbed = attack(model, inputs, targets)
    assert perturbed.min().item() >= attack.clip_min - 1e-6
    assert perturbed.max().item() <= attack.clip_max + 1e-6


def test_preserves_shape_and_does_not_mutate_inputs(attack, model, inputs, targets):
    original = inputs.clone()
    perturbed = attack(model, inputs, targets)
    assert perturbed.shape == inputs.shape
    assert torch.equal(inputs, original)


def test_output_is_detached(attack, model, inputs, targets):
    perturbed = attack(model, inputs, targets)
    assert not perturbed.requires_grad


def test_restores_model_training_mode(attack, model, inputs, targets):
    model.train()
    attack(model, inputs, targets)
    assert model.training is True


@pytest.mark.parametrize("attack_cls", [APGD, FAB, SquareAttack, AutoAttackLinf])
def test_zero_epsilon_returns_input_unchanged(attack_cls, model, inputs, targets):
    perturbed = attack_cls(eps=0.0)(model, inputs, targets)
    assert torch.allclose(perturbed, inputs)


def test_is_deterministic_for_a_fixed_seed(attack, model, inputs, targets):
    """The randomised components (APGD start, Square queries, FAB restarts)
    are seeded, so two runs on the same model and input must match."""
    import copy

    first = attack(model, inputs, targets)
    second = attack(copy.deepcopy(model), inputs, targets)
    assert torch.allclose(first, second)
