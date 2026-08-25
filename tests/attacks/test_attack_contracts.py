"""
Generic contract tests for `AdversarialAttack` implementations.

Every attack in `ATTACKS` below is run through the same checks, so a new
attack only needs to be added to that list to get baseline validation:
does it respect its epsilon radius, stay within the valid pixel range,
leave inputs/model state untouched, and actually perturb the input.

These are *not* a substitute for attack-specific tests (e.g. checking FGSM
moves in the sign-of-gradient direction) -- they just catch the common ways
an attack implementation goes wrong.
"""

import copy

import pytest
import torch

from trustfake.attacks import FGSM, UncertaintyFGSM

EPS = 0.05
CLIP_MIN, CLIP_MAX = 0.0, 1.0

# Add new attacks here to get them covered by every test below.
ATTACKS = [
    FGSM(eps=EPS, clip_min=CLIP_MIN, clip_max=CLIP_MAX),
    UncertaintyFGSM(eps=EPS, clip_min=CLIP_MIN, clip_max=CLIP_MAX),
    # ACE is not implemented yet (trustfake/attacks/ace.py), so it's excluded.
]

ATTACK_IDS = [attack.name for attack in ATTACKS]


@pytest.fixture(params=ATTACKS, ids=ATTACK_IDS)
def attack(request):
    return request.param


def test_stays_within_epsilon_ball(attack, model, inputs, targets):
    """The core guarantee of an L_inf attack: no pixel may move by more than eps."""
    perturbed = attack(model, inputs, targets)

    linf_distance = (perturbed - inputs).abs().max().item()
    assert linf_distance <= attack.eps + 1e-6, (
        f"{attack.name} moved a pixel by {linf_distance}, "
        f"exceeding its epsilon budget of {attack.eps}"
    )


def test_stays_within_valid_range(attack, model, inputs, targets):
    """Perturbed inputs must remain valid, clipped model inputs."""
    perturbed = attack(model, inputs, targets)

    assert perturbed.min().item() >= attack.clip_min - 1e-6
    assert perturbed.max().item() <= attack.clip_max + 1e-6


def test_preserves_shape_and_dtype(attack, model, inputs, targets):
    perturbed = attack(model, inputs, targets)

    assert perturbed.shape == inputs.shape
    assert perturbed.dtype == inputs.dtype


def test_does_not_mutate_inputs(attack, model, inputs, targets):
    """Attacks must return a new tensor, not perturb `inputs` in place."""
    original = inputs.clone()

    attack(model, inputs, targets)

    assert torch.equal(inputs, original)


def test_output_is_detached(attack, model, inputs, targets):
    """Perturbed examples must be plain tensors, safe to feed anywhere
    without dragging the attack's graph (and its memory) along."""
    perturbed = attack(model, inputs, targets)

    assert not perturbed.requires_grad
    assert perturbed.grad_fn is None


def test_restores_model_training_mode(attack, model, inputs, targets):
    """Attacks switch to eval() internally; they must restore whatever mode
    the model was in before returning control."""
    model.train()
    attack(model, inputs, targets)
    assert model.training is True

    model.eval()
    attack(model, inputs, targets)
    assert model.training is False


def test_does_not_change_model_weights(attack, model, inputs, targets):
    """Crafting an adversarial example must never update the model itself."""
    params_before = [p.clone() for p in model.parameters()]

    attack(model, inputs, targets)

    for before, after in zip(params_before, model.parameters(), strict=True):
        assert torch.equal(before, after)


def test_perturbation_is_nonzero(attack, model, inputs, targets):
    """Sanity check that the attack is doing *something* -- with a nonzero
    eps and a model that has real gradients, the output should differ from
    the input for at least some inputs."""
    perturbed = attack(model, inputs, targets)

    assert not torch.equal(perturbed, inputs)


@pytest.mark.parametrize(
    "attack_cls", [FGSM, UncertaintyFGSM], ids=["fgsm", "uncertainty_fgsm"]
)
def test_zero_epsilon_returns_input_unchanged(attack_cls, model, inputs, targets):
    zero_eps_attack = attack_cls(eps=0.0, clip_min=CLIP_MIN, clip_max=CLIP_MAX)

    perturbed = zero_eps_attack(model, inputs, targets)

    assert torch.allclose(perturbed, inputs)


def test_is_deterministic_for_a_fixed_model_and_input(attack, model, inputs, targets):
    """Given the same model and inputs, the attack should be repeatable --
    no hidden randomness (e.g. dropout left in train mode)."""
    model_copy = copy.deepcopy(model)

    first = attack(model, inputs, targets)
    second = attack(model_copy, inputs, targets)

    assert torch.allclose(first, second)
