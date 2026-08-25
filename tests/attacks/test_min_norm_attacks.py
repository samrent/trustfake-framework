"""Behaviour specific to the minimum-norm attacks (BB and PDPGD).

The generic contract battery checks that an attack stays inside its budget.
For a minimum-norm attack that is the least interesting property: the budget
is a cap on the answer, not the search, and the number the attack exists to
produce is the norm it needed. These tests check the properties that make
that number trustworthy -- that it is actually minimised, that a failure is
reported as a failure rather than as a tiny perturbation, and that the
closed-form trust-region step solves the problem it claims to solve.
"""

import pytest
import torch

from trustfake.attacks import PDPGD, PGD, PGDL2, BrendelBethge, CarliniWagner, DeepFool
from trustfake.attacks._common import class_margin, project_l1_ball

BIG_EPS = 100.0  # effectively uncapped: let the attack find its own norm

MIN_NORM_ATTACKS = (DeepFool, CarliniWagner, BrendelBethge, PDPGD)


def _preds(model, x):
    with torch.no_grad():
        return model(x)[2]


def test_bb_finds_a_smaller_perturbation_than_it_started_from(model, inputs):
    """The whole algorithm is 'start adversarial, walk back toward the clean
    input'. If the walk does nothing, BB degenerates to its initialisation --
    a random adversarial image -- and would still pass every generic contract
    test while reporting a meaningless norm."""
    attack = BrendelBethge(eps=BIG_EPS, steps=40)

    start, found = attack._starting_points(model, inputs, _preds(model, inputs))
    result = attack.run(model, inputs)

    start_norm = (start - inputs).flatten(1).norm(dim=1)
    final_norm = result.l2_norm
    assert found.any(), "no adversarial starting point found; test is vacuous"
    # Never worse (a step is accepted only if it moves closer), and strictly
    # better on most samples. Not every sample improves: a start already at
    # the boundary has nowhere to go, and the attack correctly leaves it.
    assert (final_norm[found] <= start_norm[found] + 1e-6).all()
    assert (final_norm[found] < start_norm[found]).float().mean() > 0.5


def test_bb_success_is_reported_not_assumed(model, inputs):
    """A sample the attack could not push across the boundary must come back
    unperturbed AND flagged. Silently returning the clean image with
    success=True would read as 'the attack succeeded with zero perturbation',
    i.e. the model has no robustness at all."""
    attack = BrendelBethge(eps=BIG_EPS, steps=20)
    result = attack.run(model, inputs)

    clean_preds = _preds(model, inputs)
    with torch.no_grad():
        adv_preds = model(result.perturbed)[2]
    assert torch.equal(result.success, adv_preds != clean_preds)


def test_bb_tiny_eps_cap_reports_failure(model, inputs):
    """Capping below the true minimum norm cannot produce a success. The cap
    is applied before success is measured, so a flip that only survives
    outside the reported budget is not counted inside it.

    The cap is checked with a float32-sized tolerance rather than an exact
    one: the projection is a round trip through `inputs + delta`, and around
    a pixel value of ~0.5 the representation error alone is ~6e-8 per
    element, which accumulates over the flattened dimension.
    """
    cap = 1e-4
    result = BrendelBethge(eps=cap, steps=10).run(model, inputs)
    assert not result.success.any()
    assert (result.l2_norm <= cap * 1.01).all()


def test_bb_trust_region_step_solves_its_constraints():
    """The closed form must satisfy both constraints exactly: land on the
    hyperplane, and stay inside the radius. This is the step BB takes on
    every iteration; if it silently violates the radius the attack is a
    different algorithm than the docstring claims."""
    torch.manual_seed(0)
    to_clean = torch.randn(6, 3, 4, 4)
    grad = torch.randn(6, 3, 4, 4)
    offset = torch.randn(6)
    radius = torch.full((6,), 2.0)

    delta = BrendelBethge._trust_region_step(to_clean, grad, offset, radius)

    achieved = (grad.flatten(1) * delta.flatten(1)).sum(dim=1)
    norms = delta.flatten(1).norm(dim=1)
    # Radius is never violated.
    assert (norms <= radius + 1e-4).all()
    # And where the radius left room, the equality constraint is met exactly.
    min_norm = (offset / grad.flatten(1).pow(2).sum(dim=1)).abs() * grad.flatten(
        1
    ).norm(dim=1)
    reachable = min_norm <= radius
    assert reachable.any(), "test is vacuous"
    assert torch.allclose(achieved[reachable], offset[reachable], atol=1e-3)


def test_pdpgd_keeps_the_smallest_adversarial_iterate(model, inputs):
    """PDPGD tracks the best iterate rather than returning the last one. The
    last iterate of a primal-dual scheme oscillates -- returning it makes the
    reported norm a function of where the loop happened to stop."""
    attack = PDPGD(eps=BIG_EPS, steps=60)
    result = attack.run(model, inputs)

    clean_preds = _preds(model, inputs)
    with torch.no_grad():
        adv_preds = model(result.perturbed)[2]
    succeeded = adv_preds != clean_preds
    assert succeeded.any(), "attack never crossed the boundary; test is vacuous"
    # Every successful sample really is adversarial and really did move.
    # `minimised`, not `l2_norm`: this instance minimises L_inf.
    assert (result.minimised[succeeded] > 0).all()
    assert torch.equal(result.success, succeeded)


def test_pdpgd_unsuccessful_samples_are_returned_clean(model, inputs):
    """Same guarantee as BB: no perturbation is invented for a sample the
    attack could not solve."""
    result = PDPGD(eps=1e-7, steps=10).run(model, inputs)
    assert not result.success.any()
    assert torch.allclose(result.perturbed, inputs, atol=1e-6)


def test_pdpgd_l2_mode_minimises_in_l2(model, inputs):
    """The prox operator differs per norm; asking for L2 must actually apply
    the L2 one. The observable difference is that an L2 solution spreads the
    perturbation while an L_inf one saturates coordinates."""
    linf = PDPGD(eps=BIG_EPS, steps=40, norm="linf").run(model, inputs)
    l2 = PDPGD(eps=BIG_EPS, steps=40, norm="l2").run(model, inputs)
    assert not torch.allclose(linf.perturbed, l2.perturbed)


def test_min_norm_attacks_agree_with_each_other(model, inputs):
    """Three independent minimum-norm attacks searching the same boundary
    should land on comparable norms. A large disagreement means one of them
    has stalled -- the failure mode that quietly overstates robustness, since
    a stalled min-norm attack still returns a perturbation and still looks
    like a completed run.

    Budgets are generous on purpose: under-budgeting is itself the stall this
    is meant to catch, so the test must not bake one in.
    """
    bb = BrendelBethge(eps=BIG_EPS, steps=150).run(model, inputs)
    deepfool = DeepFool(eps=BIG_EPS, steps=50).run(model, inputs)
    pdpgd = PDPGD(eps=BIG_EPS, steps=150, norm="l2").run(model, inputs)

    solved = bb.success & deepfool.success & pdpgd.success
    assert solved.any(), "no sample solved by all three; test is vacuous"

    reference = deepfool.l2_norm[solved]
    assert (bb.l2_norm[solved] <= reference * 1.5).all()
    assert (pdpgd.l2_norm[solved] <= reference * 1.5).all()


def test_class_margin_is_negative_exactly_when_misclassified():
    """The boundary function both attacks steer by. A sign error here makes
    every minimum-norm attack walk the wrong way while still returning
    plausible numbers."""
    logits = torch.tensor([[3.0, 1.0, 0.0], [0.5, 2.0, 0.1], [1.0, 1.0, 5.0]])
    labels = torch.tensor([0, 0, 2])

    margin = class_margin(logits, labels)

    assert margin[0] > 0  # correctly the argmax
    assert margin[1] < 0  # class 1 wins instead
    assert margin[2] > 0
    assert torch.allclose(margin, torch.tensor([2.0, -1.5, 4.0]))


def test_project_l1_ball_projects_and_leaves_interior_points_alone():
    """The L1 projection is the engine of PDPGD's L_inf prox; an incorrect
    one silently changes which norm is being minimised."""
    outside = torch.tensor([[3.0, -4.0, 1.0]])
    inside = torch.tensor([[0.1, -0.2, 0.05]])

    projected = project_l1_ball(outside, 1.0)
    untouched = project_l1_ball(inside, 1.0)

    assert torch.isclose(projected.abs().sum(), torch.tensor(1.0), atol=1e-5)
    # Soft-thresholding: coordinates keep their sign or are driven to exactly
    # zero -- it never flips one, which is what would silently change the
    # direction of the step that uses it.
    kept = projected != 0
    assert torch.equal(projected[kept].sign(), outside[kept].sign())
    # The largest coordinate survives; the smallest is the one zeroed.
    assert projected[0, 1] != 0
    assert projected[0, 2] == 0
    assert torch.allclose(untouched, inside)


def test_project_l1_ball_accepts_per_sample_radii():
    v = torch.tensor([[3.0, -4.0, 1.0], [3.0, -4.0, 1.0]])
    radii = torch.tensor([1.0, 8.0])

    projected = project_l1_ball(v, radii)

    assert torch.isclose(projected[0].abs().sum(), torch.tensor(1.0), atol=1e-5)
    # 8 is exactly the input's L1 norm, so the second row is interior.
    assert torch.allclose(projected[1], v[1])


# --------------------------------------------------------------------------
# A minimum-norm attack must never lose to a fixed-budget witness.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("norm", "witness_cls", "budget"),
    [
        ("linf", PGD, 0.02),
        ("linf", PGD, 0.05),
        ("l2", PGDL2, 0.3),
        ("l2", PGDL2, 0.5),
    ],
)
def test_pdpgd_never_loses_to_a_fixed_budget_pgd_witness(
    seed, norm, witness_cls, budget, make_conv_model
):
    """The invariant that makes a minimum-norm result mean anything.

    A fixed-budget PGD that flips a sample inside ``||delta|| <= b`` is a
    *witness* that the true minimum norm for that sample is at most ``b``.
    So a minimum-norm attack capped at the same ``b``, in the same norm, that
    reports ``success=False`` on that sample is provably wrong -- and wrong
    in the only direction that matters here: it reports robustness the model
    does not have, on a run that looks exactly like a healthy one.

    The check is per sample, not on the rate, because a matching rate can
    still hide a swapped set. And it runs on a CONV net: on the single
    `nn.Linear` fixture the boundary is a hyperplane and a mis-scaled step
    still lands on it, which is precisely where this failure hides.
    """
    model = make_conv_model(seed=seed)
    torch.manual_seed(1)
    inputs = torch.rand(32, 3, 8, 8)

    witness = witness_cls(eps=budget, steps=100)(model, inputs)
    clean = _preds(model, inputs)
    with torch.no_grad():
        broken_by_witness = model(witness)[2] != clean
    assert broken_by_witness.any(), "witness broke nothing; test is vacuous"

    result = PDPGD(eps=budget, steps=200, norm=norm).run(model, inputs)

    missed = broken_by_witness & ~result.success
    assert not missed.any(), (
        f"PDPGD({norm!r}) reported failure on {int(missed.sum())} sample(s) that "
        f"{witness_cls.__name__} broke at the identical budget {budget}; the true "
        "minimum norm for those is provably within the cap"
    )


@pytest.mark.parametrize("norm", ["linf", "l2"])
def test_pdpgd_is_invariant_to_the_models_logit_scale(norm, make_conv_model):
    """Multiplying every logit by a constant moves no decision boundary, so
    it cannot change the norm needed to cross one. An attack whose step is
    ``lr * lambda * grad`` instead inherits the model's logit scale: it
    crawls on a small-scale model (reporting robustness) and overshoots on a
    large-scale one (reporting an inflated norm), and neither shows up as
    anything other than a completed run.
    """
    torch.manual_seed(1)
    inputs = torch.rand(24, 3, 8, 8)
    gains = (0.1, 1.0, 10.0, 1000.0)

    runs = {
        gain: PDPGD(eps=BIG_EPS, steps=100, norm=norm).run(
            make_conv_model(seed=0, gain=gain), inputs
        )
        for gain in gains
    }

    reference = runs[1.0]
    assert reference.success.any(), "nothing solved; test is vacuous"
    for gain in gains:
        result = runs[gain]
        assert torch.equal(result.success, reference.success), (
            f"gain {gain} changed which samples were solved"
        )
        solved = reference.success
        assert torch.allclose(
            result.minimised[solved], reference.minimised[solved], rtol=1e-3, atol=1e-5
        ), f"gain {gain} changed the reported minimum norm"


def test_pdpgd_step_size_does_not_decide_whether_it_crosses(conv_model, conv_inputs):
    """A tenfold change in `lr` is a change of pace, not of outcome: it may
    cost precision in the reported norm, but it must not decide whether the
    boundary is reached at all. When it does, `lr` is standing in for the
    logit scale and the reported success rate is a property of the
    hyperparameter rather than of the model.
    """
    rates = [
        PDPGD(eps=0.05, steps=100, lr=lr)
        .run(conv_model, conv_inputs)
        .success.float()
        .mean()
        for lr in (0.02, 0.002)
    ]
    assert rates[0] > 0, "attack solved nothing; test is vacuous"
    assert abs(rates[0] - rates[1]) <= 0.1, (
        f"success collapsed from {rates[0]:.3f} to {rates[1]:.3f} on a 10x lr change"
    )


def test_pdpgd_result_does_not_depend_on_the_rest_of_the_batch(conv_model):
    """A sample's reported robustness must be a property of the sample and
    the model, not of whoever it was batched with. Normalising the dual step
    by the batch-wide maximum margin throttles every sample by the single
    largest-margin one present, so the same sample reports a different norm
    after a reshuffle -- and the number moves without anything about the
    model changing.
    """
    torch.manual_seed(1)
    inputs = torch.rand(8, 3, 8, 8)

    batched = PDPGD(eps=BIG_EPS, steps=60, norm="l2").run(conv_model, inputs)
    alone = torch.stack(
        [
            PDPGD(eps=BIG_EPS, steps=60, norm="l2")
            .run(conv_model, inputs[i : i + 1])
            .l2_norm[0]
            for i in range(inputs.shape[0])
        ]
    )

    assert batched.success.any(), "nothing solved; test is vacuous"
    solved = batched.success
    assert torch.allclose(batched.l2_norm[solved], alone[solved], rtol=1e-3, atol=1e-5)


# --------------------------------------------------------------------------
# What the four attacks must agree on, whatever their internals.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attack_cls", MIN_NORM_ATTACKS)
def test_failed_samples_come_back_clean_with_zero_norm(attack_cls, conv_model):
    """`AttackResult`'s contract, held identically by all four.

    A capped walk that did not flip anything is not a measurement, but
    reporting it leaves ``l2_norm == eps`` on the row -- indistinguishable
    from a sample genuinely broken at cost eps. An attack that gives up
    instead reports 0. Same outcome, opposite numbers, and a curve that
    mixes the two is measuring which implementation ran.
    """
    torch.manual_seed(1)
    inputs = torch.rand(24, 3, 8, 8)
    # A cap well below the typical minimum norm, so some samples must fail.
    result = attack_cls(eps=0.02, steps=30).run(conv_model, inputs)

    failed = ~result.success
    assert failed.any(), "everything succeeded; test is vacuous"
    assert torch.allclose(result.perturbed[failed], inputs[failed], atol=1e-6)
    assert (result.minimised[failed] == 0).all()
    assert (result.effective_eps[failed] == 0).all()


@pytest.mark.parametrize("attack_cls", MIN_NORM_ATTACKS)
def test_reported_minimised_quantity_matches_the_norm_it_minimised(
    attack_cls, conv_model, conv_inputs
):
    """`minimised_norm` names the norm, `minimised` carries the value, and
    the value is the norm of the perturbation actually returned.

    PDPGD's default and shipped config minimise L_inf while `l2_norm` was
    being filled with the L2 norm of that L_inf-minimised perturbation -- a
    different quantity, on the same axis of the same curve as the three L2
    attacks' answers, with nothing on the row to say so.
    """
    result = attack_cls(eps=BIG_EPS, steps=30).run(conv_model, conv_inputs)
    delta = (result.perturbed - conv_inputs).flatten(1)

    assert result.minimised_norm in ("l2", "linf")
    expected = (
        delta.norm(dim=1) if result.minimised_norm == "l2" else delta.abs().amax(dim=1)
    )
    assert torch.allclose(result.minimised, expected, atol=1e-6)
    # `l2_norm` is set only when L2 is what was minimised.
    assert (result.l2_norm is not None) == (result.minimised_norm == "l2")


@pytest.mark.parametrize("attack_cls", MIN_NORM_ATTACKS)
def test_min_norm_attacks_run_on_float64(attack_cls, make_conv_model, conv_inputs):
    """The harness is not float32-only, and a float32 scratch tensor inside
    an otherwise dtype-agnostic attack does not fail where it was written:
    `torch.full((n,), inf)` takes the default dtype, arithmetic silently
    promotes around it, and the mismatch only surfaces at the one operation
    that does not promote -- an index-put, several lines later, reported as
    an unrelated dtype error.
    """
    model = make_conv_model().double()
    inputs = conv_inputs.double()

    result = attack_cls(eps=0.5, steps=10).run(model, inputs)

    assert result.perturbed.dtype == torch.float64
    assert result.effective_eps.dtype == torch.float64


@pytest.mark.parametrize("attack_cls", MIN_NORM_ATTACKS)
def test_min_norm_attacks_accept_non_image_inputs(attack_cls, flat_model, flat_inputs):
    """Nothing in these algorithms is about images, and the base class's
    budget contract is written over ``(B, ...)``. Hardcoding an image's rank
    -- ``mask[:, None, None, None]`` rather than the rank-agnostic
    ``(-1,) + (1,) * (inputs.ndim - 1)`` the rest of the suite uses -- does
    not raise where it is written either: it broadcasts to the wrong rank and
    surfaces as a matmul shape error somewhere unrelated.
    """
    result = attack_cls(eps=0.5, steps=10).run(flat_model, flat_inputs)

    assert result.perturbed.shape == flat_inputs.shape
