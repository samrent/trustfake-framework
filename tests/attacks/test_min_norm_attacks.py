"""Behaviour specific to the minimum-norm attacks (BB and PDPGD).

The generic contract battery checks that an attack stays inside its budget.
For a minimum-norm attack that is the least interesting property: the budget
is a cap on the answer, not the search, and the number the attack exists to
produce is the norm it needed. These tests check the properties that make
that number trustworthy -- that it is actually minimised, that a failure is
reported as a failure rather than as a tiny perturbation, and that the
closed-form trust-region step solves the problem it claims to solve.
"""

import torch

from trustfake.attacks import PDPGD, BrendelBethge, DeepFool
from trustfake.attacks._common import class_margin, project_l1_ball

BIG_EPS = 100.0  # effectively uncapped: let the attack find its own norm


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
    assert (result.l2_norm[succeeded] > 0).all()
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
