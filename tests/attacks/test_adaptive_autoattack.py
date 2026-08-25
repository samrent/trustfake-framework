"""Behaviour specific to A^3 (Adaptive Auto Attack).

A^3's contribution is not a new gradient step -- it is how the budget is
spent. So the tests here check the two mechanisms that spend it (adaptive
initialisation, online discarding) rather than checking that the attack
perturbs something, which the generic contract battery already covers.
"""

import torch

from trustfake.attacks import PGD, AdaptiveAutoAttack
from trustfake.attacks._common import class_margin, model_logits

EPS = 0.05


def _flip_rate(model, x, x_adv):
    with torch.no_grad():
        clean = model(x)[2]
        adv = model(x_adv)[2]
    return (adv != clean).float().mean().item()


def test_adi_start_beats_a_random_start(model, inputs):
    """Adaptive Direction Initialization must actually pick a better start.
    If the candidate scoring is broken it silently degrades to 'the first
    candidate', and A^3 becomes PGD with extra forward passes."""
    attack = AdaptiveAutoAttack(eps=EPS, n_random_init=6)
    with torch.no_grad():
        preds = model(inputs)[2]
    gen = torch.Generator(device=inputs.device).manual_seed(0)

    chosen = attack._candidates(model, inputs, preds, transfer=[], gen=gen)
    random_delta = torch.empty_like(inputs).uniform_(-EPS, EPS, generator=gen)

    def margin_of(delta):
        with torch.no_grad():
            return class_margin(
                model_logits(model, attack._clamp(inputs + delta)), preds
            )

    # Lower margin = closer to (or past) the boundary = a better start.
    assert margin_of(chosen).mean() <= margin_of(random_delta).mean()


def test_adi_never_picks_a_start_worse_than_doing_nothing(model, inputs):
    """The zero delta is always in the candidate set, so the chosen start can
    never be worse than the clean point."""
    attack = AdaptiveAutoAttack(eps=EPS, n_random_init=4)
    with torch.no_grad():
        preds = model(inputs)[2]
    gen = torch.Generator(device=inputs.device).manual_seed(0)

    chosen = attack._candidates(model, inputs, preds, transfer=[], gen=gen)

    with torch.no_grad():
        clean_margin = class_margin(model_logits(model, inputs), preds)
        chosen_margin = class_margin(
            model_logits(model, attack._clamp(inputs + chosen)), preds
        )
    assert (chosen_margin <= clean_margin + 1e-6).all()


def test_osd_stops_working_on_solved_samples(model, inputs):
    """Online Statistics-based Discarding: a sample that has flipped leaves
    the active set, and the freed budget goes to the rest. Observable as the
    attack terminating early once everything is solved."""
    attack = AdaptiveAutoAttack(eps=0.5, steps=40, rounds=4)
    result = attack.run(model, inputs)

    assert result.success.all(), "everything should be solvable at this eps"
    # Solved samples keep the perturbation that solved them.
    with torch.no_grad():
        adv_preds = model(result.perturbed)[2]
    assert (adv_preds != result.clean_preds).all()


def test_a3_is_at_least_as_effective_as_plain_pgd(model, inputs, targets):
    """The point of the extra machinery. A^3 given the same eps should flip at
    least as many samples as PGD -- if it flips fewer, the budget
    redistribution is losing work rather than saving it."""
    a3 = AdaptiveAutoAttack(eps=EPS, steps=30, rounds=3)(model, inputs, targets)
    pgd = PGD(eps=EPS, steps=30)(model, inputs, targets)

    assert _flip_rate(model, inputs, a3) >= _flip_rate(model, inputs, pgd)


def _trace_rounds(attack, model, inputs):
    """Record ``(active_set_size, steps_allocated)`` for every inner call."""
    trace = []
    original = attack._apgd

    def spy(model_, inputs_, preds_, delta_, steps_, _o=original, _t=trace):
        _t.append((inputs_.shape[0], steps_))
        return _o(model_, inputs_, preds_, delta_, steps_)

    attack._apgd = spy
    attack.run(model, inputs)
    return trace


def test_a3_respects_its_budget_across_rounds(model, inputs):
    """OSD redistributes the budget; it must not inflate it. The budget is a
    COST -- ``steps * batch`` sample-iterations -- so what has to be bounded
    is the work, not the iteration count, which redistribution deliberately
    raises."""
    budgets = []
    for rounds in (1, 3):
        attack = AdaptiveAutoAttack(eps=EPS, steps=12, rounds=rounds)
        trace = _trace_rounds(attack, model, inputs)
        budgets.append(sum(active * steps for active, steps in trace))

    allowed = 12 * inputs.shape[0]
    # +rounds for the per-round integer-division remainder.
    assert budgets[0] <= allowed + 1
    assert budgets[1] <= allowed + 3 * inputs.shape[0]


def test_osd_redistributes_the_budget_freed_by_solved_samples(conv_model, conv_inputs):
    """Online Statistics-based Discarding, as the docstring and `a3.yaml`
    both describe it: the freed budget is REDISTRIBUTED, so a round on fewer
    samples runs longer.

    Splitting the iteration count evenly instead is the failure that hides:
    the active set falls away round by round while `steps_allocated` sits at
    the same number, the freed budget is simply never spent, and the attack
    is weaker than advertised on exactly the samples it was supposed to
    concentrate on -- which reads as robustness, not as a bug.
    """
    # An eps small enough that some samples survive round 1, so the active
    # set actually shrinks and there is a budget to redistribute.
    attack = AdaptiveAutoAttack(eps=0.01, steps=100, rounds=4)
    trace = _trace_rounds(attack, conv_model, conv_inputs)

    assert len(trace) > 1, "only one round ran; test is vacuous"
    first_active, first_steps = trace[0]
    shrunk = [entry for entry in trace[1:] if entry[0] < first_active]
    assert shrunk, "the active set never shrank; test is vacuous"

    for active, steps in shrunk:
        assert steps > first_steps, (
            f"a round on {active} samples (down from {first_active}) still got "
            f"{steps} steps, the same as the full batch: the freed budget was "
            "discarded, not redistributed"
        )
        # Redistributed, not invented: each round's cost stays in proportion.
        assert active * steps <= first_active * first_steps * 1.2


def test_apgd_halving_counts_progress_against_the_previous_iterate(
    conv_model, conv_inputs
):
    """Croce & Hein's condition 1 counts iterations that improved on the
    PREVIOUS iterate. Counting against the running best is a strictly harder
    bar -- once a good point is found, later iterates rarely beat it even
    while the search is moving productively -- so the counter reads low, the
    step halves early and repeatedly, and the attack freezes short of the
    boundary. That is a weaker attack reported as a more robust model.

    Read off the shipped loop: `_apgd` calls `_margin` once before the loop
    and once per step, so the recorded sequence is the iterate-by-iterate
    margin and both counters can be reconstructed from it exactly.
    """
    steps = 32
    attack = AdaptiveAutoAttack(eps=0.02, steps=steps, rounds=1)
    with torch.no_grad():
        preds = conv_model(conv_inputs)[2]

    seen = []
    original = attack._margin

    def recording(model_, x_, preds_, _o=original, _s=seen):
        value = _o(model_, x_, preds_)
        _s.append(value.clone())
        return value

    attack._margin = recording
    attack._apgd(conv_model, conv_inputs, preds, torch.zeros_like(conv_inputs), steps)

    sequence = torch.stack(seen)  # (steps + 1, B)
    assert sequence.shape[0] == steps + 1, (
        "call pattern changed; counter is not readable"
    )

    running_best = torch.cummin(sequence, dim=0).values
    against_previous = (sequence[1:] < sequence[:-1]).sum(dim=0).float()
    against_best = (sequence[1:] < running_best[:-1]).sum(dim=0).float()

    # The two disagree -- otherwise this test could not tell them apart --
    # and the against-best count is the pessimistic one.
    assert (against_best <= against_previous).all()
    assert against_previous.mean() > against_best.mean() + 0.5, (
        "the two counters agree on this fixture; test cannot distinguish them"
    )


def test_apgd_halving_rule_has_both_of_croce_and_heins_conditions():
    """Condition 2 is not implied by condition 1, so dropping it changes the
    schedule. A window can improve on the previous iterate often enough to
    clear the ``rho`` bar while never improving the best value -- that is
    oscillation across the boundary, and the only thing that catches it is
    "step size and best value both unchanged since the last checkpoint".
    """
    window = 8
    rho = 0.75
    # Sample 0: cleared condition 1, and the best value moved -> keep going.
    # Sample 1: cleared condition 1, but nothing moved at all -> condition 2.
    # Sample 2: failed condition 1 -> condition 1.
    # Sample 3: nothing moved, but the step was already halved for it, so
    #           condition 2 does not re-fire on the same stall.
    improved = torch.tensor([8.0, 8.0, 1.0, 8.0])
    eta = torch.tensor([0.1, 0.1, 0.1, 0.05])
    eta_before = torch.tensor([0.1, 0.1, 0.1, 0.1])
    best = torch.tensor([-1.0, 0.5, 0.5, 0.5])
    best_before = torch.tensor([0.5, 0.5, 0.5, 0.5])

    halve = AdaptiveAutoAttack._should_halve(
        improved, window, eta, eta_before, best, best_before, rho
    )

    assert halve.tolist() == [False, True, True, False]


def test_a3_is_deterministic_across_calls(model, inputs):
    """The harness requires it: a seeded generator, not the global RNG."""
    attack = AdaptiveAutoAttack(eps=EPS, steps=12, rounds=2)
    torch.manual_seed(999)
    first = attack(model, inputs)
    torch.manual_seed(1)
    second = attack(model, inputs)
    assert torch.allclose(first, second)
