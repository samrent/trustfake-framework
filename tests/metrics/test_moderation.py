"""WP4 selective-moderation tests. Pure numpy, no data/network/GPU."""

import json

import numpy as np
import pytest

from trustfake.metrics.moderation import (
    ACTION_NAMES,
    ALLOW,
    FLAG,
    REVIEW,
    ModerationPolicy,
    evaluate_policy,
    fit_thresholds,
    fit_uncertainty_gate,
    moderation_actions,
    moderation_report,
    p_fake_from,
)


def test_action_rule():
    p = np.array([0.1, 0.5, 0.9])
    a = moderation_actions(p, t_low=0.3, t_high=0.7)
    assert a.tolist() == [ALLOW, REVIEW, FLAG]


def test_uncertainty_gate_only_moves_to_review():
    """The second axis can only push items into REVIEW; auto-decisions can
    only decrease, never flip ALLOW<->FLAG."""
    p = np.array([0.1, 0.9, 0.1, 0.9])
    unc = np.array([0.0, 0.0, 1.0, 1.0])
    base = moderation_actions(p, 0.3, 0.7)
    gated = moderation_actions(p, 0.3, 0.7, unc, t_unc=0.5)
    for b, g in zip(base, gated, strict=True):
        assert g == b or g == REVIEW
    assert gated.tolist() == [ALLOW, FLAG, REVIEW, REVIEW]


def test_residual_risk_is_nan_when_nothing_auto_decided():
    r = evaluate_policy(np.full(10, 0.5), np.zeros(10), t_low=0.0, t_high=1.0)
    assert np.isnan(r["residual_risk"])
    assert r["coverage"] == 0.0 and r["review_rate"] == 1.0


def test_fit_meets_sla_and_minimises_review():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.5 * y + 0.25 * rng.standard_normal(4000) + 0.25, 0, 1)
    t_low, t_high = fit_thresholds(p, y, sla_residual_risk=0.1)
    r = evaluate_policy(p, y, t_low, t_high)
    assert r["residual_risk"] <= 0.1 + 1e-9
    assert r["coverage"] > 0.0  # actually decides things


def test_infeasible_sla_degrades_to_review_everything():
    rng = np.random.default_rng(1)
    p = rng.random(500)  # p_fake carries no signal
    y = rng.integers(0, 2, 500)
    assert fit_thresholds(p, y, sla_residual_risk=1e-4) == (0.0, 1.0)


def test_uncertainty_gate_quantile():
    unc = np.linspace(0, 1, 1001)
    t = fit_uncertainty_gate(unc, clean_review_budget=0.1)
    assert t == pytest.approx(0.9, abs=1e-3)


def test_policy_p_fake_and_evaluate_3class():
    """3-class: fake = synthetic (1) or tampered (2), p_fake = 1 - P(real)."""
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    targets = np.array([0, 1, 2])
    pol = ModerationPolicy(t_low=0.3, t_high=0.7, real_class=0)
    assert np.allclose(pol.p_fake(probs), [0.2, 0.9, 0.9])
    r = pol.evaluate(probs, targets)
    # real auto-allowed, both fakes auto-flagged -> full coverage, no error
    assert r["coverage"] == 1.0
    assert r["residual_risk"] == 0.0
    assert r["missed_fake_rate"] == 0.0


def test_attack_that_raises_pfake_on_reals_raises_false_flags():
    probs_clean = np.tile([0.9, 0.05, 0.05], (100, 1))
    targets = np.zeros(100, dtype=int)  # all real
    pol = ModerationPolicy(t_low=0.3, t_high=0.7, real_class=0)
    assert pol.evaluate(probs_clean, targets)["false_flag_rate"] == 0.0
    probs_attacked = np.tile([0.1, 0.45, 0.45], (100, 1))  # pushed to look fake
    assert pol.evaluate(probs_attacked, targets)["false_flag_rate"] == 1.0


def test_vectorised_fit_matches_bruteforce_optimum():
    """The vectorised threshold search must return the same optimum (same best
    achievable review rate at the SLA) as an O(grid^2 * n) brute force."""

    def brute(p, y, sla, grid=200):
        min_auto = int(np.ceil(1.0 / max(sla, 1e-9)))
        qs = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
        best_rev = 1.0
        for t_low in qs:
            for t_high in qs[qs >= t_low]:
                r = evaluate_policy(p, y, t_low, t_high)
                if r["n_auto"] < min_auto:
                    continue
                if not np.isfinite(r["residual_risk"]) or r["residual_risk"] > sla:
                    continue
                best_rev = min(best_rev, r["review_rate"])
        return best_rev

    rng = np.random.default_rng(7)
    for _ in range(4):
        y = rng.integers(0, 2, 2000)
        p = np.clip(0.5 * y + 0.3 * rng.standard_normal(2000) + 0.25, 0, 1)
        t_low, t_high = fit_thresholds(p, y, 0.1)
        r = evaluate_policy(p, y, t_low, t_high)
        assert r["residual_risk"] <= 0.1 + 1e-9
        assert r["review_rate"] == pytest.approx(brute(p, y, 0.1), abs=1e-9)


# --- action names -------------------------------------------------------------


def test_action_names_cover_every_action():
    """Every code the rule can emit has one human-readable name."""
    assert ACTION_NAMES == {ALLOW: "allow", REVIEW: "review", FLAG: "flag"}
    assert len(set(ACTION_NAMES.values())) == 3
    a = moderation_actions(np.array([0.1, 0.5, 0.9]), t_low=0.3, t_high=0.7)
    assert [ACTION_NAMES[int(x)] for x in a] == ["allow", "review", "flag"]


# --- n, and the review load split by truth ------------------------------------


def _gate_scenario():
    """100 reals at p=0.05 and 100 fakes at p=0.95: under the one-axis rule
    every item is auto-decided, and every auto-decision is right."""
    p = np.concatenate([np.full(100, 0.05), np.full(100, 0.95)])
    y = np.concatenate([np.zeros(100, int), np.ones(100, int)])
    return p, y


def test_n_is_reported_so_every_rate_is_re_derivable():
    p = np.array([0.1, 0.5, 0.9, 0.95])
    y = np.array([0, 1, 1, 1])
    r = evaluate_policy(p, y, t_low=0.3, t_high=0.7)
    assert r["n"] == 4
    assert r["n_auto"] == 3
    assert r["coverage"] * r["n"] == r["n_auto"]
    assert round(r["review_rate"] * r["n"]) == 1


def test_review_of_fakes_separates_targeted_from_indiscriminate_review():
    """The point of the split. Two gates with the SAME review rate -- one
    escalating fakes, one escalating random traffic -- agree on every other
    indicator; only review_of_fakes / review_of_reals tell them apart."""
    p, y = _gate_scenario()
    targeted = np.zeros(200)
    targeted[100:120] = 1.0  # 20 uncertain items, all of them fakes
    indiscriminate = np.zeros(200)
    indiscriminate[:10] = 1.0  # 10 reals ...
    indiscriminate[100:110] = 1.0  # ... and 10 fakes: same volume, same bill

    a = evaluate_policy(p, y, 0.3, 0.7, targeted, t_unc=0.5)
    b = evaluate_policy(p, y, 0.3, 0.7, indiscriminate, t_unc=0.5)

    for k in (
        "n",
        "coverage",
        "review_rate",
        "residual_risk",
        "missed_fake_rate",
        "false_flag_rate",
        "full_coverage_error",
        "accuracy",
    ):
        assert a[k] == pytest.approx(b[k], nan_ok=True)
    assert a["review_rate"] == pytest.approx(0.1)

    assert a["review_of_fakes"] == pytest.approx(0.20)
    assert a["review_of_reals"] == pytest.approx(0.00)
    assert b["review_of_fakes"] == pytest.approx(0.10)
    assert b["review_of_reals"] == pytest.approx(0.10)


def test_review_shares_close_the_per_class_accounting():
    """Per class, the three actions partition the items: allowed + reviewed +
    flagged = 1, so the review share cannot be inflated by double counting."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.6 * y + 0.25 * rng.standard_normal(500) + 0.2, 0, 1)
    unc = rng.random(500)
    r = evaluate_policy(p, y, 0.3, 0.7, unc, t_unc=0.8)
    a = moderation_actions(p, 0.3, 0.7, unc, 0.8)

    flagged_fakes = float((a[y == 1] == FLAG).mean())
    allowed_reals = float((a[y == 0] == ALLOW).mean())
    fake_side = r["missed_fake_rate"] + r["review_of_fakes"] + flagged_fakes
    real_side = r["false_flag_rate"] + r["review_of_reals"] + allowed_reals
    assert fake_side == pytest.approx(1.0)
    assert real_side == pytest.approx(1.0)


def test_gate_lowers_shipped_harm_but_can_raise_the_residual_rate():
    """The gate's guarantee is on COUNTS, not on the rate. It only removes
    auto-decisions, so wrong auto-decisions -- and missed_fake_rate /
    false_flag_rate, whose denominators are fixed class sizes -- can only fall.
    residual_risk divides by a SHRINKING denominator, so escalating correct
    auto-decisions makes the rate rise while strictly less harm ships. Anyone
    re-asserting 'the gate cannot raise residual risk' fails here."""
    # 90 reals auto-ALLOWED correctly; 10 reals auto-FLAGGED wrongly.
    p = np.concatenate([np.full(90, 0.05), np.full(10, 0.95)])
    y = np.zeros(100, int)
    unc = np.zeros(100)
    unc[:20] = 1.0  # escalate 20 items that were being decided CORRECTLY
    one = evaluate_policy(p, y, 0.3, 0.7)
    two = evaluate_policy(p, y, 0.3, 0.7, unc, t_unc=0.5)

    assert one["residual_risk"] == pytest.approx(0.10)
    assert two["residual_risk"] == pytest.approx(0.125)  # the RATE rose

    def shipped(r):
        return r["residual_risk"] * r["n_auto"]

    assert shipped(two) == pytest.approx(shipped(one))  # ... harm did not
    assert two["coverage"] < one["coverage"]
    assert two["false_flag_rate"] <= one["false_flag_rate"]
    assert two["missed_fake_rate"] <= one["missed_fake_rate"]


def test_gate_never_increases_shipped_harm_over_random_conditions():
    """The count invariant, checked over noisy conditions rather than one
    hand-built case."""
    rng = np.random.default_rng(21)
    for _ in range(20):
        y = rng.integers(0, 2, 400)
        p = np.clip(0.5 * y + 0.3 * rng.standard_normal(400) + 0.25, 0, 1)
        unc = rng.random(400)
        one = evaluate_policy(p, y, 0.3, 0.7)
        two = evaluate_policy(p, y, 0.3, 0.7, unc, t_unc=0.7)
        harm_one = one["residual_risk"] * one["n_auto"]
        harm_two = 0.0 if two["n_auto"] == 0 else two["residual_risk"] * two["n_auto"]
        assert harm_two <= harm_one + 1e-9
        assert two["coverage"] <= one["coverage"]
        assert two["missed_fake_rate"] <= one["missed_fake_rate"] + 1e-12
        assert two["false_flag_rate"] <= one["false_flag_rate"] + 1e-12


def test_review_shares_are_zero_for_an_absent_class():
    """No fakes in the condition: the fake-side rates are 0.0 by convention,
    not NaN, because there is no fake to have mishandled."""
    r = evaluate_policy(np.full(10, 0.5), np.zeros(10, int), t_low=0.3, t_high=0.7)
    assert r["review_of_fakes"] == 0.0
    assert r["review_of_reals"] == 1.0
    assert r["missed_fake_rate"] == 0.0


# --- p_fake straight from logits ----------------------------------------------


def test_p_fake_from_logits_matches_the_probability_path():
    """The logits entry point and ModerationPolicy.p_fake must agree, or the
    two paths carry two conventions."""
    rng = np.random.default_rng(11)
    logits = rng.standard_normal((64, 3)) * 3.0
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    pol = ModerationPolicy(t_low=0.3, t_high=0.7, real_class=0)
    assert np.allclose(p_fake_from(logits), pol.p_fake(probs))
    assert np.allclose(pol.p_fake_from_logits(logits), pol.p_fake(probs))
    # 3-class: p_fake aggregates BOTH fake columns (synthetic and tampered)
    assert np.allclose(p_fake_from(logits), probs[:, 1] + probs[:, 2])


def test_p_fake_from_binary_is_the_softmax_fake_column():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    z = logits / 1.7
    e = np.exp(z - z.max(1, keepdims=True))
    assert np.allclose(p_fake_from(logits, 1.7), (e / e.sum(1, keepdims=True))[:, 1])


def test_p_fake_from_respects_real_class():
    logits = np.array([[3.0, 0.0, 0.0]])
    assert p_fake_from(logits, real_class=0) < 0.2
    assert p_fake_from(logits, real_class=1) > 0.8


def test_binary_temperature_rescales_p_fake_without_reordering():
    """With two classes p_fake is monotone in the single logit margin, so T
    moves it against the FROZEN thresholds but cannot reorder items."""
    rng = np.random.default_rng(12)
    logits = rng.standard_normal((200, 2)) * 4.0
    cold = p_fake_from(logits, 0.5)
    warm = p_fake_from(logits, 4.0)
    order = np.argsort(cold, kind="stable")
    assert np.array_equal(order, np.argsort(warm, kind="stable"))
    assert not np.allclose(cold, warm)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_p_fake_from_rejects_non_positive_temperature(bad):
    with pytest.raises(ValueError):
        p_fake_from(np.zeros((3, 2)), bad)


def test_p_fake_from_rejects_non_2d_logits():
    with pytest.raises(ValueError):
        p_fake_from(np.zeros(3))


def test_p_fake_from_is_float64_and_does_not_saturate():
    """float32 logits, float64 softmax: two confident rows stay DISTINCT
    instead of both collapsing to exactly 1.0 and forming a tie block."""
    logits = np.array([[0.0, 30.0], [0.0, 25.0]], dtype=np.float32)
    p = p_fake_from(logits)
    assert p.dtype == np.float64
    assert np.all(p < 1.0)
    assert p[0] > p[1]


# --- the no-abstention counterfactual -----------------------------------------


def test_accuracy_and_full_coverage_error_are_the_no_abstention_baseline():
    """Residual risk is only readable next to the error the same detector
    would have had by never abstaining."""
    p = np.array([0.1, 0.4, 0.45, 0.9])
    y = np.array([0, 1, 0, 1])
    r = evaluate_policy(p, y, t_low=0.3, t_high=0.7)
    # no abstention, p > 0.5: predictions [0, 0, 0, 1] against y -> 1 error of 4
    assert r["full_coverage_error"] == pytest.approx(0.25)
    assert r["accuracy"] == pytest.approx(0.75)
    assert r["accuracy"] + r["full_coverage_error"] == pytest.approx(1.0)
    # abstaining on the two middling items removes the only error
    assert r["residual_risk"] == pytest.approx(0.0)
    assert r["residual_risk"] < r["full_coverage_error"]


# --- the asymmetric SLA on missed fakes ---------------------------------------


def _fakes_hiding_in_the_allow_zone():
    """100 reals, 95 obvious fakes, and 5 fakes scored as low as reals -- one
    of them at the very bottom of the score."""
    rng = np.random.default_rng(5)
    reals = rng.uniform(0.05, 0.40, 100)
    hiding = np.linspace(0.0, 0.35, 5)
    obvious = rng.uniform(0.90, 1.00, 95)
    p = np.concatenate([reals, hiding, obvious])
    y = np.concatenate([np.zeros(100, int), np.ones(100, int)])
    return p, y


def test_sla_missed_fake_buys_fakes_out_of_the_allow_zone_with_review():
    """The symmetric residual-risk SLA is happy to auto-allow the 5 hiding
    fakes (they are only 2.5% of auto-decisions). The asymmetric cap prices a
    missed fake higher than a false flag and pays for it in review rate."""
    p, y = _fakes_hiding_in_the_allow_zone()
    free = evaluate_policy(p, y, *fit_thresholds(p, y, sla_residual_risk=0.05))
    capped = evaluate_policy(
        p, y, *fit_thresholds(p, y, sla_residual_risk=0.05, sla_missed_fake=0.0)
    )
    assert free["missed_fake_rate"] > 0.0
    assert capped["missed_fake_rate"] == 0.0
    assert capped["review_rate"] > free["review_rate"]
    assert capped["residual_risk"] <= 0.05 + 1e-9


def test_sla_missed_fake_matches_bruteforce_optimum():
    """The vectorised search under BOTH constraints returns the same best
    review rate as an explicit O(grid^2 * n) search."""

    def brute(p, y, sla, cap, grid):
        min_auto = int(np.ceil(1.0 / max(sla, 1e-9)))
        qs = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
        best = 1.0
        for t_low in qs:
            for t_high in qs[qs >= t_low]:
                r = evaluate_policy(p, y, t_low, t_high)
                if r["n_auto"] < min_auto:
                    continue
                if not np.isfinite(r["residual_risk"]) or r["residual_risk"] > sla:
                    continue
                if r["missed_fake_rate"] > cap:
                    continue
                best = min(best, r["review_rate"])
        return best

    rng = np.random.default_rng(9)
    for cap in (0.0, 0.02, 0.10):
        y = rng.integers(0, 2, 800)
        p = np.clip(0.5 * y + 0.3 * rng.standard_normal(800) + 0.25, 0, 1)
        t_low, t_high = fit_thresholds(p, y, 0.1, sla_missed_fake=cap, grid=60)
        r = evaluate_policy(p, y, t_low, t_high)
        assert r["missed_fake_rate"] <= cap + 1e-9
        assert r["review_rate"] == pytest.approx(brute(p, y, 0.1, cap, 60), abs=1e-9)


def test_sla_missed_fake_is_ignored_when_none():
    """Passing None must leave the symmetric objective untouched."""
    p, y = _fakes_hiding_in_the_allow_zone()
    assert fit_thresholds(p, y, 0.05) == fit_thresholds(p, y, 0.05, None)


# --- reporting ----------------------------------------------------------------


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip("|").split("|")]


def _row(md: str, prefix: str) -> list[str]:
    return _cells(next(ln for ln in md.splitlines() if ln.startswith(prefix)))


def test_report_puts_the_no_abstention_error_next_to_the_residual_risk():
    """The WP4 headline is a comparison, so the two columns are adjacent."""
    p = np.array([0.1, 0.4, 0.45, 0.9])
    y = np.array([0, 1, 0, 1])
    md, payload = moderation_report({"clean": evaluate_policy(p, y, 0.3, 0.7)})
    cols = _row(md, "| condition |")
    assert cols[cols.index("resid_risk") + 1] == "full_cov_err"
    cells = _row(md, "| clean |")
    assert cells[cols.index("resid_risk")] == "0.00%"
    assert cells[cols.index("full_cov_err")] == "25.00%"
    assert cells[cols.index("acc")] == "0.7500"
    assert cells[cols.index("n")] == "4"
    assert payload["conditions"]["clean"]["accuracy"] == pytest.approx(0.75)
    assert payload["conditions"]["clean"]["full_coverage_error"] == pytest.approx(0.25)


def test_report_never_prints_a_missing_residual_risk_as_zero():
    """A condition that auto-decides nothing has no residual risk: 'n/a' in the
    table, null in the payload -- never 0.00%, which would read as perfect."""
    p, y = _gate_scenario()
    md, payload = moderation_report(
        {"clean": evaluate_policy(p, y, 0.3, 0.7)},
        policy=ModerationPolicy(t_low=0.0, t_high=1.0),
        sla_residual_risk=0.001,
    )
    assert "SLA INFEASIBLE" in md
    assert payload["policy"]["infeasible"] is True

    dead = evaluate_policy(p, y, 0.0, 1.0)
    assert np.isnan(dead["residual_risk"])  # NaN at the source, not 0.0
    md, payload = moderation_report({"clean": dead})
    cols = _row(md, "| condition |")
    assert _row(md, "| clean |")[cols.index("resid_risk")] == "n/a"
    assert payload["conditions"]["clean"]["residual_risk"] is None


def test_report_payload_is_strict_json():
    """Strict JSON (allow_nan=False) so the artifact survives any consumer;
    non-measurable indicators are null, not NaN and not 0.0."""
    p, y = _gate_scenario()
    payload = moderation_report(
        {"clean": evaluate_policy(p, y, 0.3, 0.7), "dead": evaluate_policy(p, y, 0, 1)},
        policy=ModerationPolicy(t_low=0.3, t_high=0.7, t_unc=0.5),
        sla_residual_risk=0.05,
        sla_missed_fake=0.02,
    )[1]
    round_tripped = json.loads(json.dumps(payload, allow_nan=False))
    assert round_tripped["policy"]["t_unc"] == 0.5
    assert round_tripped["sla"] == {"residual_risk": 0.05, "missed_fake": 0.02}
    assert round_tripped["conditions"]["clean"]["n"] == 200
    assert round_tripped["conditions"]["dead"]["residual_risk"] is None


def test_report_two_axis_table_shows_the_gate_moving_fakes():
    """The comparison table exists to show WHICH traffic the gate escalates."""
    p, y = _gate_scenario()
    unc = np.zeros(200)
    unc[100:120] = 1.0  # only fakes are uncertain
    one = evaluate_policy(p, y, 0.3, 0.7)
    two = evaluate_policy(p, y, 0.3, 0.7, unc, t_unc=0.5)
    md, payload = moderation_report(
        {"clean": one},
        {"clean": two},
        policy=ModerationPolicy(t_low=0.3, t_high=0.7, t_unc=0.5),
    )
    gate_table = md.split("### one axis")[1]  # the per-condition table is first
    cols = _row(gate_table, "| condition |")
    cells = _row(gate_table, "| clean |")
    assert cells[cols.index("rev_fakes 1ax")] == "0.0%"
    assert cells[cols.index("rev_fakes 2ax")] == "20.0%"
    assert cells[cols.index("review 2ax")] == "10.0%"
    entry = payload["conditions"]["clean"]
    assert entry["review_of_fakes"] == pytest.approx(0.0)
    assert entry["two_axis"]["review_of_fakes"] == pytest.approx(0.2)
    assert entry["two_axis"]["review_of_reals"] == pytest.approx(0.0)


def test_report_omits_the_two_axis_table_without_gated_results():
    p, y = _gate_scenario()
    md, payload = moderation_report({"clean": evaluate_policy(p, y, 0.3, 0.7)})
    assert "two axes" not in md
    assert "two_axis" not in json.dumps(payload)
    assert payload["policy"] is None
    assert payload["sla"] == {"residual_risk": None, "missed_fake": None}
