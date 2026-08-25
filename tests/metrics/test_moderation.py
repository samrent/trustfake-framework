"""WP4 selective-moderation tests. Pure numpy, no data/network/GPU."""

import numpy as np
import pytest

from trustfake.metrics.moderation import (
    ALLOW,
    FLAG,
    REVIEW,
    ModerationPolicy,
    evaluate_policy,
    fit_thresholds,
    fit_uncertainty_gate,
    moderation_actions,
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
