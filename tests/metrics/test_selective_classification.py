"""Defences for the selective-classification estimator.

Each test guards one specific way an AURC number lies. Pure numpy paths are
tested directly; the torchmetrics wrappers get their own integration tests.
No GPU, no data, no network.
"""

import numpy as np
import pytest
import torch

from trustfake.metrics.evaluation.selective_classification import (
    AchievedCoverage,
    AUGRiskCoverage,
    AURiskCoverage,
    EAURiskCoverage,
    NumOperatingPoints,
    RiskAtCoverage,
    augrc_from_scores,
    aurc_from_scores,
    aurc_oracle,
    coverage_at_risk,
    eaurc_from_scores,
    get_selective_classification_metrics,
    n_operating_points,
    operating_point_at_coverage,
    rc_curve,
    risk_at_coverage,
)

# ------------------------------------------------------------- pure estimator


def test_perfect_ranker_has_zero_excess_aurc():
    """A ranker that puts every correct prediction above every mistake IS the
    empirical oracle, so E-AURC must be exactly 0 -- not 'small'. If this
    fails, every E-AURC is offset by a constant."""
    rng = np.random.default_rng(0)
    correct = (rng.random(1000) > 0.2).astype(float)
    errors = 1.0 - correct
    uncertainty = errors + rng.random(1000) * 1e-6  # correct first, no cross-block ties

    assert eaurc_from_scores(uncertainty, errors) == pytest.approx(0.0, abs=1e-15)

    # The closed form r + (1-r)ln(1-r) is asymptotic: at n=1000 it is off by
    # ~1e-4, the same order as the effects reported. This is why it is never
    # used as the oracle reference.
    r = errors.mean()
    closed = r + (1.0 - r) * np.log(1.0 - r)
    empirical = aurc_oracle(errors)
    assert 1e-5 < abs(empirical - closed) < 1e-3


def test_random_ranker_anchors():
    """Uninformative uncertainty gives AURC -> r and AUGRC -> r/2 for error
    rate r: the anchors every reported number is read against."""
    for r in (0.2, 0.35):
        a, g = [], []
        for s in range(100):
            rg = np.random.default_rng(s)
            errors = (rg.random(5000) < r).astype(float)
            uncertainty = rg.random(5000)
            a.append(aurc_from_scores(uncertainty, errors))
            g.append(augrc_from_scores(uncertainty, errors))
        assert np.mean(a) == pytest.approx(r, abs=3e-3)
        assert np.mean(g) == pytest.approx(r / 2, abs=3e-3)


def test_all_correct_is_zero_risk_everywhere():
    """No errors means no risk at any coverage; guards the empty-tail edge
    where a naive implementation divides 0/0."""
    rng = np.random.default_rng(1)
    errors = np.zeros(500)
    uncertainty = rng.random(500)
    assert aurc_from_scores(uncertainty, errors) == 0.0
    assert augrc_from_scores(uncertainty, errors) == 0.0
    assert eaurc_from_scores(uncertainty, errors) == 0.0


def test_all_wrong_is_risk_one():
    """An all-wrong model has selective risk 1 at every operating point, so
    AURC must be exactly 1.0. A prepended (0, 0) point -- asserting zero risk
    at zero coverage -- drags the area below the true risk."""
    rng = np.random.default_rng(2)
    errors = np.ones(200)
    assert aurc_from_scores(rng.random(200), errors) == 1.0


def test_reversed_ranker_is_worse_than_random():
    """Negating a good score must give an AURC worse than chance: the regime
    a confidence attack drives the model into, where abstention rejects
    preferentially the predictions that were right."""
    rng = np.random.default_rng(2)
    errors = (rng.random(2000) < 0.2).astype(float)
    uncertainty = errors + rng.random(2000) * 0.01  # good: wrong rows most uncertain
    good = aurc_from_scores(uncertainty, errors)
    reversed_ = aurc_from_scores(-uncertainty, errors)
    random_ = aurc_from_scores(rng.random(2000), errors)
    assert good < random_ < reversed_


def _brute_augrc(uncertainty, errors):
    """The definition, written the slow obvious way: mean over k of
    (loss of the k most-confident rows) / n."""
    u = np.asarray(uncertainty, float)
    loss = np.asarray(errors, float)
    n = len(u)
    loss_s = loss[np.argsort(u, kind="stable")]
    return float(np.mean([loss_s[:k].sum() / n for k in range(1, n + 1)]))


def test_augrc_linear_matches_quadratic_bruteforce():
    """With distinct scores the block-weighted average equals the textbook
    mean over k = 1..n, which keeps these AURCs comparable to published
    ones."""
    rng = np.random.default_rng(3)
    errors = (rng.random(400) < 0.3).astype(float)
    uncertainty = rng.random(400)  # continuous: no ties
    assert augrc_from_scores(uncertainty, errors) == pytest.approx(
        _brute_augrc(uncertainty, errors), abs=1e-12
    )


def test_generalized_equals_coverage_times_selective():
    """R_gen == cov * R_sel elementwise -- catches the off-by-one in the
    operating-point index that otherwise shows up as a plausible-looking
    wrong curve."""
    rng = np.random.default_rng(4)
    errors = (rng.random(400) < 0.3).astype(float)
    uncertainty = rng.random(400)
    cov, sel, gen, _ = rc_curve(uncertainty, errors)
    assert np.abs(gen - cov * sel).max() < 1e-15
    assert cov[-1] == 1.0  # full coverage is always an operating point
    assert sel[-1] == pytest.approx(errors.mean())
    assert np.all(np.diff(cov) > 0)  # ties collapsed: strictly increasing


def test_permutation_invariance_under_heavy_ties():
    """THE tie test, and the regression test for the previous estimator.
    With 10 distinct score levels a naive cumulative AURC varies across row
    permutations of the same predictions -- larger than the effects being
    reported. Saturated softmax manufactures those tie blocks silently."""
    rng = np.random.default_rng(5)
    errors = (rng.random(600) < 0.25).astype(float)
    uncertainty = np.round(rng.random(600), 1)  # 10 distinct levels

    vals, naive = [], []
    for s in range(8):
        p = np.random.default_rng(s).permutation(600)
        vals.append(
            (
                aurc_from_scores(uncertainty[p], errors[p]),
                augrc_from_scores(uncertainty[p], errors[p]),
            )
        )
        # the naive convention: average over every rank, ties not collapsed
        order = np.argsort(uncertainty[p], kind="stable")
        cum = np.cumsum(errors[p][order])
        naive.append(float((cum / np.arange(1, 601)).mean()))

    assert max(v[0] for v in vals) - min(v[0] for v in vals) == 0.0
    assert max(v[1] for v in vals) - min(v[1] for v in vals) == 0.0
    spread = max(naive) - min(naive)  # the defended-against bug, measured
    assert 5e-4 < spread < 5e-2, f"fixture drifted: naive spread {spread:.2e}"


def test_non_finite_scores_raise():
    with pytest.raises(ValueError, match="non-finite"):
        aurc_from_scores(np.array([0.1, np.nan]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="empty"):
        aurc_from_scores(np.array([]), np.array([]))


# ------------------------------------------------------------- thresholds


def test_thresholds_define_the_accepted_set():
    """The 4th return value must be the threshold that reproduces the
    operating point: accepting every row with uncertainty <= thresholds[j]
    has to give back exactly coverage[j] and selective_risk[j]. Without it
    the curve cannot be turned into a deployable abstention rule -- only
    plotted."""
    rng = np.random.default_rng(11)
    errors = (rng.random(300) < 0.3).astype(float)
    uncertainty = np.round(rng.random(300), 2)  # ties, deliberately

    cov, sel, gen, thr = rc_curve(uncertainty, errors)
    assert thr.shape == cov.shape
    assert np.all(np.diff(thr) > 0)  # one threshold per distinct value

    for j in range(len(thr)):
        accepted = uncertainty <= thr[j]
        assert accepted.sum() == pytest.approx(cov[j] * 300)
        assert errors[accepted].mean() == pytest.approx(sel[j], abs=1e-15)
        assert errors[accepted].sum() / 300 == pytest.approx(gen[j], abs=1e-15)


# ------------------------------------------------- operating-point counting


def test_n_operating_points_counts_distinct_scores():
    """The curve's resolution: n distinct scores means n usable operating
    points, and collapsing them into k levels leaves exactly k."""
    rng = np.random.default_rng(12)
    assert n_operating_points(rng.random(500)) == 500
    assert n_operating_points(np.round(rng.random(500), 1)) == 11  # 0.0 .. 1.0
    assert n_operating_points(np.zeros(500)) == 1

    scores = rng.random(500)
    assert n_operating_points(scores) == len(rc_curve(scores, np.zeros(500))[0])


def test_n_operating_points_detects_float32_saturation():
    """THE guard. Confidence computed in float32 saturates to exactly 1.0 on
    wide margins, so ``1 - max_prob`` cancels to exactly 0.0 and every row
    lands in ONE tie block: n_operating_points collapses from n to 1 and the
    AURC silently degenerates into the plain error rate, while the float64
    score still ranks every row apart. An AURC reported without this count
    beside it cannot be distinguished from that degenerate case."""
    rng = np.random.default_rng(13)
    n = 400
    # binary logits with margins in [18, 26]: still distinct in float64, all
    # saturated in float32, since 1 - exp(-18) rounds to 1.0 at fp32's
    # 2^-24 resolution but not at float64's 2^-53
    margin = rng.uniform(18.0, 26.0, size=n)
    logits = torch.tensor(np.stack([np.zeros(n), margin], axis=1))

    unc32 = 1.0 - torch.softmax(logits.float(), dim=1).max(dim=1).values
    unc64 = 1.0 - torch.softmax(logits.double(), dim=1).max(dim=1).values
    errors = (rng.random(n) < 0.2).astype(float)

    assert n_operating_points(unc32.numpy()) == 1  # every row tied at 0.0
    assert n_operating_points(unc64.numpy()) == n  # every row still ranked

    # ... and that is not cosmetic: with one operating point the AURC IS the
    # error rate, i.e. the ranking has been thrown away entirely.
    assert aurc_from_scores(unc32.numpy(), errors) == pytest.approx(errors.mean())
    assert aurc_from_scores(unc64.numpy(), errors) != pytest.approx(errors.mean())


def test_float64_upcast_does_not_rescue_a_saturated_float32_score():
    """The honest limit of the float64 rule: casting an already-saturated
    fp32 score to float64 restores nothing, so the softmax itself has to be
    computed in float64 upstream. This is why n_operating_points is the
    detector and not the fix."""
    rng = np.random.default_rng(14)
    margin = rng.uniform(18.0, 26.0, size=200)
    logits = torch.tensor(np.stack([np.zeros(200), margin], axis=1))
    late_upcast = (
        (1.0 - torch.softmax(logits.float(), dim=1).max(dim=1).values).double().numpy()
    )
    assert n_operating_points(late_upcast) == 1


# ------------------------------------------------------ operating points


def _tied_fixture(n=100, levels=10, seed=15):
    """n rows over `levels` equally sized tie blocks: coverage is achievable
    only on the 1/levels grid."""
    rng = np.random.default_rng(seed)
    uncertainty = np.repeat(np.arange(levels, dtype=float), n // levels)
    errors = (rng.random(n) < 0.3).astype(float)
    return uncertainty, errors


def test_operating_point_reports_achieved_coverage_not_the_target():
    """With a tie block straddling the target, the reachable coverage sits
    BELOW it. Printing 'risk@cov0.85' for a point actually taken at 0.8 is
    the lie this return value exists to prevent."""
    uncertainty, errors = _tied_fixture()
    cov, sel, gen, thr = operating_point_at_coverage(uncertainty, errors, 0.85)

    assert cov == pytest.approx(0.8)  # achieved, not the 0.85 requested
    accepted = uncertainty <= thr
    assert accepted.sum() == 80
    assert sel == pytest.approx(errors[accepted].mean())
    assert gen == pytest.approx(sel * cov)
    assert risk_at_coverage(uncertainty, errors, 0.85) == sel


def test_achieved_coverage_can_exceed_the_target_when_everything_ties():
    """A fully saturated score has ONE operating point, at coverage 1.0.
    Asking for 0.8 then returns 1.0 -- above the target -- and the risk is
    the plain error rate. Reporting 0.8 there would be pure fiction."""
    uncertainty = np.zeros(100)
    errors = np.concatenate([np.ones(25), np.zeros(75)])
    cov, sel, _, _ = operating_point_at_coverage(uncertainty, errors, 0.8)
    assert cov == 1.0
    assert sel == pytest.approx(0.25)


def test_target_coverage_outside_the_unit_interval_raises():
    uncertainty, errors = _tied_fixture()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="target_coverage"):
            operating_point_at_coverage(uncertainty, errors, bad)


def test_coverage_at_risk_is_monotone_and_bounded():
    """The deployment question -- how much traffic can be auto-decided under
    an SLA -- must be monotone in the SLA, hit 1.0 when the SLA is vacuous,
    and return 0.0 (not the smallest coverage) when nothing qualifies."""
    rng = np.random.default_rng(16)
    errors = (rng.random(500) < 0.2).astype(float)
    uncertainty = errors + rng.random(500) * 0.5  # informative ranking

    coverages = [
        coverage_at_risk(uncertainty, errors, r) for r in (0.0, 0.05, 0.1, 0.2)
    ]
    assert coverages == sorted(coverages)
    assert coverage_at_risk(uncertainty, errors, 1.0) == 1.0
    assert coverage_at_risk(uncertainty, errors, -1.0) == 0.0

    # whatever it returns must actually satisfy the SLA
    cov, sel, _, _ = rc_curve(uncertainty, errors)
    achieved = coverage_at_risk(uncertainty, errors, 0.05)
    assert sel[cov == achieved][0] <= 0.05 + 1e-12


# ------------------------------------------------------------- weightings


def test_block_and_uniform_weights_agree_without_ties():
    """With distinct scores every block has size 1, so the two conventions
    are the same number -- which is what makes 'block' the safe default."""
    rng = np.random.default_rng(17)
    errors = (rng.random(400) < 0.3).astype(float)
    uncertainty = rng.random(400)
    for fn in (aurc_from_scores, augrc_from_scores, eaurc_from_scores):
        assert fn(uncertainty, errors, "block") == pytest.approx(
            fn(uncertainty, errors, "uniform"), abs=1e-12
        )


def _uneven_tied_fixture():
    """100 singletons at low uncertainty, all correct, then ONE 900-row tie
    block, all wrong: 101 operating points of wildly unequal weight."""
    uncertainty = np.concatenate([np.arange(100, dtype=float), np.full(900, 1000.0)])
    errors = np.concatenate([np.zeros(100), np.ones(900)])
    return uncertainty, errors


def test_block_and_uniform_weights_diverge_under_ties():
    """Under ties they are genuinely different estimators: 'uniform' lets a
    single-row operating point count as much as a 900-row block, so the
    90% of the data that is wrong contributes 1/101 of the answer. The gap
    here is 0.80 against 0.01 -- two orders above the effects being
    reported -- so a table that does not say which convention it used is
    unreadable."""
    uncertainty, errors = _uneven_tied_fixture()
    block = aurc_from_scores(uncertainty, errors, "block")
    uniform = aurc_from_scores(uncertainty, errors, "uniform")

    assert n_operating_points(uncertainty) == 101 < uncertainty.size
    assert block == pytest.approx(0.81)  # 900 wrong rows carry their weight
    assert uniform < 0.01  # ... and here they carry 1/101 of it
    assert block - uniform > 0.5


def test_unknown_weighting_raises():
    rng = np.random.default_rng(19)
    with pytest.raises(ValueError, match="weights"):
        aurc_from_scores(rng.random(10), np.zeros(10), "trapezoid")


# --------------------------------------------------------- torchmetrics wrap


def _feed(metric, uncertainty, errors, batches=4, dtype=torch.float64):
    """Drive the Metric interface: 2-class probs whose argmax correctness
    matches `errors` against all-zero targets."""
    n = len(uncertainty)
    correct = 1.0 - errors
    probs = np.stack([0.1 + 0.8 * correct, 0.9 - 0.8 * correct], axis=1)
    targets = np.zeros(n, dtype=np.int64)
    for chunk in np.array_split(np.arange(n), batches):
        metric.update(
            torch.tensor(probs[chunk], dtype=torch.float32),
            torch.tensor(targets[chunk]),
            torch.tensor(uncertainty[chunk], dtype=dtype),
        )
    return metric.compute()


def test_metric_classes_match_pure_functions():
    """Batched accumulation through the Metric interface must equal the pure
    single-pass value."""
    rng = np.random.default_rng(6)
    errors = (rng.random(500) < 0.3).astype(float)
    uncertainty = np.round(rng.random(500), 1)  # with ties, deliberately

    assert _feed(AURiskCoverage(), uncertainty, errors) == pytest.approx(
        aurc_from_scores(uncertainty, errors), abs=1e-12
    )
    assert _feed(AUGRiskCoverage(), uncertainty, errors) == pytest.approx(
        augrc_from_scores(uncertainty, errors), abs=1e-12
    )
    assert _feed(EAURiskCoverage(), uncertainty, errors) == pytest.approx(
        eaurc_from_scores(uncertainty, errors), abs=1e-12
    )


def test_metric_is_row_order_invariant():
    """Feeding the same predictions in a different row order must give the
    same AURC to the last bit -- the previous estimator failed this."""
    rng = np.random.default_rng(7)
    errors = (rng.random(600) < 0.25).astype(float)
    uncertainty = np.round(rng.random(600), 1)

    values = []
    for s in range(4):
        p = np.random.default_rng(s).permutation(600)
        values.append(_feed(AURiskCoverage(), uncertainty[p], errors[p]))
    assert max(values) - min(values) == 0.0


def test_metric_upcasts_the_score_to_float64_at_the_boundary():
    """A float32 score handed to the Metric must be stored as float64: the
    ranking downstream is only as fine-grained as the score, so the cast has
    to happen before anything accumulates, not inside compute()."""
    rng = np.random.default_rng(20)
    errors = (rng.random(200) < 0.3).astype(float)
    uncertainty = rng.random(200)

    metric = AURiskCoverage()
    value = _feed(metric, uncertainty, errors, dtype=torch.float32)
    assert all(s.dtype == torch.float64 for s in metric.scores)
    assert all(p.dtype == torch.float64 for p in metric.probs)
    assert value == pytest.approx(
        aurc_from_scores(np.float32(uncertainty).astype(np.float64), errors), abs=1e-12
    )


def test_operating_point_metrics_match_the_pure_functions():
    """n_operating_points, risk@cov and achieved_cov go through the Metric
    interface so they land in the same logged row as the AURC they qualify."""
    uncertainty, errors = _tied_fixture(n=500, levels=10, seed=21)

    assert _feed(NumOperatingPoints(), uncertainty, errors) == 10.0
    assert _feed(RiskAtCoverage(0.85), uncertainty, errors) == pytest.approx(
        risk_at_coverage(uncertainty, errors, 0.85), abs=1e-12
    )
    assert _feed(AchievedCoverage(0.85), uncertainty, errors) == pytest.approx(0.8)


def test_collection_logs_the_guard_beside_the_aurc():
    """The collection must carry n_operating_points and a risk / achieved
    coverage PAIR per target, all from one accumulation: an AURC logged
    without its operating-point count, or a risk@cov without the coverage it
    actually reached, is not reportable."""
    uncertainty, errors = _tied_fixture(n=500, levels=10, seed=22)
    collection = get_selective_classification_metrics(coverages=(0.8, 0.5))

    assert set(collection.keys()) == {
        "aurc",
        "augrc",
        "eaurc",
        "n_operating_points",
        "risk@cov0_8",
        "achieved_cov@0_8",
        "risk@cov0_5",
        "achieved_cov@0_5",
    }

    out = _feed(collection, uncertainty, errors)
    assert float(out["aurc"]) == pytest.approx(aurc_from_scores(uncertainty, errors))
    assert float(out["n_operating_points"]) == 10.0
    for target, key in ((0.8, "0_8"), (0.5, "0_5")):
        assert float(out[f"risk@cov{key}"]) == pytest.approx(
            risk_at_coverage(uncertainty, errors, target), abs=1e-12
        )
        assert float(out[f"achieved_cov@{key}"]) == pytest.approx(target)


def test_collection_weighting_is_threaded_to_every_area_metric():
    """Choosing 'uniform' must reach all three areas, or a table would mix
    two estimators under one heading."""
    uncertainty, errors = _uneven_tied_fixture()
    out = _feed(
        get_selective_classification_metrics(weights="uniform"), uncertainty, errors
    )
    for key, fn in (
        ("aurc", aurc_from_scores),
        ("augrc", augrc_from_scores),
        ("eaurc", eaurc_from_scores),
    ):
        assert float(out[key]) == pytest.approx(
            fn(uncertainty, errors, "uniform"), abs=1e-12
        )
    assert float(out["aurc"]) != pytest.approx(aurc_from_scores(uncertainty, errors))
