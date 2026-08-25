"""Defences for the selective-classification estimator.

Each test guards one specific way an AURC number lies. Pure numpy paths are
tested directly; the torchmetrics wrappers get their own integration tests.
No GPU, no data, no network.
"""

import numpy as np
import pytest
import torch

from trustfake.metrics.evaluation.selective_classification import (
    AUGRiskCoverage,
    AURiskCoverage,
    EAURiskCoverage,
    augrc_from_scores,
    aurc_from_scores,
    aurc_oracle,
    eaurc_from_scores,
    rc_curve,
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
    cov, sel, gen = rc_curve(uncertainty, errors)
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


# --------------------------------------------------------- torchmetrics wrap


def _feed(metric, uncertainty, errors, batches=4):
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
            torch.tensor(uncertainty[chunk], dtype=torch.float64),
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
