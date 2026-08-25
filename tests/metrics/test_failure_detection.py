"""Defences for the two AUROCs: the failure one must not report 0.0 where
it means "undefined", and the detection one must not be confused with it.
No GPU, no data, no network."""

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import BinaryAUROC

from trustfake.metrics.evaluation.failure_detection import (
    DetectionAUROC,
    FailureAUROC,
    get_detection_metrics,
    get_failure_detection_metrics,
)


def _fd(uncertainty, errors, batches=3):
    """Drive FailureAUROC over several batches, as the eval pipe does."""
    metric = FailureAUROC()
    u = torch.as_tensor(uncertainty, dtype=torch.float32)
    e = torch.as_tensor(errors, dtype=torch.long)
    for chunk in torch.chunk(torch.arange(len(u)), batches):
        metric.update(u[chunk], e[chunk])
    return float(metric.compute())


def _binary_auroc(uncertainty, errors):
    metric = BinaryAUROC()
    metric.update(
        torch.as_tensor(uncertainty, dtype=torch.float32),
        torch.as_tensor(errors, dtype=torch.long),
    )
    return float(metric.compute())


# ------------------------------------------------------------- the NaN guard


def test_no_errors_is_nan_not_zero():
    """A split the model got entirely right has no failure to detect, so
    the AUROC is undefined. torchmetrics reports 0.0, which in this column
    reads as PERFECTLY INVERTED failure detection -- the signature of a
    successful confidence attack -- on a split that simply had nothing to
    rank. Averaged into a table, that 0.0 manufactures the attack."""
    rng = np.random.default_rng(0)
    uncertainty = rng.random(200)
    errors = np.zeros(200)

    assert np.isnan(_fd(uncertainty, errors))
    assert _binary_auroc(uncertainty, errors) == 0.0  # the defended-against value


def test_all_errors_is_nan_not_zero():
    """Same on the other side: nothing correct to rank the errors against."""
    rng = np.random.default_rng(1)
    uncertainty = rng.random(200)
    errors = np.ones(200)

    assert np.isnan(_fd(uncertainty, errors))
    assert _binary_auroc(uncertainty, errors) == 0.0


def test_constant_confidence_is_nan_not_chance():
    """A saturated score ranks nothing, so its failure AUROC is undefined --
    not 0.5. torchmetrics reports 0.5, which reads as 'chance-level
    ranking' and is indistinguishable from a model that ranks badly; the
    real state is that there is no ranking at all, exactly what
    n_operating_points == 1 says on the AURC side."""
    rng = np.random.default_rng(2)
    errors = (rng.random(200) < 0.3).astype(float)
    uncertainty = np.full(200, 0.5)

    assert np.isnan(_fd(uncertainty, errors))
    assert _binary_auroc(uncertainty, errors) == 0.5  # the defended-against value


def test_guard_does_not_fire_on_a_normal_split():
    """The guard must be inert whenever the AUROC exists, and the value must
    still be the AUROC -- checked against sklearn, not against torchmetrics
    alone."""
    rng = np.random.default_rng(3)
    errors = (rng.random(500) < 0.25).astype(float)
    uncertainty = errors * 0.4 + rng.random(500) * 0.6  # informative

    value = _fd(uncertainty, errors)
    assert not np.isnan(value)
    assert value == pytest.approx(roc_auc_score(errors, uncertainty), abs=1e-6)
    assert value > 0.6


def test_guard_fires_through_the_collection():
    """The pipe logs the collection, not the class, so the guard has to
    survive the MetricCollection wrapper."""
    collection = get_failure_detection_metrics()
    assert set(collection.keys()) == {"fd_auroc"}
    collection.update(torch.rand(50), torch.zeros(50, dtype=torch.long))
    assert np.isnan(float(collection.compute()["fd_auroc"]))


def test_failure_auroc_upcasts_the_score():
    """float32 in, float64 stored: the ranking is only as fine-grained as
    the score, so the cast happens before accumulation."""
    metric = FailureAUROC()
    metric.update(torch.rand(64, dtype=torch.float32), torch.randint(0, 2, (64,)))
    assert all(p.dtype == torch.float64 for p in metric.preds)


# -------------------------------------------------- detection vs failure


def _three_class_batch(n_per_class=200, seed=4):
    """A model that separates real from fake perfectly but confuses
    synthetic with tampered, with an uncertainty score that says nothing
    about which rows it got wrong."""
    rng = np.random.default_rng(seed)
    n = 3 * n_per_class
    targets = np.repeat([0, 1, 2], n_per_class)

    probs = np.zeros((n, 3))
    probs[targets == 0] = [0.90, 0.05, 0.05]
    fake = targets != 0
    # p(real) = 0.02 for every fake, but the mass is split between the two
    # fake classes at random, so half of them are mislabelled
    split = rng.random(int(fake.sum()))
    probs[fake, 0] = 0.02
    probs[fake, 1] = 0.98 * split
    probs[fake, 2] = 0.98 * (1.0 - split)

    preds = probs.argmax(axis=1)
    errors = (preds != targets).astype(float)
    uncertainty = rng.random(n)  # uninformative about the model's own errors
    return probs, targets, errors, uncertainty


def test_detection_auroc_is_not_failure_auroc():
    """The cheap embarrassing error, made impossible to commit silently: on
    ONE batch the detection AUROC is 1.0 (real vs fake is perfectly ranked)
    while the failure AUROC is chance (the model has no idea which of its
    own answers are wrong). Reporting either under a bare 'AUROC' heading
    is a different claim about the system."""
    probs, targets, errors, uncertainty = _three_class_batch()

    detection = DetectionAUROC()
    detection.update(torch.tensor(probs), torch.tensor(targets))
    detection_value = float(detection.compute())
    failure_value = _fd(uncertainty, errors)

    assert detection_value == pytest.approx(1.0)
    assert failure_value == pytest.approx(0.5, abs=0.05)
    assert detection_value - failure_value > 0.4
    assert errors.sum() > 0  # the split really does contain failures


def test_detection_auroc_folds_every_non_real_class_into_fake():
    """SID-Set's synthetic and tampered are both fake, so the detection
    label is ``target != real_class`` and the score is 1 - P(real) -- the
    repo-wide convention in metrics.moderation. Checked against sklearn on
    the binarised labels."""
    probs, targets, _, _ = _three_class_batch(seed=5)
    metric = DetectionAUROC(real_class=0)
    metric.update(torch.tensor(probs), torch.tensor(targets))

    expected = roc_auc_score((targets != 0).astype(int), 1.0 - probs[:, 0])
    assert float(metric.compute()) == pytest.approx(expected, abs=1e-6)


def test_detection_auroc_inverts_when_the_score_is_negated():
    """A detector that ranks reals above fakes scores 0, not 0.5: the metric
    is directional, which is what makes it usable as an attack indicator."""
    targets = torch.tensor([0, 0, 1, 2])  # two reals, one synthetic, one tampered
    good = torch.tensor(
        [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.2, 0.0, 0.8]]
    )
    # p(real) high exactly on the fakes and low on the reals
    inverted = torch.tensor(
        [[0.1, 0.9, 0.0], [0.2, 0.0, 0.8], [0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]
    )

    for probs, expected in ((good, 1.0), (inverted, 0.0)):
        metric = DetectionAUROC()
        metric.update(probs.double(), targets)
        assert float(metric.compute()) == pytest.approx(expected)


def test_detection_auroc_is_nan_when_a_split_has_one_class_only():
    """An all-real (or all-fake) split has no detection AUROC. Same
    argument as the failure guard: 0.0 there would read as an inverted
    detector."""
    metric = DetectionAUROC()
    probs = torch.tensor([[0.9, 0.05, 0.05], [0.7, 0.2, 0.1]]).double()
    metric.update(probs, torch.tensor([0, 0]))
    assert np.isnan(float(metric.compute()))


def test_detection_collection_is_separate_from_failure_detection():
    """They consume different inputs -- (probs, targets) against
    (uncertainty, errors) -- so they cannot share a collection, and the
    separation is the point rather than an accident of plumbing."""
    detection = get_detection_metrics()
    assert set(detection.keys()) == {"detection_auroc"}
    assert set(get_failure_detection_metrics().keys()) == {"fd_auroc"}

    probs, targets, _, _ = _three_class_batch(seed=6)
    detection.update(torch.tensor(probs), torch.tensor(targets))
    assert float(detection.compute()["detection_auroc"]) == pytest.approx(1.0)
