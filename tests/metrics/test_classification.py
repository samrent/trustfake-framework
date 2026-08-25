"""Top-1 and macro accuracy are two different numbers, and the headline
claims are made about the first. No GPU, no data, no network."""

import pytest
import torch

from trustfake.metrics.evaluation.classification import (
    get_multiclass_classification_metrics,
)

NUM_CLASSES = 3


def _scores(preds, targets):
    collection = get_multiclass_classification_metrics(NUM_CLASSES)
    collection.update(torch.tensor(preds), torch.tensor(targets))
    return {k: float(v) for k, v in collection.compute().items()}


def _imbalanced_targets():
    """800 real, 100 synthetic, 100 tampered -- SID-Set's shape."""
    return [0] * 800 + [1] * 100 + [2] * 100


def test_top1_accuracy_is_reported_alongside_macro():
    collection = get_multiclass_classification_metrics(NUM_CLASSES)
    assert {"accuracy", "accuracy_top1"} <= set(collection.keys())


def test_top1_is_the_fraction_of_rows_predicted_correctly():
    targets = _imbalanced_targets()
    preds = list(targets)
    preds[:50] = [1] * 50  # 50 wrong rows out of 1000

    assert _scores(preds, targets)["accuracy_top1"] == pytest.approx(0.95)


def test_same_top1_accuracy_can_hide_a_large_macro_gap():
    """THE reason both are reported. Two prediction sets with IDENTICAL
    top-1 accuracy differ by 0.15 in macro accuracy, because 50 errors
    spread over the majority class cost almost nothing per-class while the
    same 50 errors inside a minority class halve its recall. A headline
    claim like 'accuracy identical to 4 dp under the attack' is a top-1
    claim, and quoting the macro number under it changes what was proven."""
    targets = _imbalanced_targets()

    errors_in_majority = list(targets)
    errors_in_majority[:50] = [1] * 50  # 50 of the 800 reals mislabelled

    errors_in_minority = list(targets)
    errors_in_minority[800:850] = [0] * 50  # 50 of the 100 synthetics

    majority = _scores(errors_in_majority, targets)
    minority = _scores(errors_in_minority, targets)

    assert majority["accuracy_top1"] == minority["accuracy_top1"] == pytest.approx(0.95)
    assert majority["accuracy"] == pytest.approx((750 / 800 + 1 + 1) / 3)
    assert minority["accuracy"] == pytest.approx((1 + 50 / 100 + 1) / 3)
    assert majority["accuracy"] - minority["accuracy"] > 0.1


def test_macro_accuracy_is_unchanged_by_the_addition():
    """The new key must not disturb the existing one: macro accuracy is
    still the mean of the per-class recalls."""
    targets = _imbalanced_targets()
    preds = list(targets)
    preds[800:900] = [0] * 100  # the whole synthetic class lost

    scores = _scores(preds, targets)
    assert scores["accuracy"] == pytest.approx((1 + 0 + 1) / 3)
    assert scores["accuracy_top1"] == pytest.approx(0.9)
    # macro accuracy IS macro recall, which is the other half of why the
    # top-1 number has to be reported separately
    assert scores["recall"] == pytest.approx(scores["accuracy"])
