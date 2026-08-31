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


# ------------------------------------------------------ per-class rows


def test_per_class_rows_carry_sid_set_names():
    """Three classes resolve to SID-Set's names, so a logged key says
    ``recall_tampered`` rather than ``recall_2``."""
    scores = _scores(_imbalanced_targets(), _imbalanced_targets())
    for name in ("real", "synthetic", "tampered"):
        assert scores[f"recall_{name}"] == pytest.approx(1.0)
        assert scores[f"precision_{name}"] == pytest.approx(1.0)


def test_tampered_collapse_is_legible_only_in_the_per_class_rows():
    """THE motivating case: every tampered image predicted synthetic. Both
    averages move -- but into values that could mean many things (macro 2/3
    is also 'a third of every class lost'), and the binary fake-vs-real fold
    forgives the error entirely (a tampered image called synthetic is still
    a caught fake). Only recall_tampered names the class that died, and only
    precision_synthetic shows where its rows went."""
    targets = _imbalanced_targets()
    preds = list(targets)
    preds[900:1000] = [1] * 100  # the whole tampered class read as synthetic

    scores = _scores(preds, targets)
    assert scores["recall_tampered"] == pytest.approx(0.0)
    assert scores["recall_synthetic"] == pytest.approx(1.0)
    assert scores["recall_real"] == pytest.approx(1.0)
    assert scores["precision_synthetic"] == pytest.approx(0.5)
    assert scores["accuracy"] == pytest.approx(2 / 3)


def test_per_class_names_fall_back_to_indices_off_sid_set():
    """A head with a different class count must not silently borrow SID-Set
    semantics."""
    collection = get_multiclass_classification_metrics(4)
    collection.update(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
    keys = set(collection.compute().keys())
    assert {"recall_class_0", "recall_class_3"} <= keys
    assert "recall_real" not in keys


def test_mislabelling_class_names_is_refused():
    """A name list of the wrong length would not crash -- it would caption
    every per-class row with the wrong class, which is worse."""
    with pytest.raises(ValueError, match="class_names"):
        get_multiclass_classification_metrics(3, class_names=["real", "fake"])
