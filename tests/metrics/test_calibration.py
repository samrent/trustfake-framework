"""Calibration unit tests: temperature fitting and ECE/NLL/Brier. No data,
no network, no GPU."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassCalibrationError

from trustfake.metrics.calibration import (
    BrierScore,
    ExpectedCalibrationError,
    NegativeLogLikelihood,
    calibrate_temperature,
    ece_from_scores,
    fit_temperature,
    get_calibration_metrics,
)
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper


def _overconfident(n=5000, c=3, scale=4.0, seed=0):
    """Accurate but over-confident logits: aligned with labels, then inflated."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, c, (n,), generator=g)
    logits = torch.zeros(n, c)
    logits[torch.arange(n), y] = 2.0
    logits += torch.randn(n, c, generator=g) * 0.8
    return logits * scale, y


def test_fit_temperature_minimises_nll_and_is_monotone_in_overconfidence():
    """Fitting always reduces (never increases) NLL vs T=1, and a more
    over-confident model needs a larger temperature."""

    def nll_at(logits, y, temp):
        m = NegativeLogLikelihood()
        m.update(torch.softmax(logits / temp, 1), y)
        return float(m.compute())

    temps = []
    for scale in (1.0, 2.0, 4.0):
        logits, y = _overconfident(scale=scale)
        temp = fit_temperature(logits, y)
        assert nll_at(logits, y, temp) <= nll_at(logits, y, 1.0) + 1e-6
        temps.append(temp)
    assert temps[0] < temps[1] < temps[2]


def test_fit_temperature_cools_an_overconfident_model():
    """An over-confident model needs T > 1, and it must reduce NLL."""
    logits, y = _overconfident(scale=4.0)
    temp = fit_temperature(logits, y)
    assert temp > 1.2

    def nll(temp):
        m = NegativeLogLikelihood()
        m.update(torch.softmax(logits / temp, 1), y)
        return float(m.compute())

    assert nll(temp) < nll(1.0)


def test_temperature_preserves_accuracy_but_moves_calibration():
    """T cannot change the argmax (accuracy identical), but ECE/NLL/Brier move.
    In 3-class this is a real effect -- temperature is a live variable, unlike
    the binary case where MSP is monotone in the single margin."""
    logits, y = _overconfident(scale=4.0)
    temp = fit_temperature(logits, y)

    acc1 = (logits.argmax(1) == y).float().mean()
    acc_t = ((logits / temp).argmax(1) == y).float().mean()
    assert torch.equal(acc1, acc_t)

    m1 = get_calibration_metrics(3)
    m1.update(torch.softmax(logits, 1), y)
    s1 = m1.compute()
    m_t = get_calibration_metrics(3)
    m_t.update(torch.softmax(logits / temp, 1), y)
    s_t = m_t.compute()
    for k in ("ece", "nll", "brier"):
        assert float(s1[k]) != float(s_t[k])


def test_nll_matches_cross_entropy():
    """NLL over probs must equal cross-entropy over the corresponding logits."""
    logits, y = _overconfident(scale=2.0)
    probs = torch.softmax(logits, 1)
    m = NegativeLogLikelihood()
    m.update(probs, y)
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert float(m.compute()) == pytest.approx(float(ce), abs=1e-5)


def test_brier_bounds_and_perfect_case():
    """Brier is 0 for perfect one-hot predictions and > 0 otherwise."""
    y = torch.tensor([0, 1, 2, 1])
    perfect = torch.nn.functional.one_hot(y, 3).float()
    m = BrierScore(num_classes=3)
    m.update(perfect, y)
    assert float(m.compute()) == pytest.approx(0.0, abs=1e-9)

    uniform = torch.full((4, 3), 1 / 3)
    m2 = BrierScore(num_classes=3)
    m2.update(uniform, y)
    assert float(m2.compute()) == pytest.approx(4 / 6, abs=1e-6)  # 3*(1/3)^2... = 2/3


def test_metrics_accumulate_in_batches():
    """Batched updates must equal a single-pass update."""
    logits, y = _overconfident(scale=3.0)
    probs = torch.softmax(logits, 1)
    single = get_calibration_metrics(3)
    single.update(probs, y)
    s = single.compute()

    batched = get_calibration_metrics(3)
    for chunk in torch.split(torch.arange(len(y)), 512):
        batched.update(probs[chunk], y[chunk])
    b = batched.compute()
    for k in ("ece", "nll", "brier"):
        assert float(s[k]) == pytest.approx(float(b[k]), abs=1e-5)


def test_calibrate_temperature_over_a_loader_fits_and_resets_wrapper():
    """The loader-collecting helper: it fits T over the calib loader and
    returns it, resetting the wrapper's own temperature to 1.0 during the
    pass so fitting always sees raw logits (the caller sets the result)."""
    torch.manual_seed(0)
    wrapper = BaseWrapper(
        normalization_layer=nn.Identity(),
        model=nn.Linear(12, 3),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )
    wrapper.temperature = 5.0  # a stale value the helper must ignore while fitting
    x = torch.randn(64, 12) * 3.0
    y = x[:, :3].argmax(1)  # labels correlated with inputs so a finite T exists
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    temp = calibrate_temperature(wrapper, loader, device="cpu")
    assert 0.01 <= temp <= 100.0
    assert wrapper.temperature == 1.0  # helper returns T; caller assigns it


# --------------------------------------------------------------- ECE variants


def _calibrated_binary(n=20000, seed=0):
    """Perfectly calibrated binary confidences: conf ~ U(0.5, 1) and the
    prediction is right with probability exactly conf."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, n)
    correct = (rng.random(n) < conf).astype(float)
    return conf, correct


def test_default_variant_reproduces_the_torchmetrics_ece():
    """The new knobs must not move the default number: equal-width bins over
    [0, 1] with the L1 norm is exactly what the pipe reports today, so the
    variants are an addition and not a silent redefinition."""
    logits, y = _overconfident(scale=1.0)  # unsaturated: no confidence at 1.0
    probs = torch.softmax(logits.double(), 1)
    assert float(probs.max()) < 1.0

    reference = MulticlassCalibrationError(num_classes=3, n_bins=15, norm="l1")
    reference.update(probs, y)
    mine = ExpectedCalibrationError()
    mine.update(probs, y)

    # the residual ~1e-7 gap is torchmetrics binning in float32; this
    # implementation stays in float64, which is the only difference
    assert float(mine.compute()) == pytest.approx(float(reference.compute()), abs=1e-6)


def test_equal_mass_and_equal_width_disagree_on_a_skewed_distribution():
    """Both are called 'ECE' and they are not the same estimator. A trained
    classifier's confidence piles up near 1.0, so equal-width bins put most
    of the mass in one bin and report that bin's average gap as if it were
    resolved, while equal-mass spends its resolution where the data is."""
    logits, y = _overconfident(scale=4.0)
    probs = torch.softmax(logits, 1)

    values = {}
    for scheme in ("equal_width", "equal_mass"):
        metric = ExpectedCalibrationError(scheme=scheme)
        metric.update(probs, y)
        values[scheme] = float(metric.compute())

    # the top equal-width bin alone holds most of the data
    conf = probs.max(1).values.double().numpy()
    assert (conf > 14 / 15).mean() > 0.5
    assert abs(values["equal_width"] - values["equal_mass"]) > 1e-3


def test_binary_domain_stops_wasting_half_the_bins():
    """Top-label confidence for 2 classes lives in [0.5, 1], so 7 of 15
    equal-width bins over [0, 1] can never be filled and the number is not
    comparable to a multiclass ECE binned the same way."""
    conf, correct = _calibrated_binary(n=5000, seed=1)

    edges = np.linspace(0.0, 1.0, 16)
    filled = np.unique(np.clip(np.digitize(conf, edges[1:-1]), 0, 14))
    assert len(filled) == 8  # 7 bins below 0.5 are structurally empty

    over_unit = ece_from_scores(conf, correct, 15, "equal_width", (0.0, 1.0))
    over_binary = ece_from_scores(conf, correct, 15, "equal_width", (0.5, 1.0))
    assert over_unit != pytest.approx(over_binary)


def test_ece_binning_bias_grows_with_the_bin_count():
    """ECE is biased upward by binning even when calibration is perfect, and
    the bias grows with the number of bins -- so a bin count is part of the
    number and two ECEs at different bin counts are not comparable. This is
    also why NLL and Brier are reported beside it."""
    conf, correct = _calibrated_binary()

    values = [
        ece_from_scores(conf, correct, bins, "equal_width", (0.0, 1.0))
        for bins in (15, 50, 200)
    ]
    assert values[0] < values[1] < values[2]
    assert values[2] > 3 * values[0]  # perfectly calibrated data, all of it bias


def test_norms_are_ordered_l1_le_l2_le_max():
    """The three norms answer different questions on the same bins: L1 is
    the mass-weighted mean gap, L2 its root-mean-square, and 'max' is the
    worst bin (MCE), which is not an average at all. Their order is forced
    by Jensen, so a table quoting one under another's name is provably
    reporting a different quantity."""
    logits, y = _overconfident(scale=4.0)
    probs = torch.softmax(logits, 1)
    conf = probs.max(1).values.double().numpy()
    correct = (probs.argmax(1) == y).double().numpy()

    l1 = ece_from_scores(conf, correct, 15, norm="l1")
    l2 = ece_from_scores(conf, correct, 15, norm="l2")
    mx = ece_from_scores(conf, correct, 15, norm="max")
    assert l1 < l2 < mx


def test_ece_rejects_unknown_scheme_or_norm():
    conf, correct = _calibrated_binary(n=100)
    with pytest.raises(ValueError, match="scheme"):
        ece_from_scores(conf, correct, scheme="sturges")
    with pytest.raises(ValueError, match="norm"):
        ece_from_scores(conf, correct, norm="l3")


def test_calibration_collection_reports_the_equal_mass_variant():
    """The pipe logs the collection, so both ECE conventions have to travel
    with the run -- reporting only the flattering one is a choice made at
    write-up time otherwise."""
    logits, y = _overconfident(scale=4.0)
    probs = torch.softmax(logits, 1)

    collection = get_calibration_metrics(3)
    assert set(collection.keys()) == {"ece", "ece_equal_mass", "nll", "brier"}
    collection.update(probs, y)
    out = collection.compute()
    assert float(out["ece"]) != pytest.approx(float(out["ece_equal_mass"]))

    binary = get_calibration_metrics(2)
    assert "ece_binary_domain" in binary.keys()  # only where [0,1] wastes bins


def test_ece_variant_accumulates_in_batches():
    """Quantile edges are computed over ALL rows at compute time, so the
    equal-mass variant must not depend on how the batches were cut."""
    logits, y = _overconfident(scale=3.0)
    probs = torch.softmax(logits, 1)

    single = ExpectedCalibrationError(scheme="equal_mass")
    single.update(probs, y)
    batched = ExpectedCalibrationError(scheme="equal_mass")
    for chunk in torch.split(torch.arange(len(y)), 512):
        batched.update(probs[chunk], y[chunk])
    assert float(single.compute()) == pytest.approx(float(batched.compute()), abs=1e-12)


def test_calibrate_temperature_empty_loader_is_a_noop():
    wrapper = BaseWrapper(
        normalization_layer=nn.Identity(),
        model=nn.Linear(12, 3),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )
    empty = DataLoader(
        TensorDataset(torch.empty(0, 12), torch.empty(0, dtype=torch.long))
    )
    assert calibrate_temperature(wrapper, empty, device="cpu") == 1.0
