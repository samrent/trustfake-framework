"""Calibration unit tests: temperature fitting and ECE/NLL/Brier. No data,
no network, no GPU."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.metrics.calibration import (
    BrierScore,
    NegativeLogLikelihood,
    calibrate_temperature,
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
