"""Device-resolution tests."""

import torch

from trustfake.utils import lightning_accelerator, resolve_device


def test_resolves_to_an_available_device():
    d = resolve_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cuda", "mps", "cpu")


def test_cpu_override_always_honoured():
    assert resolve_device("cpu").type == "cpu"


def test_unavailable_preference_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # requesting cuda when unavailable must not raise; it falls back
    d = resolve_device("cuda")
    assert d.type in ("mps", "cpu")


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("TRUSTFAKE_DEVICE", "cpu")
    assert resolve_device().type == "cpu"


def test_lightning_accelerator_mapping():
    assert lightning_accelerator("cpu") == "cpu"
    assert lightning_accelerator() in ("gpu", "mps", "cpu")
