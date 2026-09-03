"""The depth teacher wrapper: preprocessing, output frame, frozenness.

Stub backbone only; `transformers` is never imported here. A guarded smoke
test at the bottom exercises the real HF class from a RANDOM config (no
download) where transformers happens to be installed, and is skipped
elsewhere.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trustfake.depth import DepthTeacher, FakeDepthTeacher, dpt_resize_hw
from trustfake.depth.teacher import DEPTH_FRAME


def test_resize_rule_matches_the_checkpoints_processor():
    assert dpt_resize_hw(224, 224) == (518, 518)
    assert dpt_resize_hw(518, 518) == (518, 518)
    # keep_aspect_ratio: the side closest to scale 1 wins; 480 -> 518 is
    # the smaller change, so both sides scale by 518/480 and round to 14s.
    assert dpt_resize_hw(480, 640) == (518, 686)
    assert dpt_resize_hw(100, 100) == (518, 518)
    # never below one patch
    assert dpt_resize_hw(1, 1, target=1, multiple=14) == (14, 14)
    with pytest.raises(ValueError, match="positive"):
        dpt_resize_hw(0, 10)


def _teacher(**kw):
    kw.setdefault("output_size", 8)
    kw.setdefault("input_size", 16)
    kw.setdefault("multiple", 2)
    return FakeDepthTeacher(**kw)


def test_output_is_in_the_shared_frame_at_the_requested_grid():
    torch.manual_seed(0)
    x = torch.rand(3, 3, 16, 16)
    d = _teacher()(x)
    assert d.shape == (3, 1, 8, 8) and d.dtype == torch.float32
    flat = d.flatten(1)
    assert torch.allclose(flat.median(dim=1).values, torch.zeros(3), atol=1e-6)
    assert torch.allclose(flat.abs().mean(dim=1), torch.ones(3), atol=1e-5)
    assert _teacher().frame == DEPTH_FRAME


def test_preprocessing_applies_its_own_normalisation():
    """The datamodule must not normalise for the teacher; the teacher does
    it from buffers on raw [0, 1] pixels."""
    t = _teacher(input_size=16, multiple=1)
    x = torch.rand(2, 3, 16, 16)
    pre = t.preprocess(x)
    expected = (x - t.pixel_mean) / t.pixel_std
    assert torch.allclose(pre, expected)


def test_preprocessing_resizes_to_the_rule():
    t = _teacher(input_size=32, multiple=4)
    assert t.preprocess(torch.rand(1, 3, 16, 16)).shape == (1, 3, 32, 32)
    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        t.preprocess(torch.rand(3, 16, 16))


def test_teacher_stays_in_eval_mode_and_frozen():
    class _BN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(x).mean(1)

    t = DepthTeacher(_BN(), output_size=4, input_size=8, multiple=1)
    t.train()
    assert not t.training and not t.backbone.training
    assert all(not p.requires_grad for p in t.parameters())


def test_input_gradient_survives_the_teacher():
    """Frozen via requires_grad_(False), not no_grad: the evaluation wrapper
    keeps the gradient through the teacher so an attack on the depth score
    sees the reference move too."""
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    _teacher()(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_backbone_output_shapes_are_normalised():
    class _Flat(nn.Module):
        def forward(self, x):
            return x.mean(1)  # (B, H, W)

    class _Obj(nn.Module):
        def forward(self, x):
            class Out:
                predicted_depth = x.mean(1)

            return Out()

    class _Bad(nn.Module):
        def forward(self, x):
            return "nope"

    for backbone in (_Flat(), _Obj()):
        t = DepthTeacher(backbone, output_size=4, input_size=8, multiple=1)
        assert t(torch.rand(2, 3, 8, 8)).shape == (2, 1, 4, 4)
    with pytest.raises(ValueError, match="predicted_depth"):
        DepthTeacher(_Bad(), output_size=4, input_size=8, multiple=1)(
            torch.rand(1, 3, 8, 8)
        )


def test_describe_records_the_instrument():
    d = _teacher().describe()
    assert d["frame"] == DEPTH_FRAME
    assert d["output_size"] == 8 and d["input_size"] == 16
    assert d["teacher"] == "fake_luminance_blur"


def test_real_hf_class_from_a_random_config_has_the_expected_shape():
    """No download: a random-weight DepthAnything from a bare config, only
    to pin the (B, H, W) output convention the wrapper relies on."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.DepthAnythingConfig()
    backbone = transformers.DepthAnythingForDepthEstimation(cfg).eval()
    t = DepthTeacher(backbone, output_size=8, input_size=56, multiple=14)
    with torch.no_grad():
        d = t(torch.rand(1, 3, 56, 56))
    assert d.shape == (1, 1, 8, 8)
