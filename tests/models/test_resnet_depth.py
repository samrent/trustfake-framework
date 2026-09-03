"""The depth path on the ResNet backbone and the wrapper, and what it must
NOT change.

THE test is `forward` bit-identity: the refactor that exposes the layer3 and
layer4 maps must leave the classification forward op-for-op identical, and
the head must be invisible to it -- otherwise every existing checkpoint,
attack and evaluation would be scored on a different model without any
symptom. The second family pins the checkpoint contract: the baseline
state_dict keeps its historical keys, and a depth checkpoint and a plain one
refuse to load into each other under the strict loading src/test.py uses.
No network, no data.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import trustfake.models.torch.resnet as resnet_module
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.torch import BinaryFoldClassifier, resnet18
from trustfake.models.wrapper import BaseWrapper

SIZE = 64
BASELINE_KEYS = 122


def _pair():
    torch.manual_seed(0)
    plain = resnet18(num_classes=3)
    torch.manual_seed(0)
    depth = resnet18(num_classes=3, depth_head=True)
    return plain.eval(), depth.eval()


def test_forward_is_bit_identical_with_and_without_the_head():
    plain, depth = _pair()
    x = torch.rand(4, 3, SIZE, SIZE)
    assert torch.equal(plain(x), depth(x))


def test_head_construction_leaves_backbone_init_untouched():
    """The head is built after the backbone draws its random numbers, so a
    seed gives the same backbone with and without the head."""
    plain, depth = _pair()
    for key, value in plain.state_dict().items():
        assert torch.equal(value, depth.state_dict()[key]), key


def test_baseline_state_dict_keeps_its_historical_keys():
    plain, depth = _pair()
    assert len(plain.state_dict()) == BASELINE_KEYS
    assert not any(k.startswith("depth_head") for k in plain.state_dict())
    assert set(plain.state_dict()) < set(depth.state_dict())
    assert all(
        k.startswith("depth_head.")
        for k in set(depth.state_dict()) - set(plain.state_dict())
    )


def test_forward_with_depth_shares_the_backbone_pass():
    _, depth = _pair()
    x = torch.rand(2, 3, SIZE, SIZE)
    logits, dmap = depth.forward_with_depth(x)
    assert torch.equal(logits, depth(x))
    assert dmap.shape == (2, 1, SIZE // 2, SIZE // 2)


def test_forward_with_depth_without_a_head_is_refused():
    plain, _ = _pair()
    with pytest.raises(ValueError, match="depth_head=True"):
        plain.forward_with_depth(torch.rand(1, 3, SIZE, SIZE))


def test_input_gradients_flow_through_both_heads():
    _, depth = _pair()
    x = torch.rand(2, 3, SIZE, SIZE, requires_grad=True)
    logits, dmap = depth.forward_with_depth(x)
    (logits.sum() + dmap.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_strict_loading_refuses_a_checkpoint_from_the_other_arm():
    """src/test.py loads strictly: a depth checkpoint into a plain module
    and a plain checkpoint into a depth module must both fail loudly, never
    silently drop or randomly initialise the head."""
    plain, depth = _pair()
    with pytest.raises(RuntimeError, match="depth_head"):
        plain.load_state_dict(depth.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="depth_head"):
        depth.load_state_dict(plain.state_dict(), strict=True)


def _fake_imagenet_state_dict(num_classes=1000):
    torch.manual_seed(7)
    return resnet18(num_classes=num_classes).state_dict()


def test_pretrained_loading_tolerates_only_fc_and_depth_head(monkeypatch):
    """The ImageNet checkpoint has no depth head; those keys (and fc when the
    class count differs) may be missing, nothing else may be."""
    monkeypatch.setattr(
        resnet_module.model_zoo, "load_url", lambda url: _fake_imagenet_state_dict()
    )
    reference = _fake_imagenet_state_dict()

    model = resnet18(pretrained=True, num_classes=3, depth_head=True)
    assert torch.equal(model.conv1.weight, reference["conv1.weight"])
    # fc was re-initialised (3 classes), the head is fresh: both present.
    assert model.fc.weight.shape[0] == 3
    assert model.depth_head is not None

    same_classes = resnet18(pretrained=True, num_classes=1000, depth_head=True)
    assert torch.equal(same_classes.fc.weight, reference["fc.weight"])


def test_pretrained_loading_refuses_a_mismatched_checkpoint(monkeypatch):
    """Before, a non-matching checkpoint under strict=False loaded nothing and
    the model trained from random weights while claiming to be pretrained."""
    bad = _fake_imagenet_state_dict()
    del bad["layer4.1.conv2.weight"]
    monkeypatch.setattr(resnet_module.model_zoo, "load_url", lambda url: bad)
    with pytest.raises(RuntimeError, match="missing"):
        resnet18(pretrained=True, num_classes=3)

    extra = _fake_imagenet_state_dict()
    extra["not.a.key"] = torch.zeros(1)
    monkeypatch.setattr(resnet_module.model_zoo, "load_url", lambda url: extra)
    with pytest.raises(RuntimeError, match="unexpected"):
        resnet18(pretrained=True, num_classes=3)


# --------------------------------------------------------------------------
# The wrapper's separate path
# --------------------------------------------------------------------------


class _Norm(nn.Module):
    def forward(self, x):
        return (x - 0.5) / 0.25


def _wrap(model, norm=None):
    return BaseWrapper(
        normalization_layer=norm if norm is not None else nn.Identity(),
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    ).eval()


def test_wrapper_depth_path_normalizes_like_forward():
    _, depth = _pair()
    wrapper = _wrap(depth, _Norm())
    x = torch.rand(2, 3, SIZE, SIZE)
    logits, dmap = wrapper.forward_with_depth(x)
    ref_logits, ref_dmap = depth.forward_with_depth(_Norm()(x))
    assert torch.equal(logits, ref_logits) and torch.equal(dmap, ref_dmap)
    assert torch.equal(logits, wrapper(x)[0])


def test_wrapper_forward_is_untouched_by_the_head():
    plain, depth = _pair()
    x = torch.rand(2, 3, SIZE, SIZE)
    a = _wrap(plain)(x)
    b = _wrap(depth)(x)
    assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))


def test_wrapper_refuses_models_without_a_depth_path():
    plain, depth = _pair()
    assert not _wrap(plain).has_depth_head
    with pytest.raises(ValueError, match="forward_with_depth"):
        _wrap(plain).forward_with_depth(torch.rand(1, 3, SIZE, SIZE))
    # The binary fold hides the head: the eval-side guard has to know that.
    folded = _wrap(BinaryFoldClassifier(depth, real_class=0))
    assert not folded.has_depth_head
    with pytest.raises(ValueError, match="binary fold"):
        folded.forward_with_depth(torch.rand(1, 3, SIZE, SIZE))
