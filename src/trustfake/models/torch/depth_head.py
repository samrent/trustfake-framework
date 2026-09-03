"""The auxiliary depth decoder for Track C.

A light FPN-style head: the layer3 and layer4 feature maps of a ResNet are
projected to a common width, merged at layer3's grid, and upsampled three
times with a conv-BN-ReLU block each. At a 224x224 input that is 14 -> 28 ->
56 -> 112, so the head's natural output grid is half the input -- the
resolution the teacher targets are precomputed at.

Two things about it are deliberate. It contains NO dropout: `MCDropoutWrapper`
flips every dropout module under the wrapper's model into train mode at
evaluation time, and a stochastic depth head would silently turn the
depth-consistency score into a noise source. And it has no output activation:
the teacher's map is relative inverse depth and the loss is scale-and-shift
invariant, so the head's raw output is the right target space; squashing it
would only fight the alignment.

Upsampling is NEAREST followed by a 3x3 conv, not bilinear: training runs
under `deterministic: true`, and on CUDA that mode throws on the backward of
bilinear (and bicubic, linear) interpolation. The CPU suite would never see
it; the first GPU step would. The conv after each nearest step smooths the
blocks, which is the standard FPN recipe anyway.

It is reachable only through `ResNet.forward_with_depth`. `ResNet.forward`
never calls it, which is what makes the head free at inference and invisible
to every attack in the repo.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

__all__ = ["DepthHead"]


def _conv_bn_relu(width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


class DepthHead(nn.Module):
    """FPN-lite decoder from (layer3, layer4) feature maps to one depth map.

    Args:
        in_channels_l3: Channels of the layer3 map (256 for BasicBlock
            ResNets, 1024 for Bottleneck ones).
        in_channels_l4: Channels of the layer4 map (512 / 2048).
        width: Internal channel width.
        num_upsamples: Number of 2x upsampling blocks after the merge. With
            layer3 at stride 16, three of them put the output at stride 2.
    """

    def __init__(
        self,
        in_channels_l3: int = 256,
        in_channels_l4: int = 512,
        width: int = 128,
        num_upsamples: int = 3,
    ):
        super().__init__()
        if width <= 0:
            msg = f"width must be positive, got {width}"
            raise ValueError(msg)
        if num_upsamples < 0:
            msg = f"num_upsamples must be non-negative, got {num_upsamples}"
            raise ValueError(msg)
        self.lateral3 = nn.Conv2d(in_channels_l3, width, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels_l4, width, kernel_size=1)
        self.smooth = _conv_bn_relu(width)
        self.up = nn.ModuleList([_conv_bn_relu(width) for _ in range(num_upsamples)])
        self.out = nn.Conv2d(width, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f3: Tensor, f4: Tensor) -> Tensor:
        """(B, C3, h, w), (B, C4, h/2, w/2) -> (B, 1, h * 2**num_upsamples, ...)."""
        x = F.interpolate(self.lateral4(f4), size=f3.shape[-2:], mode="nearest")
        x = self.smooth(x + self.lateral3(f3))
        for block in self.up:
            x = block(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return self.out(x)
