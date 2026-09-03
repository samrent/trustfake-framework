"""Scale-and-shift-invariant depth loss, and the one depth frame it lives in.

Monocular depth from a frozen teacher (Depth Anything V2) is RELATIVE: the
teacher emits inverse depth up to an unknown per-image scale and shift, so
nothing in Track C is a depth benchmark. The auxiliary head is a regulariser,
and a regulariser needs a loss that ignores the two degrees of freedom the
teacher never committed to. MiDaS (Ranftl et al., TPAMI 2020) does that by
aligning every map to a zero-median / unit-mean-absolute-deviation frame
before comparing:

    d_hat = (d - median(d)) / mean|d - median(d)|

Applied to BOTH operands, the L1 between them is invariant to an affine
change of either one -- a prediction equal to the target times 3 plus 7
scores exactly zero -- and a CONSTANT prediction scores exactly 1.0 (the
target's own mean absolute deviation, in its own frame). That gives the
number a fixed floor which is the same for every image, which is what lets
it double as a rejection score later.

The same frame serves three places, and they have to agree or the residual
means nothing: the precomputed teacher targets on disk are stored in it, the
training loss re-applies it (the operation is idempotent up to float16
rounding), and the depth-consistency score at evaluation applies it to the
student's head and the online teacher alike. Every one of those calls
`normalize_depth`; none re-implements it.

Precision: the statistics are computed in float32 with autocast disabled.
Training runs bf16-mixed, and a median over a 12,544-pixel map taken at 8
mantissa bits is not the median (the evidential head opts out of autocast for
the same reason, see `trustfake.models.wrapper.evidential`).

Determinism: training runs under `deterministic: true`, and on CUDA that
mode THROWS for `Tensor.median(dim=...)` (it returns indices too) and for
the backward of linear/bilinear/bicubic interpolation. The CPU test suite
cannot see either. So the median here is taken through a sort, which is not
on the throw list and yields the identical lower median, and `resize_depth`
is documented as no-grad-only (precompute and the online teacher).

Shapes are strict on purpose. The loss refuses a prediction and a target of
different spatial size rather than resampling one of them: the precompute
script and the depth teacher both emit maps at the head's own grid, and a
silent interpolation here would let a resolution mismatch between the two
train quietly against a blurred target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

__all__ = [
    "normalize_depth",
    "resize_depth",
    "ssi_l1_per_image",
    "ScaleShiftInvariantL1",
]


def _as_maps(depth: Tensor, name: str) -> Tensor:
    """Accept (B, 1, H, W) or (B, H, W); return (B, 1, H, W) float32."""
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[1] != 1:
        msg = (
            f"{name} must be a batch of single-channel depth maps, "
            f"(B, 1, H, W) or (B, H, W); got shape {tuple(depth.shape)}"
        )
        raise ValueError(msg)
    return depth.float()


def _lower_median(flat: Tensor) -> Tensor:
    """Per-row lower median of a (B, N) tensor, shape (B, 1), via sort.

    Exactly what `torch.median(dim=1)` returns for the values, without its
    indices output -- the form that raises under
    `torch.use_deterministic_algorithms(True)` on CUDA.
    """
    n = flat.shape[1]
    k = (n - 1) // 2
    return flat.sort(dim=1).values[:, k : k + 1]


def normalize_depth(depth: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-image zero-median / unit-MAD frame, in float32.

    Args:
        depth: (B, 1, H, W) or (B, H, W). Any dtype; returned as float32 in
            the input's shape.
        eps: Floor on the mean absolute deviation, so a constant map maps
            to zeros instead of NaN.

    The median is the LOWER median (the smaller of the two middle values for
    an even pixel count), as `torch.median` defines it, computed through a
    sort (see the module docstring for why not `median` or `quantile`).
    """
    maps = _as_maps(depth, "depth")
    with torch.autocast(device_type=maps.device.type, enabled=False):
        flat = maps.flatten(1)
        median = _lower_median(flat)
        centred = flat - median
        scale = centred.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        return (centred / scale).reshape(depth.shape)


def resize_depth(depth: Tensor, size: int | tuple[int, int]) -> Tensor:
    """Resample (B, 1, H, W) depth maps to `size` with antialiased bilinear
    interpolation; identity when the size already matches.

    One implementation, used by the precompute script and the online
    teacher alike, so the training targets and the evaluation reference go
    through the same resampling. Antialiased bilinear resampling has no
    deterministic CUDA backward, so this is for teacher outputs only --
    never for the head's map inside a training loss.
    """
    maps = _as_maps(depth, "depth")
    if isinstance(size, int):
        size = (size, size)
    if tuple(maps.shape[-2:]) == tuple(size):
        return maps
    return F.interpolate(
        maps, size=size, mode="bilinear", align_corners=False, antialias=True
    )


def ssi_l1_per_image(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-image scale-and-shift-invariant L1, shape (B,), float32.

    Both operands are put in the median/MAD frame first, so the value is
    invariant to an affine change of either. Differentiable w.r.t. both
    arguments (an adaptive attack will want the gradient through `target`
    when the target is an online teacher).

    Raises:
        ValueError: on a shape mismatch. The loss does not resample.
    """
    p = _as_maps(pred, "pred")
    t = _as_maps(target, "target")
    if p.shape != t.shape:
        msg = (
            "pred and target must have identical shapes; got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}. Emit the head "
            "and the teacher at the same grid instead of resampling here."
        )
        raise ValueError(msg)
    with torch.autocast(device_type=p.device.type, enabled=False):
        diff = normalize_depth(p, eps) - normalize_depth(t, eps)
        return diff.abs().flatten(1).mean(dim=1)


class ScaleShiftInvariantL1(nn.Module):
    """`ssi_l1_per_image` reduced over the batch.

    Args:
        eps: MAD floor, see `normalize_depth`.
        reduction: "mean" (default) or "none" for the per-image vector.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "none"):
            msg = f"reduction must be 'mean' or 'none', got {reduction!r}"
            raise ValueError(msg)
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        per_image = ssi_l1_per_image(pred, target, self.eps)
        if self.reduction == "none":
            return per_image
        return per_image.mean()

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction={self.reduction!r}"
