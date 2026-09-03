"""Depth-consistency rejection scores (Track C).

Both scores here consume, beside the class probabilities every other score
takes, the student's depth map and the frozen teacher's map for the same
input. `DepthConsistencyWrapper` is the only caller that has all three; a
plain `BaseWrapper` never calls these, and the config validator refuses the
pairing (`trustfake.instantiator.WRAPPER_UNCERTAINTY_SCORE_TARGETS`).

`DepthConsistencyScore` is the per-image scale-and-shift-invariant L1 between
the two maps -- the training loss evaluated against an online teacher. The
hypothesis it tests: an attack crafted against the classifier drags the
shared backbone, so the head drifts from the teacher, and the residual ranks
attacked or misclassified inputs above clean ones better than 1 - MSP does.
Higher = more uncertain, like every score in this package; finite by
construction (the frame's MAD is floored).

`CombinedDepthScore` mixes 1 - MSP and the residual. The two live on
incomparable scales, so each is mapped through the empirical CDF of its own
values on the IN-DOMAIN calib split before a weighted sum -- fitted once in
`src/test.py`, between temperature scaling and the moderation gate, from the
same calib source those come from, and applied unchanged to a shifted set
(see `ood-thresholds-come-from-in-domain-calib`). Rank-averaging within a
test batch was rejected because it would make a row's score depend on what
else was in its batch; a z-score because the residual is heavy-tailed; a
fitted combiner because it is a second model to validate. The ECDF is
piecewise-linear between reference points so a gradient attack on the
combined score is not blind to it.

An UNFITTED combined score raises at the first call rather than returning
something: a silently unnormalised sum would rank almost exactly like
whichever component has the larger scale, and read as a result.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

from trustfake.losses.depth import ssi_l1_per_image

__all__ = [
    "DepthAwareScore",
    "DepthConsistencyScore",
    "CombinedDepthScore",
    "depth_consistency_residual",
    "ecdf_interp",
]


def depth_consistency_residual(
    depth_pred: Tensor, depth_target: Tensor, eps: float = 1e-6
) -> Tensor:
    """Per-image residual between the head and the teacher, shape (B,)."""
    if depth_pred.shape[0] != depth_target.shape[0]:
        msg = (
            f"batch sizes differ: {depth_pred.shape[0]} predictions vs "
            f"{depth_target.shape[0]} teacher maps"
        )
        raise ValueError(msg)
    return ssi_l1_per_image(depth_pred, depth_target, eps)


def ecdf_interp(reference_sorted: Tensor, values: Tensor) -> Tensor:
    """Empirical CDF of `values` against an ascending reference, with linear
    interpolation between reference points (so it is differentiable almost
    everywhere) and clamped to [0, 1] outside the reference range."""
    ref = reference_sorted.to(values.device, values.dtype)
    n = ref.numel()
    if n == 0:
        msg = "empty reference: fit the score on a non-empty calib split"
        raise ValueError(msg)
    if n == 1:
        return (values >= ref[0]).to(values.dtype)
    # position of each value in the reference, as a fractional index
    idx = torch.searchsorted(ref, values.detach(), right=True).clamp(1, n - 1)
    lo = ref[idx - 1]
    hi = ref[idx]
    frac = ((values - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    return ((idx - 1).to(values.dtype) + frac) / (n - 1)


class DepthAwareScore(Metric):
    """Marker base: `update(probabilities, depth_pred, depth_target)`."""

    def __init__(self, **metric_kwargs):
        super().__init__(**metric_kwargs)
        self._enable_grad = True

    def update(self, probabilities: Tensor, depth_pred: Tensor, depth_target: Tensor):
        raise NotImplementedError


class DepthConsistencyScore(DepthAwareScore):
    """u(x) = SSI-L1(head(x), teacher(x)). `probabilities` is accepted for
    signature symmetry with the other scores and ignored."""

    def __init__(self, eps: float = 1e-6, **metric_kwargs):
        super().__init__(**metric_kwargs)
        self.eps = eps
        self.residual: list[Tensor]
        self.add_state("residual", default=[], dist_reduce_fx="cat")

    def update(self, probabilities: Tensor, depth_pred: Tensor, depth_target: Tensor):
        self.residual.append(
            depth_consistency_residual(depth_pred, depth_target, self.eps)
        )

    def compute(self) -> Tensor:
        if self.residual:
            return torch.cat(self.residual, dim=0)
        return torch.empty(0)


class CombinedDepthScore(DepthAwareScore):
    """u(x) = w * F_msp(1 - max p) + (1 - w) * F_depth(residual), with F the
    empirical CDFs fitted on the calib split via `fit_reference`.

    Args:
        weight: w in [0, 1]. 1.0 reproduces the max-probability ranking,
            0.0 the depth-consistency ranking.
        eps: MAD floor of the residual.
    """

    def __init__(self, weight: float = 0.5, eps: float = 1e-6, **metric_kwargs):
        super().__init__(**metric_kwargs)
        if not 0.0 <= weight <= 1.0:
            msg = f"weight must be in [0, 1], got {weight}"
            raise ValueError(msg)
        self.weight = float(weight)
        self.eps = eps
        self.score: list[Tensor]
        self.add_state("score", default=[], dist_reduce_fx="cat")
        # Plain attributes, not states or buffers: they must never enter a
        # state_dict (the checkpoint is the model's, not the calibration's).
        self._ref_msp: Tensor | None = None
        self._ref_depth: Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self._ref_msp is not None and self._ref_depth is not None

    def fit_reference(self, msp_calib: Tensor, residual_calib: Tensor) -> None:
        """Record the calib-split distributions of both components."""
        msp = msp_calib.detach().flatten().float().cpu()
        res = residual_calib.detach().flatten().float().cpu()
        if msp.numel() == 0 or res.numel() == 0:
            msg = "fit_reference needs non-empty calib scores"
            raise ValueError(msg)
        if not (torch.isfinite(msp).all() and torch.isfinite(res).all()):
            msg = "fit_reference received non-finite calib scores"
            raise ValueError(msg)
        self._ref_msp = msp.sort().values
        self._ref_depth = res.sort().values

    def _require_fitted(self) -> None:
        if not self.fitted:
            msg = (
                "CombinedDepthScore.fit_reference was never called; fit it on the "
                "in-domain calib split (src/test.py does so after temperature "
                "scaling) before scoring anything."
            )
            raise ValueError(msg)

    def components(
        self, probabilities: Tensor, depth_pred: Tensor, depth_target: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(1 - max prob, residual), the two raw components, shape (B,) each."""
        msp = 1.0 - torch.max(probabilities, dim=1).values
        residual = depth_consistency_residual(depth_pred, depth_target, self.eps)
        return msp, residual

    def update(self, probabilities: Tensor, depth_pred: Tensor, depth_target: Tensor):
        self._require_fitted()
        msp, residual = self.components(probabilities, depth_pred, depth_target)
        combined = self.weight * ecdf_interp(self._ref_msp, msp.float()) + (
            1.0 - self.weight
        ) * ecdf_interp(self._ref_depth, residual.float())
        self.score.append(combined)

    def compute(self) -> Tensor:
        if self.score:
            return torch.cat(self.score, dim=0)
        return torch.empty(0)
