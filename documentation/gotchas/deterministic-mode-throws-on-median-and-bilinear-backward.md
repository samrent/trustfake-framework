---
type: gotcha
title: Under deterministic mode on CUDA, Tensor.median(dim) and bilinear-interpolate backward throw
status: current
as_of: 2026-09-03
source: "torch 2.5.1 use_deterministic_algorithms throw list, caught by the Track C critic before any GPU run"
tags: [training, determinism, cuda, track-c]
links: [specs/track-c-depth-auxiliary]
---

**Trap.** Training runs with `trainer.deterministic: true`, which calls
`torch.use_deterministic_algorithms(True)` (not warn-only). On CUDA that mode RAISES for
`Tensor.median(dim=...)` (its indices output has no deterministic kernel) and for the backward of
`F.interpolate` in `linear`/`bilinear`/`bicubic`/`trilinear` modes. On CPU neither throws, so
the whole test suite is green and the first GPU training step dies. Both ops are exactly what a
depth loss (per-image median) and a decoder head (bilinear upsampling) reach for first.

**Fix.** `trustfake.losses.depth.normalize_depth` takes the lower median through a sort
(`flat.sort(dim=1).values[:, (n-1)//2]`, bit-equal to `torch.median`'s value and pinned by a
test that also spies on `Tensor.median`); `DepthHead` upsamples with `mode="nearest"` followed by
a 3×3 conv (a test spies on `F.interpolate` and refuses any other mode). Antialiased bilinear
resampling survives only in `resize_depth`, used for teacher outputs under no-grad.

**Still not proven on the box:** `jobs/track_c_depth.sh` runs two training batches and one
depth-score evaluation under the real trainer config as its first step, before any store time is
spent. Evaluation uses `deterministic: warn`, so the online teacher's bicubic resize only warns.
