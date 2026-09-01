---
type: reference
title: A better backbone is not a more trustworthy one
status: current
as_of: 2026-09-01
source: "measured, 24-cell matrix 2026-09-01"
tags: [track-b, confidence-axis, clip]
links: [2026-09-01-track-b-full-matrix]
---

The CLIP probe is the better model on every clean metric in-domain (accuracy 0.8811 vs 0.8249,
aurc 0.0316 vs 0.0511, detection_auroc 0.9500 vs 0.9166) **and the more fragile one on the
confidence axis**:

| | Phi clean | Phi under query_underconf | drop |
|---|---:|---:|---:|
| clip_probe | 0.8284 | **0.4837** | 0.345 |
| resnet | 0.8156 | 0.7386 | 0.077 |

A **4.5x larger** collapse, on the model with the better representation. The inversion repeats
under shift (CLIP 0.6665 -> 0.4782). Accuracy does not move in any of these cells, so nothing that
monitors accuracy can see it.

**Why this matters for the project's thesis.** The confidence axis is not a property that improves
with representation quality -- it can get *worse*. A team that adopts a foundation backbone on
clean-metric evidence would be shipping a model whose self-knowledge is easier to destroy, and
would have no instrument to notice. That is the argument for the harness, made against a model
chosen for being good rather than for being weak.

**Untested and the obvious next step:** an adversarially-robustified encoder
(`chs20/FARE4-ViT-B-32-laion2B-s34B-b79K`, same architecture and pretraining as the backbone in
use) would test whether encoder-level hardening repairs the confidence axis with no head
retraining -- prediction P2 from the VLM theory note. The head is 1539 parameters, so the swap is
minutes of compute.
