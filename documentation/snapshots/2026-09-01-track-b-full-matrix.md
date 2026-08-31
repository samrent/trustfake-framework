---
type: snapshot
title: Track B full matrix — 2 models x 3 datasets x 4 conditions
status: current
as_of: 2026-09-01
source: "jobs/track_b_matrix.sh on the 3090 box; limit_test=1000, 24/24 cells, 0 failures"
tags: [track-b, results, confidence-axis]
links: [clip-is-more-fragile-on-the-confidence-axis, 2026-08-31-track-b-first-results]
---

24 cells, no failures. Full table in `_runs/logs/track_b_matrix/RESULTS.md` on the box.

## The confidence axis, worst of the two query attacks

| model | dataset | accuracy clean -> attacked | Phi clean -> attacked | via |
|---|---|---|---|---|
| clip_probe | SID-Set | 0.8811 -> **0.8811** | 0.8284 -> **0.4837** | query_underconf |
| clip_probe | So-Fake-OOD | 0.4810 -> 0.4810 | 0.6665 -> 0.4782 | query_underconf |
| resnet | SID-Set | 0.8249 -> 0.8249 | 0.8156 -> 0.7386 | query_underconf |

Accuracy is bit-identical in every row -- argmax preservation held on all six. **The CLIP probe's
Phi falls to chance in-domain while its accuracy does not move at all.**

## Prediction axis (PGD, in-domain)

| model | accuracy clean -> pgd | Phi |
|---|---|---|
| clip_probe | 0.8811 -> 0.1045 | 0.8284 -> 0.6765 |
| resnet | 0.8249 -> 0.0982 | 0.8156 -> 0.3629 |

## What is NOT readable here

- **FakeClue is degenerate.** Both models are at chance, so Phi ranks almost nothing; CLIP's Phi
  *rises* under attack (0.5054 -> 0.5604), which is noise, not robustness. Do not quote FakeClue
  robustness numbers.
- **Two PGD rows are artifacts.** CLIP on So-Fake-OOD shows recall_tampered 0.0503 -> 0.8492 under
  PGD while accuracy falls; the attack is pushing predictions into the tampered class, not
  improving tampered detection. Same shape for the ResNet's recall_synthetic (0.0055 -> 0.5525).
- `query_underconf` is far stronger than `query_overconf` here. A summary that reports only
  overconf understates the effect by roughly an order of magnitude.
