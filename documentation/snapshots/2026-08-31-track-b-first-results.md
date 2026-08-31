---
type: snapshot
title: Track B first results — CLIP probe vs ResNet, in-domain and cross-dataset
status: current
as_of: 2026-08-31
source: "jobs/track_b_chain.sh on the 3090 box; limit_test=1000, profile=train, batch 32"
tags: [track-b, results]
links: [clip-backbone-fixes-cross-dataset, clip-low-pass-destroys-tampered]
---

## In-domain (SID-Set)

| metric | CLIP probe | e8_standard ResNet |
|---|---:|---:|
| accuracy | **0.8811** | 0.8249 |
| fd_auroc | **0.8284** | 0.8156 |
| aurc | **0.0316** | 0.0511 |
| detection_auroc | **0.9500** | 0.9166 |

CLIP probe per-modality: `detection_auroc_synthetic` 0.9971 / `recall_synthetic` 0.9972;
`detection_auroc_tampered` 0.8962 / `recall_tampered` 0.8418; `recall_real` 0.8043.

Trainable parameters: **1,539** of 87.9 M (the linear head only).

## Cross-dataset (FakeClue, zero-shot, binary fold)

| metric | CLIP probe | e8_standard ResNet |
|---|---:|---:|
| accuracy | 0.5207 | 0.5148 |
| fd_auroc | 0.5054 | 0.4313 |
| detection_auroc | 0.5198 | 0.4880 |

Read against FakeClue's **0.638 majority floor** and **0.690 metadata rule** — both models are
below both. The binary fold merges synthetic and tampered, so no per-modality breakout exists here.

## So-Fake-OOD

Not yet measured. Both legs failed on the calib guard and were re-run after
[[ood-thresholds-come-from-in-domain-calib]] shipped.
