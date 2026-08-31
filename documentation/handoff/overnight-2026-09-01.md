---
type: handoff
title: Overnight task brief — backbone grid, 2026-09-01
status: active
as_of: 2026-09-01
source: "written before the run; results to be filled in by the collating agent"
tags: [track-b, overnight, clip]
links: [current-state, 2026-09-01-track-b-full-matrix]
---

# What is running

`jobs/track_b_backbone_grid.sh`, launched 2026-09-01 ~00:50 CEST, detached
(setsid). Three arms, each: train head (frozen backbone) then evaluate
in-domain clean, in-domain `query_underconf`, and So-Fake-OOD clean.

1. `clip_vit_b32_tecoa_probe` — TeCoA4, a *supervised* robustification
   objective against FARE's unsupervised one, same architecture and base.
2. `clip_vit_b16_probe` — standard ViT-B/16. **The control.**
3. `clip_vit_b16_fare_probe` — FARE4 ViT-B/16.

Logs and `.done` markers: `${LOGS_PATH}/track_b_grid/`. Every step is
resumable; re-running the script skips completed steps. A step that succeeded
with a *bad* number is also skipped, so delete its marker to redo it.

# What the grid is for

Two mechanisms were confounded in the frozen-vs-fitted analysis and this
separates them:

|        | standard              | FARE4                      |
|--------|-----------------------|----------------------------|
| B/32   | measured 2026-08-31   | measured 2026-09-01        |
| B/16   | **arm 2**             | **arm 3**                  |

Both B/16 arms share the `laion2B-s34B-b88K` base; both B/32 arms share
`b79K`. Without arm 2, any B/16 improvement reads as a robustness result when
it may be nothing but the finer patch grid removing the 768x3072 rank bound.

Arm 1 asks whether the confidence-axis effect belongs to *adversarial
fine-tuning in general* or to FARE specifically — a far more transportable
claim if the two agree.

# The numbers to compare against

Established 2026-08-31/09-01, `limit_test=1000`, thresholds from in-domain calib:

| arm | acc | Phi clean | Phi under query_underconf | OOD acc | OOD rec_tam |
|---|---:|---:|---:|---:|---:|
| B/32 standard | 0.8811 | 0.8284 | **0.4837** (chance) | 0.4810 | 0.0503 |
| B/32 FARE4 | 0.7873 | 0.7793 | **0.7485** | 0.5116 | 0.2963 |
| ResNet-18 | 0.8249 | 0.8156 | 0.7386 | 0.3516 | 0.2725 |

The FARE result is the headline: an **11x smaller Phi collapse** with the head
untouched, which is P2 from the VLM theory note confirmed. It costs clean
accuracy (-0.094) and buys a 5.9x recovery of OOD tampered recall.

# Read the results this way

1. **Phi clean -> query_underconf** per arm. Does TeCoA reproduce FARE's
   stability? Does B/16-standard show any of it (it should NOT — it has no
   adversarial training; if it does, the effect is not robustness).
2. **recall_tampered**, in-domain and OOD. B/16's square projection removes
   the rank bound; if tampered improves for B/16-standard, patch size is a
   real mechanism independent of robustness.
3. **Mean accuracy is the wrong headline.** Read per-modality rows.

# Known confound, do not repeat

The crop diagnostic run on 2026-09-01 evaluated a **resize-trained** checkpoint
under `input_mode=crop`. That is a covariate shift, not a control — the repo's
own `input_mode` docstring requires fitting and evaluating under the same
mode. Its in-domain accuracy (0.5834 vs 0.8811) mostly measures the protocol
mismatch. A clean test needs a crop-*trained* arm. Do not report the crop
numbers as evidence about the low-pass.

# Boundaries for the collating agent

DO: read logs, collate results into a table, update `documentation/snapshots/`
and this file, commit and push.

DO NOT: launch new training or evaluation runs; modify `src/` or `configs/`;
kill any process (the `sotto` dictation service stays up unless a human says
otherwise); delete anything under `_runs/`; force-push.

If a step failed, record that it failed and why. Do not retry it.
