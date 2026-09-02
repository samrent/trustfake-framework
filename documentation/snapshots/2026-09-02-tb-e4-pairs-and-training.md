---
type: snapshot
title: TB-E4 — pair admission (H5) and the first full fine-tune (H6)
status: current
as_of: 2026-09-02
source: "overnight run on the 3090 box, jobs/tb_e4/; markers + logs under _runs/logs/tb_e4/"
tags: [track-b, curation, training, results, tb-e4]
links: [specs/tb-e4-pairs-and-training, 2026-09-01-tb-e3-curation-ladder]
---

# TB-E4 — the numbers

Pairs staged at QF-85 with ORIGINALS RESIZED TO THEIR EDIT'S DIMENSIONS (the first pass
proved resolution was the residual tell: AUDITS edits ship at 256x256, G1 stayed 0.999
until the resize; amendment registered pre-training). Assembled C2' arm: 179,624 rows —
AUDITS' 150k pairs admitted at G1 0.499 (from 0.999); sagi_d (0.561) and tgif (0.786)
excluded by the gate as registered. G2 dropped 43,712 rows vs the frozen legs.

## detection_auroc on the frozen legs

| model | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| TB-E3 winner (C3a probe) | 0.7523 | 0.7458 | 0.9732 | 0.5434 |
| Arm A (probe, C2' arm; 3 seeds) | 0.7492 | 0.7068 | 0.8319 | 0.6055 |
| **Arm B (full fine-tune, C2' arm; 1 seed)** | **0.7593** | **0.7858** | **0.9766** | **0.7315** |

Arm B per-modality: L2 tampered 0.7775 (project best by ~0.10), L1 synthetic 0.9759,
L1 tampered 0.5442 (in-domain tampered still weak — the resize-224 hypothesis stands).
Selection was on the dev legs only (best epoch 4 of 5, mean dev 0.7290); the frozen legs
were read once per arm.

## Verdicts (pre-registered)

- **H5 — not adopted.** Pair admission wins L4 (+0.0621, 2SD 0.0041) and holds L1, but
  costs L2 −0.039 and L3 −0.141 at probe level. Mechanism (recorded, untested): the class
  mass flips tampered-heavy and 75k 256px-resized news originals redefine the real-class
  direction. Budget-balancing across the axes is the registered follow-up.
- **H6 — ADOPTED: full training on curated data.** Arm B beats Arm A beyond Arm A's seed
  noise on all three shifted legs (L2 +0.079, L3 +0.145, L4 +0.126) and gains L1. The
  frozen-probe regime was masking the composition's value: the fine-tune RECOVERS the L3
  damage the pair-heavy composition causes at probe level AND keeps the L4 gain. Single
  seed; caveat recorded.
- **8/255 battery on Arm B** (SID prefix-1000, T=1): clean 0.660 / fd_auroc 0.708.
  PGD-40 white-box: 0.154 (demolished, as every undefended encoder). query_underconf
  400q: argmax 100% preserved but **fd_auroc 0.517 — chance**. The frozen-probe winner
  had HELD at 0.723 under the same attack: **fine-tuning sells the black-box
  confidence-axis stability that the frozen encoder provided**. The clean-generalisation
  gains and the confidence-axis regression are one trade, and hardening (or freezing
  late layers) is the registered follow-up before any deployment claim.

## Standing state

Best clean-shift detector in the project: Arm B (`_runs/out/tb_e4/armB/best.pt`,
epoch-4 CLIPProbeClassifier, unfrozen). Best confidence-robust detector: the TB-E3 C3a
frozen probe. Nothing dominates both axes yet — that sentence is the next experiment.

## Arm C readout (2026-09-02 midday): not adopted — collapse at the official budget

PGD-3 AT at 8/255 from Arm B's best: clean detection L1 0.524 / L2 0.504 / L3 0.593 /
L4 0.532; top-1 ~0.35 (constant-predictor regime). Adoption rule fails 3/3. The mechanism is
the one the repo already documented for forensic AT at 8/255: the perturbation budget exceeds
the evidence, and training collapses onto a constant output. The eps-warmup knob exists in the
Track A pipes for exactly this reason and was not in the registered Arm C recipe — recipe
error, not a fundamental negative. Registered follow-ups: warmup+TRADES (Arm C'), or the
DeltaCLIP adversarially pre-trained backbone as a column. The axis trade stands: Arm B for
scores, the frozen probe for abstention.
