---
type: snapshot
title: Track C first results — depth head, 8/255, one seed, λ=1.0
status: current
as_of: 2026-09-04
source: "jobs/track_c_depth.sh on the 3090 box, 2026-09-03 15:30 → 2026-09-04 09:50 UTC; limit_test=1000; 102 cells collated by jobs/summarise_track_c.py, 0 failed; white-box query cells trimmed to L1 by PI decision"
tags: [track-c, results, depth, robustness, confidence-axis]
links: [specs/track-c-depth-auxiliary, decisions/track-c-white-box-cells-are-in-domain-only, gotchas/ace-never-reads-the-uncertainty-score, 2026-08-31-track-a-arm-inventory, handoff/current-state]
---

# Track C — first results

Matrix {standard, pgd_at} × {baseline, +depth} at λ=1.0, seed 1, 12 epochs, the Track A recipe.
Both legs, six conditions, three scorings; the full per-cell table is `RESULTS_track_c.md` at
the branch root and in `_runs/backups/track_c-20260904-0950.tar.gz`. Every depth-score number
carries the spec's caveat: a gradient adaptive attack on the residual was not run.

**Recipe check.** `track_c_standard`'s validation curve equals Track A's `e8_standard` to the
last digit on all 12 epochs. Selected epochs: standard 4, pgd_at 11, standard_depth 7,
pgd_at_depth 7. Training cost: 807 / 1352 / 822 / 1545 s. The depth head learns: SSI-L1 on
validation 0.507 (standard_depth) and 0.611 (pgd_at_depth) against 1.0 for a constant map.

## Gates

| gate | result |
|---|---|
| G1 store | 30 shards, 25,320 rows, every map finite (569 MB, ~40 img/s to build) |
| G2 degeneracy | \|ρ\| vs 1−MSP on 7,059 calib rows: 0.2534 (standard_depth), 0.2792 (pgd_at_depth) — independent |
| G3 clean floor | **fails for both AT arms**: pgd_at 0.7003, pgd_at_depth 0.6935 on L1 (floor 0.75). Track A's 8/255 arms sit at 0.69–0.77, so this is the cost of 8/255 on this task, not a broken fit; the floor was set without an AT reference. Open for the PI. |
| G4 collapse | min n_operating_points 424 (L2, standard_depth, pgd, msp); all ≥ 32 |
| G5 smoke | passed after the eval-batch fix (spec change log 2026-09-03) |

## L1 — SID-Set held-out slice, accuracy

| arm | clean | pgd | jpeg | rec_tampered (clean) |
|---|---:|---:|---:|---:|
| standard | 0.8249 | 0.0992 | 0.7769 | 0.6994 |
| standard_depth | 0.8317 | 0.1539 | 0.7868 | 0.7532 |
| pgd_at | 0.7003 | 0.6484 | 0.6982 | 0.7690 |
| pgd_at_depth | 0.6935 | 0.6880 | 0.6945 | 0.6709 |

## L1 — Φ (fd_auroc) by scoring, the two depth arms

| arm | cond | msp | depth_tr | depth_wb | combined_tr |
|---|---|---:|---:|---:|---:|
| standard_depth | clean | 0.8393 | 0.5761 | — | 0.7498 |
| standard_depth | pgd | 0.4937 | 0.5361 | — | 0.5367 |
| standard_depth | query_underconf | 0.7123 | 0.5589 | 0.5808 | 0.6571 |
| standard_depth | query_overconf | 0.8083 | 0.5779 | 0.5591 | 0.7094 |
| standard_depth | ace_uint8 | 0.1610 | 0.6070 | = tr | 0.3481 |
| standard_depth | jpeg | 0.7729 | 0.5598 | — | 0.7022 |
| pgd_at_depth | clean | 0.7665 | 0.5805 | — | 0.7255 |
| pgd_at_depth | pgd | 0.7500 | 0.6208 | — | 0.7350 |
| pgd_at_depth | query_underconf | 0.7625 | 0.5750 | 0.5947 | 0.7289 |
| pgd_at_depth | query_overconf | 0.7689 | 0.5779 | 0.5716 | 0.7236 |
| pgd_at_depth | ace_uint8 | 0.7217 | 0.5792 | = tr | 0.6955 |
| pgd_at_depth | jpeg | 0.7678 | 0.5752 | — | 0.7255 |

Baselines' msp Φ for reference: standard clean 0.8156, pgd 0.3630, query_underconf 0.7365,
ace 0.2153; pgd_at clean 0.7422, pgd 0.7363, query_underconf 0.7413, ace 0.7009.

## L2 — So-Fake-OOD, in-domain thresholds

Every arm at chance: accuracy 0.3241–0.3516 on every condition, Φ 0.45–0.57 for msp and
0.44–0.52 for the depth score. Nothing on this leg is readable beyond "reproduces Track B's
cross-dataset collapse". `standard` under ACE shows the anchor pattern here too (Φ 0.0222,
accuracy unchanged).

## H(a)–H(d), by the pre-registered rules

- **H(a) — no clean effect.** standard_depth − standard = **+0.0068** on L1 clean, under the
  +0.01 rule, at the one λ tried. Not pre-registered but visible: tampered recall +0.054, JPEG
  +0.0099, PGD +0.055 from a floor.
- **H(b) — not readable under G3 as written**, because both AT arms are under the 0.75 clean
  floor. The relative number: pgd_at_depth beats pgd_at under `pgd` by **+0.0396** with clean
  within **−0.0068** — the rule's +0.02 / −0.01 would be met. Two cautions before it becomes a
  claim: (1) the bump is measured under the same 7-step PGD the arm trained on, and the depth
  arm's PGD drop is 0.0055 against 0.052 for pgd_at — the spec's separate Square/AutoAttack run
  is the gradient-masking check; (2) one seed.
- **H(c) — fails on both arms.** The depth residual trails max-probability under
  `query_underconf` by 0.15 (standard_depth) and 0.19 (pgd_at_depth); under `pgd` it leads by
  +0.042 on standard_depth only because msp collapsed to 0.49 there, and trails by 0.13 on
  pgd_at_depth. The combined score is never better than its better component (best case a tie,
  0.5367 vs 0.5361) and worse than msp wherever msp is good. Not pre-registered: under ACE on
  standard_depth the residual holds Φ 0.607 where msp inverts to 0.161 — the one cell family
  where depth is the better score; on pgd_at_depth msp holds 0.722 and the residual (0.579)
  trails it. ACE never targets the residual (gotcha).
- **H(d) — the white-box drop is within ±0.02.** query_underconf tr→wb: 0.5589→0.5808 and
  0.5750→0.5947; query_overconf: 0.5779→0.5591 and 0.5779→0.5716. No `_wb` at or below 0.5;
  the watchdog did not fire. The 400-query attack on the residual neither breaks nor helps it.
  There is no `_tr` claim for this number to sit beside.

## What the residual measures, on this evidence

Its Φ on clean inputs is 0.58 on both arms: the residual barely tracks misclassification. It
also barely moves under 8/255 perturbations of either kind (wb ≈ tr; ACE leaves it at 0.61 while
inverting msp). An attack on the classifier does not need to drag the depth head, so the head's
disagreement with the teacher does not flag the attack. The per-image residual (calib mean 0.51 /
0.60, std 0.29 / 0.32) is dominated by how well the head fits each image, not by whether the
classifier is under attack.

## Cost notes that shaped the run

- White-box through the fp32 teacher at 518 px: ~0.85 GB/image; `EVAL_BATCH=8`.
- A white-box query cell: 8,229 / 8,226 s (400 queries × 125 batches of teacher forwards).
  The 16 pre-registered `_wb` query cells were ~37 GPU-hours; four ran (PI decision).
- msp cell ~1 min; depth-scored cell ~5 min (the 7,059-row calib pass through the teacher);
  query `_tr` cell ~9.5 min; ACE `_wb` = `_tr` and should not be scheduled.
- Chain wall time 18.3 h including the trim restart.

## Not readable here

A gradient adaptive attack on the residual; any λ other than 1.0; seeds; Square/AutoAttack on
the AT pair; L2 white-box; anything against Track A/B arms.
