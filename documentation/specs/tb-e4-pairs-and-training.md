---
type: spec
title: TB-E4 — pair-matched tampered admission + the first full training run
status: active
as_of: 2026-09-01
source: "registered 2026-09-01 evening, PI sign-off same day; executes overnight on the 3090 box"
tags: [track-b, experiments, curation, training, tb-e4, spec]
links: [specs/tb-e3-curation-ladder, 2026-09-01-tb-e3-curation-ladder, decisions/tb-e3-mapping-is-strict-drop, decisions/ood-thresholds-come-from-in-domain-calib]
---

# TB-E4 — pairs, then pixels

**Question.** TB-E3 proved matching buys generation-shift generalisation and starves the
tampered axis (L4 −0.114; the winner carries 3.2k tampered rows). TB-E4 tests the directed
fix — admit original/edit PAIRS at scale with the compression tell removed — and then trains
the first full (unfrozen) detector on the winning composition.

## Inherited, unchanged

Frozen legs L1–L4 and calib (tb_e3/legs/, never revisited); all TB-E3 exclusions (L3
generators by pattern everywhere, PowerPaint/ps-sp tools, So-Fake-Set, FakeClue); thresholds
from SID calib only; 8/255 for any adversarial evaluation; G2 content decontamination at
max-cosine 0.95 against the legs; every step marker-gated and resumable.

## C2′ — pair-matched tampered admission

- **Paired environments**: AUDITS train+val (manipulated + the per-method original/ files
  on disk, skipped in TB-E3 as unshipped rows — here they are the point), SAGI-D train+val
  (img_path + src_path), TGIF training tools, IMD2020 real_life (the 414 pairs that exist).
- **The tell is removed by construction**: BOTH members of every pair are decoded and
  re-encoded to uniform JPEG quality 85 before embedding and training. Within a pair the
  photo, resolution and now the compression profile are identical; only the edit differs.
  (TB-E3's own dataset section anticipated re-encoding — CASIA's admission rule.)
- Pairs whose original is missing are dropped. PowerPaint-containing rows stay excluded
  (frozen L4). Re-encoded rows get their own feature cache (tb_e4/features_qf85/) with the
  amended preprocessing policy recorded in its manifest.
- **Composition** = TB-E3's C3a synthetic side (CF matched pairs + Synthbuster + deduped
  VISION reals, taken verbatim from the C3a arm parquets) + the C2′ tampered pairs, minus
  the dev-shift holdouts below.
- **Gates re-run on the assembled arm**: G1 per environment must clear < 0.55 — including
  SAGI-D this time; an environment that still fails after the QF-85 re-encode is EXCLUDED
  and recorded (fixed-not-trained realized). G2 re-run against the legs in embedding space
  (standard-B/16 space on the qf85 embeddings). G3 minimums as before.

## Dev-shift holdouts — selection only, frozen at registration

The frozen legs are read ONCE per arm, at final evaluation. All model selection and any
iteration happens on internal dev legs, held out of training here and now:

- **dev-L3**: every CF-Small GigaGAN row (largest single generator, 17.6k) + a seeded 2k
  reserve of CF reals (seed 20260902, disjoint from the L3 real reserve).
- **dev-L4**: TGIF flux1filldev (-sp and -fr, all splits) + TGIF orig validation rows
  as negatives.

## Arms and recipes (pre-registered)

- **Arm A — probe control**: TB-E2 recipe on cached features of the C2′ composition,
  3 seeds. Purpose: the curation-level read at probe cost.
- **Arm B — full fine-tune**: ViT-B/16 unfrozen + linear head, pixel level, single seed
  (seed 1; the single-seed caveat is reported, with Arm A's seed SD as the yardstick).
  AdamW, backbone lr 1e-5 / head lr 1e-3, weight decay 0.05, cosine to zero, batch 64 with
  AMP, max 5 epochs, horizontal flip as the only augmentation. **Pixel protocol parity**:
  Arm B consumes the exact 224×224 pixel stream the feature cache was computed from (packed
  uint8, bit-identical resize), so A-vs-B isolates "unfreezing" and nothing else. Selection:
  best epoch by mean detection_auroc over dev-L3 + dev-L4. Resumable from last checkpoint.
- **Arm C — adversarial fine-tune (stretch, conditional)**: only if Arm B completes before
  06:00 local: continue from B's best checkpoint, PGD-3 at 8/255, 2 epochs, same selection.
  Recorded as run-or-not in the change log; absence is a schedule fact, not a result.

## Decision rules — written before any fit

- **H5 (pair admission)**: Arm A beats TB-E3's C3a on L4 by > 2× pooled seed SD without
  losing > 0.01 detection_auroc on L2 or L3 → pair-matching becomes the project's tampered
  curation policy (decisions leaf). If it wins L4 but pays more than that on L2/L3, record
  the tension, adopt nothing.
- **H6 (does unfreezing pay)**: Arm B beats Arm A on ≥ 2 of the 3 shifted frozen legs by
  more than Arm A's 2× seed SD, without losing > 0.05 on L1 → full training on curated data
  becomes the default going forward; else the frozen-probe regime stands.
- **Battery**: PGD-40 and query_underconf at 8/255 on Arm B's final model (and Arm C's if
  it exists), TB-E2 protocol, labeled.
- Borderline (within 2× SD): add seeds to Arm A / rerun B before a verdict; do not soften.

## Runbook

1. Register this spec (this file) — done at write time.
2. Build the qf85 pair cache (extract pairs, re-encode both members, embed frozen B/16).
3. Assemble C2′; freeze dev legs; run G1/G2/G3; exclusions recorded if G1 fails.
4. Arm A: 3 probe fits + frozen-leg eval + H5 verdict.
5. Pack the 224px pixel stream; Arm B fine-tune with dev-leg selection.
6. Arm B final eval on frozen legs + battery; Arm C if the clock allows.
7. Snapshot + decisions + handoff + artifact twin + _runs backup; verdicts H5/H6.

## Change log

- 2026-09-01 — registered, PI sign-off relayed by the user the same evening.
- 2026-09-01 (pre-training amendment, measured) — the QF-85 re-encode alone did NOT realize
  the stated construction: AUDITS edits ship at 256×256 against native-size originals
  (G1 stayed 0.999 post-re-encode; sagi_d 0.815, tgif 0.766 — resolution is the residual
  tell). Staging now resizes each ORIGINAL to its edit's exact dimensions before the QF-85
  save, making pair members resolution-identical as the spec claims. Caches wiped and the
  chain re-run from staging; the first (invalid) Arm A pass is discarded — its G1 exclusions
  were the machinery working as registered.
