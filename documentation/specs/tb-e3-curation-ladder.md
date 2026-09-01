---
type: spec
title: TB-E3 — the curation ladder (multi-dataset hybrid, data-selection ablation)
status: active
as_of: 2026-09-01
source: "registered 2026-09-01; PI answers of the same day folded in (budget 8/255, FakeClue exclusion is policy)"
tags: [track-b, experiments, curation, generalisation, spec]
links: [handoff/current-state, decisions/ood-thresholds-come-from-in-domain-calib, decisions/track-a-and-track-b-are-separate-tables, rejected/clip-backbone-fixes-cross-dataset, gotchas/clip-normalization-lives-in-the-model, gotchas/gpu-concurrency-is-negative-on-this-box, gotchas/limit-test-prefix-needs-a-shuffled-order, reference/fakeclue-dataset-properties]
---

# TB-E3 — the curation ladder

**For the executing agent.** This spec is self-contained given this repo plus this `documentation/`
brain. Read `handoff/current-state.md` and the linked leaves before starting. Execution happens on
the 3090 box from `~/Desktop/FILES/PROJECTS/trustfake/framework`. Human-readable twin of this spec
(fill its tables at the end): https://claude.ai/code/artifact/385a09af-4ed3-4ad2-bce6-088b42a845e0

## Objective

Decide whether **data curation** (versus naive pooling) improves generalisation of the
frozen-probe detector, and **which curation decision carries the effect** — before any expensive
multi-dataset training run is designed around unexamined data. Track B context: the backbone is
not the generalisation lever (see `rejected/clip-backbone-fixes-cross-dataset.md`); multi-dataset
training is the remaining lever; this experiment makes its data composition an ablated,
attributable choice instead of a belief.

## Policy constants (PI-confirmed 2026-09-01)

- **Attack budget: ℓ∞ ε = 8/255** is the project's official threat model. TB-E3's core is not an
  attack experiment, but any adversarial evaluation run under it uses 8/255. Note when reporting:
  published hardened CLIP weights (FARE4/TeCoA4) are 4/255 products — an 8/255 attack on them is
  a beyond-training-budget evaluation and must be labeled as such.
- **FakeClue is excluded** — training AND evaluation. Standing policy, not a per-experiment choice
  (metadata floor 0.690, test/train identity leakage; see `reference/fakeclue-dataset-properties.md`).
- **Thresholds come from in-domain calibration only** (`decisions/ood-thresholds-come-from-in-domain-calib.md`).
- **Track A and Track B never share a table** (`decisions/track-a-and-track-b-are-separate-tables.md`).

## Open inputs (recorded, not blocking)

- EV-AT arm training ε values are **unknown** ("e8_standard" suggests ε=8/255 but is unconfirmed).
  Needed only for cross-family reporting, not for TB-E3. Do not guess; ask or read the Track A
  configs if present in `_runs/out/`.
- Dataset availability/licensing for the §Datasets shortlist must be confirmed at step R1 and the
  confirmed list recorded in this spec's change log (update in place, per conventions).

## Instrument — held fixed across every arm

- Encoder: **ViT-B/16, standard weights, frozen** (open_clip). Embed every image **once**; cache
  512-d features + a manifest (dataset, version, preprocessing hash). All arms fit on cached
  features. CLIP normalization lives in the model, never the datamodule
  (`gotchas/clip-normalization-lives-in-the-model.md`).
- Head: linear, 1,539 parameters, same recipe as the TB-E2 backbone grid.
- Arms are **index lists** (image ids + selection seed), never file copies. Reproducible from
  (manifest hash, seed).
- Jobs sequential (`gotchas/gpu-concurrency-is-negative-on-this-box.md`); chain-style `.done`
  markers; back up `_runs/` when the ladder completes.
- Capped/test splits must be shuffled before any prefix cap
  (`gotchas/limit-test-prefix-needs-a-shuffled-order.md`).

## Evaluation legs — freeze BEFORE curation starts, write a frozen-legs manifest

| leg | set | measures |
|---|---|---|
| L1 | SID-Set test | in-domain baseline |
| L2 | So-Fake-OOD | natural distribution shift — untouched; no sibling (So-Fake-Set) data anywhere in training |
| L3 | ≥2 held-out generators (entire generators reserved at ingestion) | unseen generation mechanisms |
| L4 | ≥1 held-out manipulation tool/framework (reserved at ingestion) | unseen editing mechanisms |

Prefer L3 generators *newer* than any training generator (temporal realism). L3/L4 membership is
chosen at ingestion, recorded in the frozen-legs manifest, and never revisited.

## Datasets — candidate pool (confirm availability at R1)

- **Synthetic environments:** GenImage (primary; generator labels = environment labels);
  Community Forensics; WildFake; ArtiFact; Synthbuster.
- **Tampered environments (scarce, thesis-critical):** MAGIC; TGIF; SAGI-D; DEFACTO; IMD2020;
  NIST MFC (needs registration); CASIA v2.
- **Real reinforcement:** reals shipped with the above; RAISE; Dresden; VISION. Span camera-native
  AND web-recompressed reals, or "real = clean pipeline" becomes the next shortcut.
- **Eval-only:** So-Fake-OOD; optionally Chameleon as a brutal second synthetic eval.
- **Excluded:** FakeClue (policy); So-Fake-Set (sibling of L2); FF++-derived content unless
  identity-quarantined.

## Validity gates — an arm that fails a gate is fixed, not trained

- **G1 metadata baseline:** a headers-only classifier (aspect ratio, resolution, JPEG quality; no
  pixels) must score detection_auroc **< 0.55 per environment** on the arm's training set.
- **G2 leakage:** zero identity overlap and zero near-duplicates (cosine threshold in embedding
  space; record the threshold) between any training arm and any of L1–L4. Cross-dataset dedup is
  mandatory (COCO-derived sets overlap).
- **G3 cell minimums:** every (class × environment) cell meets a recorded minimum count; per-class
  metrics on undersized cells are reported with n and flagged, never headlined.
- **G4 label convention:** 20-image manual spot-check of the unified label map per ingested
  dataset before its images enter any arm.

## Arms

| arm | adds | isolates |
|---|---|---|
| C0 | naive pool, as shipped | baseline |
| C1 | hygiene: unified labels, dedup, identity/content firewall vs L1–L4 | cost of dirt |
| C2 | matched nuisance marginals per class within each environment | value of killing shortcuts |
| C3a | strict mapping: unmappable binary fakes dropped | mapping policy (vs C3b) |
| C3b | multi-task mapping: unmappables train a binary objective only | mapping policy (vs C3a) |
| C4 | equalized budgets per (class × environment) cell | dominance effects |
| C5a | random subsample at budget B | selection method (vs C5b) |
| C5b | k-center coverage coreset at the same budget B | selection method (vs C5a) |

- 3 selection seeds per arm. Probe is convex; variance is sampling variance — that is what the
  error bars must cover.
- **Size confound:** run every comparison twice — natural size, and all arms subsampled to the
  smallest arm's N.
- Fit count: 8 arms × 2 sizes × 3 seeds = **48 probe fits** on cached features + 4-leg evaluation
  each. Order of a day, sequential.

## Metrics per arm

detection_auroc overall AND per-class (tampered especially) on L1–L4; fd_auroc (watchdog);
aurc; per-environment risk variance (invariance diagnostic); G1 metadata-baseline score;
N per (class × environment) cell.

## Decision rules — written before the first fit; do not relitigate after seeing numbers

- **H1 (does curation matter):** C2 beats C1 on L2–L4 by more than 2× pooled seed SD. If C0 ≈ C5
  on all shifted legs, curation is not the lever: record the null in `rejected/` and stop.
- **H2 (mapping policy):** the C3 variant that wins on **tampered detection_auroc under shift**
  becomes the project's mapping decision (write it to `decisions/`). Ties → strict-drop (C3a).
- **H3 (coreset):** C5b must beat C5a beyond seed noise on ≥2 shifted legs, else coreset
  machinery is dropped permanently.
- **H4 (phase 2, winning curation only):** {pooled ERM, GroupDRO, V-REx}; an invariance objective
  is adopted only if it beats pooled ERM on L3+L4 without losing > 0.01 on L1. Curation table and
  objective table are separate tables.
- **Watchdog:** any curation stage that degrades fd_auroc by > 0.05 is flagged for follow-up
  before adoption; no decision rule attached.
- Borderline calls (within 2× SD): add 5 seeds before a verdict, do not soften the rule.

## Runbook

1. Confirm dataset availability + licenses; record versions in the change log below.
2. Ingest to unified schema: label map (G4), environment tag, nuisance metadata (aspect,
   resolution, JPEG quality).
3. Embed everything once with frozen B/16; cache features + manifest hashes.
4. Choose and freeze L3/L4; write the frozen-legs manifest.
5. Cross-dedup all training candidates against L1–L4 (G2).
6. Generate arms C0–C5b as index lists, 3 seeds each; run G1–G3 per arm.
7. Fit all probes sequentially; evaluate L1–L4; collate one results table on the box.
8. Apply H1–H3; record verdicts.
9. Phase 2 on the winning curation; apply H4.
10. Back up `_runs/`; write `snapshots/<date>-tb-e3-curation-ladder.md` (numbers) and any new
    `decisions/` leaves (choices + why); update `handoff/current-state.md`; fill the tables in the
    artifact twin (republish with `url` = the link at the top of this spec); update this spec's
    status and change log.

## Definition of done

H1–H3 have verdicts backed by the collated table; the mapping decision exists as a `decisions/`
leaf; snapshots leaf written; handoff updated; `_runs/` backed up; artifact twin filled. If any
gate could not be satisfied for a dataset, that dataset's exclusion is recorded with the reason.

## Change log

- 2026-09-01 — registered. PI answers folded in: budget 8/255; FakeClue exclusion is policy;
  EV-AT ladder spec unknown (open input). Datasets pending availability confirmation (R1).
