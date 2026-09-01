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
  a beyond-training-budget evaluation and must be labeled as such. No off-the-shelf robust CLIP
  exists above ε=4/255 at any size (see `gotchas/robust-clip-checkpoint-availability.md`);
  matching the 8/255 budget on the defense side would mean fine-tuning an encoder ourselves.
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
- **One preprocessing policy, recorded in the manifest:** how RAW (RAISE NEF) is developed, how
  sub-224 images (ArtiFact 200 px) travel to input size, and any re-encoding (CASIA uniform JPEG).
  The route to 224 is itself a per-dataset signature that headers-only G1 cannot fully see —
  matched marginals in C2 must include source resolution and the resample direction.

## Evaluation legs — freeze BEFORE curation starts, write a frozen-legs manifest

| leg | set | measures |
|---|---|---|
| L1 | SID-Set test | in-domain baseline |
| L2 | So-Fake-OOD | natural distribution shift — untouched; no sibling (So-Fake-Set) data anywhere in training |
| L3 | ≥2 held-out generators (entire generators reserved at ingestion) | unseen generation mechanisms |
| L4 | ≥1 held-out manipulation tool/framework (reserved at ingestion) | unseen editing mechanisms |

Prefer L3 generators *newer* than any training generator (temporal realism). L3/L4 membership is
chosen at ingestion, recorded in the frozen-legs manifest, and never revisited.

## Datasets — R1 verified 2026-09-01, details in `reference/dataset-availability-tb-e3.md`

- **Synthetic environments:** Community Forensics (primary: 4,803 generators, per-image labels,
  ungated) + GenImage via the **unbiased-genimage.org bias-controlled splits** (raw GenImage has a
  documented JPEG/size confound); Synthbuster as a small clean env; ArtiFact optional (200 px,
  pre-JPEG'd); ELSA D3 optional (LAION link-rot on reals).
- **Tampered environments (scarce, thesis-critical):** AUDITS (530k, masks, 11 method labels —
  supersedes the never-released MAGIC) + **TGIF2** (not TGIF: adds FLUX.1 + random masks against
  boundary-cheating) + SAGI-D (dedup vs RAISE — it sources from it) + IMD2020 (the only human-made
  in-the-wild edits). DEFACTO optional (COCO leakage tax). NIST MFC: **file the request on day 1**
  (portal down, agreements take days–weeks; email mfc_poc@nist.gov), treat as a later bonus.
  CASIA v2 only re-encoded to uniform JPEG (metadata alone scores ~0.92 AUC on it raw), never
  headline. AutoSplice only via its JPEG-75 control variant.
- **Real reinforcement:** VISION (primary: 35 devices, CC BY-SA, and its Facebook/WhatsApp/YouTube
  re-shares cover the web-recompressed-real cell natively); RAISE (RAW, form-gated; also
  Synthbuster's paired real source). Span camera-native AND web-recompressed reals.
- **Eval-only:** So-Fake-OOD; Chameleon (email-gated — request on day 1).
- **Excluded:** FakeClue (policy); So-Fake-Set (sibling of L2); WildFake (sole host ModelScope,
  license unverifiable); Dresden (official host dead, mirror provenance unofficial); MAGIC (never
  released); FF++-derived content unless identity-quarantined.
- **L4 candidates:** reserve one editing tool never trained on — e.g. Firefly (in TGIF2) or one
  AUDITS method.

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
- 2026-09-01 — R1 availability verification completed (three web research passes; details in
  `reference/dataset-availability-tb-e3.md`). Shortlist updated: MAGIC→AUDITS, TGIF→TGIF2,
  WildFake/Dresden dropped, GenImage restricted to bias-controlled splits, CASIA/AutoSplice
  admitted only with format controls. Day-1 actions added: NIST MFC request, Chameleon email.
  Preprocessing-policy requirement added to the instrument section.
- 2026-09-01 (execution, R2 onward) — protocol constants chosen and registered BEFORE any arm
  was fitted; implementation in `src/trustfake/curation/` on branch tb-e3-curation-ladder:
  - **Instrument validated**: cached-feature fits reproduce TB-E2's B/16 in-domain cell to
    ≤0.0095 on every metric (accuracy 0.9092 vs 0.9093; `jobs/tb_e3/sanity_b16.py`).
  - **G2 threshold**: max cosine ≥ 0.95 in frozen-B/16 space (exact dups ≈ 1.0). Applies vs
    L1–L4 to EVERY arm including C0 (the gate protects the measurement, not the arm). C1's
    within-pool dedup is within-CLASS only — an original and its edit are a cross-class
    near-identical pair and are the tampered signal, not dirt. **G3 minimum**: 200/cell.
  - **L3 frozen**: entire generators DALL·E 3, Midjourney v5, Adobe Firefly (Synthbuster
    folders; reserved across all environments by name pattern) + a seeded 6k CF-Small real
    reserve as negatives. Newest closed/commercial available in-pool; L2 already carries the
    2025-era shift (FLUX_2, GPT4o, Ideogram3, Recraftv3 — verified in G4 sheet).
  - **L4 frozen**: AUDITS PowerPaint (test-only in AUDITS by its own split design, zero
    training presence) with AUDITS test-Authentic negatives; plus TGIF ps-sp
    (Photoshop/Firefly generative fill) with TGIF originals as negatives when cached. If
    SAGI-D is ever admitted to training, its PowerPaint rows must be dropped (L4 conflict).
  - **C-arm operationalizations**: C0–C2 carry the naive unmappable→synthetic fold
    (colleague-repo default); C4/C5 build on strict-drop (H2 tie default); C5 budget =
    |C4|, so C4/C5a/C5b compare allocation at a fixed total. Note: the ingested shortlist
    has **zero unmappable rows** (every dataset's fake modality is known per-image), so
    C3a ≡ C3b structurally and H2 resolves by the tie rule unless a binary-fake env lands.
  - **GenImage deferred** (open item, not a drop): 654 GB monolithic on Dataverse or
    Drive-quota gdown; does not fit the box's disk beside the core (1.3 TB free at start).
    Unbiased-genimage metadata CSV (224 MB) noted for a later, subset-based admission.
  - **Not yet actioned, needs a human**: NIST MFC email, Chameleon email, RAISE form.
- 2026-09-01 (later) — Kaggle access token provided by the user; **SAGI-D admitted** to the
  pool (giakop/sagi-d, 95,839 fakes / 6 modern inpainting tools + sequential multi-tool
  combos, sources COCO 64.6k / RAISE 25.7k / OpenImages 5.6k test-only; in-env originals
  ingested as reals via src_path). The frozen-L4 conflict rule is realized in the pool
  filter: any row whose generator mentions PowerPaint (pure or in a combo, ~18.6k rows)
  is excluded from every arm. RAISE-lineage overlap is left to G2. ArtiFact stays out
  (optional in R1; 200 px pre-JPEG'd, low marginal value beside CF-Small).
- 2026-09-01 (pre-fit amendment, from the scratch smoke run) — **C2 is propensity-stratified
  matching.** Both exact schemes were measured on the smoke pool and annihilate it — joint
  5-axis cells keep 3%, sequential per-axis min-quotas keep 0.5% (SID-Set to zero) — because
  class-conditional nuisance distributions barely overlap (CF fakes PNG-native vs JPEG-heavy
  reals: CASIA's lesson at pool scale; itself a headline observation about naive pooling).
  Final scheme, standard in causal matching: per environment, fit the G1 nuisance classifier
  out-of-fold, stratify into deciles of predicted propensity, equalize class counts within
  strata — balances every covariate G1 reads while retaining the whole overlap region. The
  arbiter of shortcut death stays the per-arm G1 re-run recorded with every fit. Amended
  BEFORE any official fit; the smoke run also caught two cache-key collisions (CF image_name
  reuse, AUDITS NEWS/COCO id overlap) and AUDITS' COCO zero-padded filenames.
- 2026-09-01 (pre-fit interpretation + preview observations; official run untouched):
  - **G1's "fixed, not trained" binds C2 and above** — the arms whose claim is cleanliness.
    C0/C1 fail G1 by construction (being dirty is their purpose; a literal reading would
    forbid the ladder's own baselines and make H1 unaskable), so on the naive rungs G1 is
    a reported diagnostic. Resolved before the official run.
  - Preview (partial pool, 1 seed, provisional): naive-pool G1 is 0.995-1.0 in EVERY
    multi-class environment — the headers-only shortcut is total. Propensity-C2 drives
    per-env G1 to ~0.51, at heavy cost: AUDITS retains 6 rows, SID-Set 9 (near-zero
    nuisance overlap: AUDITS re-encodes manipulations, SID squares its fakes) — their C2
    cells are G3-flagged; C2 is effectively matched-CF + matched-IMD2020 + single-class
    passthroughs. C5b's k-center partially re-imports nuisance outliers (arm G1 back to
    ~0.61). All provisional; the official 48-fit run decides everything.
