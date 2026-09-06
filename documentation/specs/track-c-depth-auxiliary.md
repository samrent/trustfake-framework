---
type: spec
title: Track C — auxiliary depth: multi-task robustness and a depth-consistency rejection score
status: active
as_of: 2026-09-03
source: "registered 2026-09-03 with the code (branch claude/depth-auxiliary-robustness-track-b98707); no number yet"
tags: [track-c, experiments, depth, robustness, rejection-score, spec]
links: [handoff/current-state, decisions/depth-head-lives-in-the-model-group, decisions/ood-thresholds-come-from-in-domain-calib, decisions/track-a-and-track-b-are-separate-tables, gotchas/deterministic-mode-throws-on-median-and-bilinear-backward, gotchas/three-scorings-of-one-checkpoint-collide-in-merge-by-key-collators, gotchas/freezing-a-backbone-must-not-use-no-grad, gotchas/gpu-concurrency-is-negative-on-this-box]
---

# Track C — auxiliary depth

**For the executing agent.** Self-contained given this repo plus this brain. Read
`handoff/current-state.md` and the linked leaves first. Everything runs on the 3090 box from
`~/Desktop/FILES/PROJECTS/trustfake/framework`; nothing here has been run yet -- the code and
tests exist, the numbers do not. `jobs/track_c_depth.sh` is the runbook in executable form.

## Objective

Test whether monocular depth adds value to synthetic-image detection on two axes:

1. **Multi-task depth regularisation** (Mao et al., ECCV 2020, *Multitask Learning Strengthens
   Adversarial Robustness*): an auxiliary depth head trained beside classification,
   `L = CE + λ·L_depth`, dropped at inference. Hypothesis: the attacker must fool two objectives
   with misaligned gradients through one backbone, so robust accuracy improves modestly. On
   synthetic-image detection this would be a new data point.
2. **Depth-consistency rejection score**: the per-image residual between the model's own depth head
   and a frozen teacher run online on the same input, fed through the existing failure-detection
   and selective-classification metrics. Hypothesis: attacks crafted against the classifier drag the
   shared backbone, the head drifts from the teacher, and the residual detects attacked or
   misclassified inputs better than `1 − MSP`.

Answers: (a) does depth help clean accuracy, (b) does the Mao et al. bump reproduce here, (c) does
depth consistency beat max-prob as a rejection score under attack.

## Policy constants

- **ε = 8/255** (`adv_eps=0.03137`), 7 inner steps, 2-epoch warm-up — the Track A protocol. Not
  the parallel fork's 4/255.
- **Thresholds and the combined score's reference come from the in-domain calib split**
  (`calib_datamodule=sid_set` on So-Fake-OOD). `src/test.py` fits the combined score's ECDF
  reference there, after temperature and before the moderation gate.
- **Track C is its own table.** The baselines (`standard`, `pgd_at`) are trained by the chain
  under identical settings; nothing is borrowed from the e8_* arms (whose fit profile is not
  established) or from Track B.
- **FakeClue excluded** (standing policy), and the depth score refuses a binary fold anyway.

## Instrument — held fixed across every arm

- **Teacher:** Depth Anything V2 small, `depth-anything/Depth-Anything-V2-Small-hf` at revision
  `5426e4f0f36572d16453bbda7a8389317b1bef99`, frozen (`requires_grad_(False)`, never `no_grad`
  around the student). Its own preprocessing lives inside `trustfake.depth.DepthTeacher`: the
  checkpoint's resize rule (224 → 518, sides a multiple of 14), ImageNet mean/std, bicubic. The
  datamodule never normalises for it. Precision: fp16 autocast for the offline precompute
  (no gradient), fp32 online — a gradient through an fp16 teacher underflows, and a white-box
  attack would silently see the student half only. Images are decoded EXIF-corrected exactly as
  HF datasets decodes them for the datamodule (`trustfake.data._images`).
- **Frame:** zero-median / unit-mean-absolute-deviation per image (`trustfake.losses.depth`),
  one implementation shared by the precomputed targets, the training loss and the score. A
  constant prediction scores exactly 1.0. The median is sort-based (see the determinism gotcha).
- **Targets:** precomputed once per fit shard (`src/precompute_depth.py`, store
  `$DATA_PATH/sid_set_depth/dav2_small_518_224`), float16 at 112×112 (the head's grid at a 224
  view), keyed by manifest uid, with `manifest.json` recording teacher, revision, input size,
  grid, view (`image_size`, `squarecrop`, resize mode). The datamodule refuses a store whose
  recorded view differs from its own and any fit row without a target.
- **Head:** `DepthHead` — lateral 1×1 on layer3/layer4, merge at stride 16, three nearest+conv
  upsamplings to stride 2, 1×1 to one channel; no dropout, no output activation; ~0.69M params.
  Reached only via `forward_with_depth`; `ResNet.forward` is bit-identical to before.
- **Loss:** scale-and-shift-invariant L1 in the shared frame, fp32 under autocast.
- **Backbone/recipe:** ResNet-18 ImageNet init, seed 1, 12 epochs, profile `train`, batch 32,
  Adam 5e-4, `robust_val_steps=3`, selection on `val_f1_score` — the Track A recipe.
- **Adversarial variant:** `pgd_at_depth` keeps `pgd_at`'s CE-only inner maximisation; the head is
  supervised on the ADVERSARIAL input against the CLEAN image's target. AWP, if ever enabled,
  ascends the multi-task loss.

## Evaluation legs

| leg | set | thresholds from | measures |
|---|---|---|---|
| L1 | SID-Set held-out slice (profile `train`, `limit_test=1000`) | its own calib | in-domain |
| L2 | So-Fake-OOD | SID-Set calib | shift |

Conditions (cheap first): `clean`, `pgd`, `query_underconf`, `query_overconf`, `ace_uint8`,
`+corruption=jpeg`. Square/AutoAttack are a separate, expensive run.

## Attack scoring — what the attack sees when the score is depth-aware

| cell suffix | `depth_attack_scoring` | meaning |
|---|---|---|
| `_tr` | `transfer` | the attack optimises `1 − MSP`; the depth score is computed on the final perturbed batch. "An attack crafted against the classifier, scored by depth" — the standard-attack protocol. |
| `_wb` | `white_box` | the attack optimises the depth score itself, teacher included (gradient flows through the teacher). The adaptive number; for the `query_*` attacks it is a gradient-free adaptive attack for free. |

Prediction-axis attacks (`pgd`, corruptions) never read the score, so only `transfer` runs for
them (identical outcome, a fraction of the cost). **Caveat carried by every depth-score number:**
a gradient adaptive attack whose loss also minimises the residual is a follow-up (needs an attack
class with a lazily built teacher; `AttackResult` has no field for the achieved residual).

## Validity gates — a cell that fails a gate is not read

- **G1 (store):** `manifest.json` matches the run's view; every fit uid covered; every stored
  map finite. Enforced by `setup()`/the precompute, never warned.
- **G2 (degeneracy, the σ seam):** on the in-domain calib split, |Spearman ρ| between the depth
  residual and `1 − MSP` must be **< 0.98**; ≥ 0.98 means the "new" uncertainty is the old one
  relabelled (TODO §4). Computed by `src/test.py` for every depth-aware scoring and written to
  `depth_calib_gate.json` beside the metrics; the collator prints a verdict per arm.
- **G3 (clean floor):** an arm below 0.75 clean accuracy is a broken fit, not a result.
- **G4 (collapse):** `nat_n_operating_points ≥ 32` for the scored condition.
- **G5 (smoke):** before any store time is spent, `jobs/track_c_depth.sh` runs, on the smoke
  profile with the real teacher: two training batches of `pgd_at_depth` under the real trainer
  config (deterministic, bf16-mixed); one full `depth_combined` evaluation with white-box FGSM
  (calib pass, gate, temperature; test split capped to `SMOKE_LIMIT=64`); and
  `jobs/track_c_smoke_teacher_grad.py`, which proves on the device that the white-box gradient
  reaches the input THROUGH the teacher. The CUDA-only failure modes (deterministic-mode ops,
  fp16 gradient underflow through the teacher) cannot be seen on the Mac.

## Arms

| arm | pipe | model | adds | isolates |
|---|---|---|---|---|
| `standard` | `standard` | `resnet18` | — | the baseline |
| `standard_depth_l{λ}` | `standard_depth` | `resnet18_depth` | depth term | (a) clean effect of depth |
| `pgd_at` | `pgd_at` | `resnet18` | AT | the AT baseline |
| `pgd_at_depth_l{λ}` | `pgd_at_depth` | `resnet18_depth` | AT + depth term on x_adv | (b) the Mao et al. bump |
| `trades`, `trades_depth_l{λ}` | optional (`WITH_TRADES=1`) | | | second AT arm |

λ ladder: default `1.0`; `LAMBDAS="0.1 0.3 1.0"` for the sweep. One seed first; a borderline
result gets seeds, not a softened rule.

## Metrics per cell

`accuracy`, `fd_auroc` (Φ), `aurc`, `n_operating_points`, `recall_tampered`,
`moderation_2axis_residual_risk`, `moderation_2axis_review_rate` (the two-axis indicators are
the ones that read the score; the gate-free `moderation_*` pair is score-independent); per arm
the G2 correlation.

## Decision rules — written before the first fit

- **H(a)** depth helps clean accuracy iff `standard_depth` − `standard` ≥ +0.01 accuracy on L1
  at every λ tried, else "no clean effect".
- **H(b)** the Mao et al. bump reproduces iff `pgd_at_depth` beats `pgd_at` on `pgd` accuracy by
  ≥ +0.02 with clean accuracy within −0.01; a bump bought with clean accuracy is a trade, not a
  gain.
- **H(c)** depth consistency beats max-prob as a rejection score iff, on the SAME arm, Φ under
  `pgd` and under `query_underconf` (`_tr`) is higher for `depth` than for `msp` by ≥ 0.03 with
  G2 passed. The `combined` score is read second: it is the deployable form only if it is never
  worse than the better component.
- **H(d)** the adaptive caveat is quantified by `depth_wb` vs `depth_tr` on the query attacks:
  the drop is the number that goes beside every `_tr` claim.
- Watchdog: any `_wb` Φ at or below 0.5 while `_tr` is high means the score is trivially
  attackable — report it as such.

## Runbook

1. On the box: `git pull`, `uv pip install transformers` into `.venv`, confirm network for the
   ~99 MB teacher fetch (or pre-seed `~/.cache/huggingface`).
2. `LIMIT_SHARDS=1 bash jobs/precompute_depth_targets.sh` — a timing probe; read img/s.
3. `setsid bash jobs/track_c_depth.sh` (smoke → store → arms → cells). Sequential only.
4. `python3 jobs/summarise_track_c.py $LOGS_PATH/track_c_depth > RESULTS_track_c.md`.
5. Back up `_runs/`; write `snapshots/<date>-track-c-first-results.md`; record decisions in
   `decisions/`; update `handoff/current-state.md`; update this spec's status and change log.

## Definition of done

The matrix {standard, pgd_at} × {baseline, +depth} at one λ, both legs, all conditions, three
scorings, G1–G5 recorded, and H(a)–H(d) each answered with a number or "not readable, because".

## Change log

- 2026-09-03 — registered with the implementation; no run yet.
- 2026-09-03 — adversarial review of the diff (4 reviewers, 3 refuters per finding): fixed the
  smoke step's Hydra override, temperature scaling on an unfitted combined score, fp16 teacher
  gradient underflow (online teacher now fp32), EXIF-inconsistent precompute decoding, a
  training-time wrapper/arm combination that crashed on the first batch, and the collator's
  score-independent moderation columns and mislabelled gate rows.
