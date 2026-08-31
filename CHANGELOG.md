# Changelog

## Unreleased

**Making the tampered-class failure measurable.** The class the detector
struggles with is the tampered third of SID-Set, and until now the harness
could not say so: per-class metrics were commented out, the detection AUROC
folded both fake modalities into one number carried by the easy half, and the
input pipeline resized every image to 224² — a ~4.6× low-pass that erases the
high-frequency, local evidence a tampered edit leaves before the model ever
trains on it.

- **Per-class classification rows.** `recall_real / recall_synthetic /
  recall_tampered` (precision beside each), named rather than numbered, in
  every classification table — train/val logs and every eval condition,
  clean and attacked. Per-class accuracy and F1 deliberately omitted:
  multiclass accuracy at `average="none"` *is* per-class recall, and F1 is
  derivable.
- **Per-modality detection AUROC.** `detection_auroc_tampered` /
  `detection_auroc_synthetic`: the deployed `p(fake)` score against real,
  one fake modality at a time, NaN where a split holds no rows of it. A
  tampered ranking at chance is invisible inside the all-fakes fold (a test
  pins the case where the fold reads exactly 0.5 while the breakouts read
  1.0 and 0.0) and cannot hide in its own row.
- **`input_mode: crop`** on the datamodule (default `resize`, the historical
  behaviour, byte-identical): fixed-size crops at native resolution — random
  for train, centre for val/calib/test — no resampling anywhere. Preserves
  the pixel statistics the resize destroyed; costs a bounded field of view
  (an off-crop edit is invisible, so tampered recall is *understated* on
  large images — stated in the config and README, dense/multi-crop scoring
  is the follow-up in TODO §4). Refused in combination with `squarecrop`,
  whose whole job is to feed the resize that crop mode removes.

Also recorded: the SID_Set shards already include a `mask` column (binary
mask of the manipulated region), so mask-guided crops, area-stratified
tampered recall and the localization extension need no new download.

**The confidence-axis audit, ported to a CLIP encoder (seam 1).** The attack
battery and the metric stack were already written against
`TrustFakeWrapper`'s `x -> (logits, probs, preds, uncertainty)` contract and
nothing else, so auditing a vision encoder for the same failure mode needed a
module that satisfies that contract, not a second harness.

- **`trustfake.models.torch.clip`**: `CLIPZeroShotClassifier` — a frozen image
  tower plus frozen, L2-normalized text prototypes as an `x -> logits` module
  (`logit_scale * cos(f(x), P)`), dropping into `BaseWrapper` unchanged.
  Prototypes are a buffer, so a checkpoint round-trips without the text tower;
  `build_text_prototypes` does prompt ensembling over normalized embeddings;
  `clip_zeroshot` builds from an `open_clip` checkpoint with the import kept
  lazy, so the test suite runs without the dependency and without the network.
- **The transfer is pinned by test, not asserted.** `QueryConfidence` — the
  gradient-free instrument — runs against the zero-shot head and keeps all
  three of its guarantees there: no parameter gradient touched, argmax
  preserved, uncertainty moved in the requested direction, inside the budget.
  That is the claim "the apparatus is task-agnostic" made falsifiable.
- **Two silent failure modes are written down as tests.** Double normalization
  (datamodule `Normalize` on top of the encoder's own) raises nothing and
  changes every number; CLIP's native `logit_scale` (~100) saturates `1 − MSP`
  into a constant, so failure detection ranks nothing while accuracy is
  bit-identical — the confidence axis destroyed by a *configuration*, which is
  the same lesson as the confidence attack, arriving through the front door.

Encoder parameters are frozen (seam 1 audits a pretrained encoder); a test pins
that this does not block the input gradients the gradient attacks need.

Smoke on the real `ViT-B-32/laion2b_s34b_b79k` checkpoint (procedural images,
**not** SID-Set — no dataset number is claimed here): the head builds, prototypes
come out `(3, 512)` and unit-norm, `logit_scale` is 100.0 as expected, and
zero-shot semantics are correct end-to-end, which is what says the normalization
is right. `QueryConfidence` at eps = 8/255, 150 queries, then moves the
confidence axis on a CLIP encoder with the instrument unchanged: `over` takes
mean `1 - MSP` from 3.32e-02 to 1.12e-04 (a ~300x collapse), `under` to 1.35e-01,
both with the argmax preserved on every sample, inside the budget, and touching
no parameter gradient. Saturation turns out to be input-dependent: the same head
reads 6.85e-05 on unambiguous inputs and 3.32e-02 (max 0.27) on ambiguous ones,
so "the native scale saturates" has to be checked against the real distribution
rather than assumed.

## TF_03 — 2026-08-26

Three adversarial audits over the TF_02 code, then the fixes. 31 bugs, every
one reproduced by running it before it was fixed and by a test that fails
without the fix. 696 unit tests (was 545). Every experiment is now pinned at
**8/255**.

None of these announced itself. Each one trained, converged and produced a
plausible curve — which is why they are listed rather than quietly repaired.

### Results-corrupting
- **BatchNorm contamination on every adversarial arm.** The inner PGD ran the
  model in train mode, so a 10-step adversary made 12 BatchNorm updates per
  batch instead of 1: every intermediate iterate folded into the running
  statistics, and every attack forward normalised by a perturbed batch rather
  than by the deployed model. On the default ResNet-18 this touched every WP3
  number. Each arm now makes exactly the updates its objective justifies.
- **A minimum-norm attack that lost to plain PGD.** PDPGD reported robustness
  a fixed-budget PGD refuted at the same radius (0.719 vs 0.086 success on one
  seed) — an un-normalised primal step that tracked the model's logit scale,
  and a proximal threshold three orders of magnitude below the perturbation it
  was meant to shrink. All four minimum-norm attacks are now guaranteed never
  to lose to a fixed-budget witness at the same radius, and that guarantee is
  a test.
- **The geometry-controlled baseline printed the raw split's floor.** On real
  shards the `nonsquare` subset's true floor is 1.0000 where the headline
  printed 0.6657, so a model scoring 0.85 read as clearing a bar it was far
  below.
- **ECE understated by 26x** on saturated confidence — the exact regime it is
  reported beside torchmetrics' `ece` in.

### Silent selection failures
- `mode: max` was hardcoded while `monitor` was configurable, so any
  lower-is-better metric kept the **worst** epoch.
- Robust validation attacked at the warm-up epsilon, so on a frozen model the
  metric fell 1.0000 -> 0.0352 across five epochs purely from the ramp, and the
  checkpoint kept epoch 0.
- **AWP was a no-op on `conf_reg`**: the planned with/without ablation would
  have produced two bit-identical arms.
- `standard`, `conf_reg` and `evidential_adversarial` accepted
  `robust_val_steps`, dropped it silently, then died on "metric not available"
  — so the classical baselines could be selected on robustness and the
  flagship arms could not.

### Also
`ace` and `ace_uint8` both logged as `ace` (one condition overwriting the
other); `ParamACE` reported its label-free default as label-using; `real_class`
never reached the detection AUROC; the manifest's leakage firewall was
`assert`, stripped by `python -O`; EV-AT's IKL statistics absorbed validation
rows via Lightning's sanity pass; the four minimum-norm attacks disagreed about
what a failed sample reports.

### Added
`src/sweep.py` — a method comparison pinned at 8/255, ranked on **confidence
resilience** (mean failure-detection AUROC under `ace_uint8` and `overconf`,
subject to a clean-accuracy floor) rather than on robust accuracy. It detects
training collapse explicitly: a degenerate model posts a respectable accuracy
at the majority class, and what exposes it is the uncertainty score going
near-constant, so anything under 32 distinct operating points is flagged and
excluded rather than scored.

## TF_02 — 2026-08-25

Second milestone: the attack and defence surfaces are complete, and the two
protocol controls the dataset demands are in. Nothing in `TODO.md` §2 (unported
attacks) or §4 (protocol/method extensions) is outstanding. 545 unit tests;
every arm and every new attack exercised end-to-end on real SID-Set shards on
the 3090.

### Attacks — the three that were "not yet ported" (§2), plus two gaps
- **BB** (Brendel & Bethge, NeurIPS 2019), **PDPGD** (Matyasko & Chau 2021) and
  **A³** (Liu et al., CVPR 2022), all as **native reimplementations** — no
  non-PyPI research repo and no new heavyweight dependency. Where an
  implementation departs from its reference the docstring says so and a test
  pins the consequence.
- **PGD-L2**: a fixed budget in L2, missing from the port. Robustness does not
  transfer between norms, so an L∞-only table cannot say the model is robust.
- **`ace_uint8`**: ACE on the 1/255 pixel grid — the realisable file-upload
  threat model, as distinct from an attacker with post-decode tensor access.
- **Attack taxonomy** (`attack_registry()`): family / direction / label use /
  norm / minimum-norm, built by instantiating each attack so it cannot drift
  from the code. Results can now be grouped by threat family, and rows produced
  with ground truth can be told from rows produced without it.
- `AttackResult` gained `l2_norm` and `success`. For a minimum-norm attack
  `eps` is a cap, not a budget: the norm it *needed* is the result, and a
  sample it could not solve is returned unperturbed and flagged rather than
  silently reported as a tiny perturbation.
- Ground-truth use is now opt-in (`use_labels`) on FGSM/BIM/PGD/PGD-L2,
  defaulting to the model's own prediction (the realisable threat model).

### Defences — the confidence axis, and the modifier
- **`at_conf`**: adversarial training whose inner maximisation is the
  *confidence* attack rather than cross-entropy. The first arm in the harness
  aimed at the failure the harness measures.
- **`conf_reg`**: a direct penalty on confident mistakes, no inner adversary.
- **`at_kl`**: the AT + consistency-KL hybrid — cross-entropy on the
  *adversarial* forward, which is what distinguishes it from TRADES.
- **`mart`** (Wang et al. 2020): margin-aware AT with a
  misclassification-weighted KL — the only classical arm whose objective is a
  function of the model's own confidence.
- **AWP** (Wu et al. 2020) as a composable modifier (`awp_gamma`), not an arm:
  it applies to every pipe including EV-AT, where the weight adversary attacks
  `L_EV + β·L_REA` rather than a cross-entropy proxy.
- **Robust model selection**: `robust_val_steps` + `selection_metric`, so a
  defence arm can be selected on `val_robust_accuracy` instead of on the clean
  macro-F1 it deliberately trades away.
- `ikl_ema` exposed: the IKL global weight is a running mean that does not
  converge on a short run, so its rate is a live knob rather than a constant.

### Protocol controls (§4)
- **Common corruptions** as a first-class evaluation condition
  (`+corruption=jpeg|webp|downscale|gaussian_noise|gaussian_blur`), kept out of
  the eps-ball contract because a corruption is distributional shift, not a
  bounded perturbation.
- **Geometry-controlled evaluation**, the honest answer to the dataset's
  `width == height → fake` shortcut: a `geometry_filter`
  (`none|square|nonsquare|matched`) on the reported splits, and a `squarecrop`
  pre-transform applied *before* the resize (after it, every image is already
  square and the crop is a no-op). Plus the decode-scale residue baseline that
  a centre crop does **not** remove, and a protocol-aware `headline()` so a
  controlled row never prints the uncontrolled 0.98 beside it.

### Metrics and moderation
- `n_operating_points` beside AURC — the guard against reading a saturated,
  temperature-dependent AURC as a precise number; float64 confidence upcast so
  fp32 softmax saturation cannot manufacture tie blocks.
- `risk@coverage`, `coverage_at_risk`, achieved (never target) coverage;
  uniform as well as block weighting; `rc_curve` now returns thresholds.
- **Detection AUROC** separated from failure AUROC and both reported — they
  answer different questions and conflating them is the cheap error.
- Failure AUROC returns `NaN`, not `0.0`, where it is undefined (no errors, all
  errors, constant confidence) — a split with no errors was reading as perfect
  failure detection.
- ECE variants (equal-mass, binary-domain, L1/L2/max) and top-1 accuracy beside
  the macro average.
- Moderation: `review_of_fakes` / `review_of_reals` (a gate that reviews the
  right traffic now scores differently from one that reviews random traffic),
  `n`, `accuracy` beside `full_coverage_error`, `p_fake_from(logits, T)`, and a
  markdown/JSON report of the one-axis vs two-axis comparison.

### Corrected
- An invariant inherited from the original code was **false**: the uncertainty
  gate was documented as unable to raise residual risk. It can — residual risk
  is a rate over a shrinking denominator, so escalating correct decisions
  raises it while the count of wrong auto-decisions is unchanged. The real
  invariants are on counts; docstrings corrected and pinned by tests.
- BB's trust-region radius now adapts. With a fixed radius, clipping to the
  valid pixel range can reject a step, and the attack then retries the same
  rejected step forever — stalling at its random starting point while still
  reporting a perturbation norm.
- PDPGD's L1-ball projection no longer uses `cumsum`, which has no
  deterministic CUDA kernel and therefore raised under the harness's
  `use_deterministic_algorithms(True)`. Replaced with a bisection that uses
  only deterministic reductions.

## TF_01 — 2026-08-25

First milestone: the TrustFake harness's advances ported into the framework
(PR #1, merge commit `294e2c0`). The framework now implements the full WP1–WP4
pipeline plus the evidential method it benchmarks.

### Evaluation & metrics
- Shard-disjoint fit/calib/test split manifest (closes the `val == test`
  selection leak); reproducibility/provenance checks (`trustfake.data.verify`).
- Selective metrics: tie-collapsed, block-weighted AURC/AUGRC + E-AURC.
- Calibration (3-class): temperature scaling on calib + ECE/NLL/Brier, reported
  clean vs adversarial.
- Trivial metadata baselines (`width == height → fake`): the floor accuracy is
  read against.

### Attacks (16 configs)
- Native: FGSM, BIM, PGD, DeepFool, C&W, TR; ACE and (η,ω)-ACE; over-/under-
  confidence; the evidence-targeted adversary. Widened attack interface
  (`AttackResult`) for accept-check logits and per-sample effective ε.
- Via the `autoattack` dependency: APGD, FAB, Square, AutoAttack.

### Methods & training
- **EV-AT** (arXiv:2607.03075): evidential Dirichlet head + posterior-entropy
  score, `L_EV`, the log-Dirichlet discrepancy (IKL/KL/L2), the evidence-
  targeted adversary, and the min-max training module.
- Adversarial-training baselines: PGD-AT and TRADES, with an ε warm-up.
- Pipelines: `standard | pgd_at | trades | evidential_adversarial`.

### Moderation (WP4)
- Two-threshold selective moderation fitted on calib and frozen, reported clean
  vs adversarial, with an uncertainty gate. Threshold fitting vectorised
  (~1600× on a 15k-row split).

### Infrastructure
- Device abstraction (`resolve_device`): CUDA → MPS → CPU.
- 262 unit tests; verified end-to-end on real SID-Set (a streamed partial
  slice) on Apple Metal.

### Validated
- The project thesis reproduces on real data: ACE preserves accuracy
  (0.558 → 0.558) while failure-detection AUROC collapses (0.555 → 0.0003) and
  the WP4 residual risk on auto-decisions goes 14% → 48% at a 15% SLA. The
  geometry baseline scores 0.97 on the real test slice.

### Not included (deliberate)
- Attacks A³, PDPGD, BB — each needs a non-PyPI repo or a new heavy dependency;
  a deliberate dependency decision. See `TODO.md`.
- Full-scale training/results on SID-Set (needs the cluster). See `TODO.md`.
