# Changelog

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
