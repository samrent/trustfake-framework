# Changelog

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
