# TrustFake — remaining work

State as of milestone TF_02. Everything below is what is *not* done; what is
done is in the README and `CHANGELOG.md`. 545 unit tests pass, and every attack
and every training arm has been exercised end-to-end on real SID-Set shards on
a 3090 — but on a smoke-sized slice, so **no publishable numbers exist yet**.

## 0. Validated so far

- Device abstraction (`trustfake.utils.resolve_device`): CUDA → MPS → CPU, so
  the same code runs on the 3090 and on this Mac's Metal.
- The full pipeline runs end-to-end on real SID-Set. Every training pipe
  (`standard`, `pgd_at`, `trades`, `at_kl`, `mart`, `at_conf`, `conf_reg`,
  `evidential_adversarial`) trains, with and without AWP; every attack config
  and every corruption config evaluates and reports the full metric suite.
- The thesis reproduces on real data. On a small real detector, ACE leaves
  accuracy identical (0.558 → 0.558) while failure-detection AUROC collapses
  (0.555 → 0.0003) and the WP4 residual risk on auto-decisions goes 14% → 48%
  at a 15% clean SLA — an accuracy monitor sees a healthy system. The geometry
  baseline `width == height → fake` scores 0.97 on the real test slice.
- These are on small slices and few epochs — the direction is right, the
  magnitudes are not publishable. Section 1 is what turns this into results.

## 1. Produce the results (blocking — needs GPU time + the full dataset)

The method, the baselines and the protocol controls are implemented and
unit-tested. No numbers exist yet. This is the actual deliverable, and it is
now the *only* thing between the repo and a paper-shaped result.

- [ ] Fetch the full SID-Set shards to `${DATA_PATH}/sid_set`
      (`jobs/download_sidset.sh`).
- [ ] Train each arm on the same protocol, **matched on optimiser steps**:
      `standard`, `pgd_at`, `trades`, `at_kl`, `mart`, `at_conf`, `conf_reg`,
      `evidential_adversarial`, each with and without `awp_gamma`.
- [ ] Select each defence arm on `val_robust_accuracy`
      (`robust_val_steps=5 selection_metric=val_robust_accuracy`), not on the
      clean macro-F1 it deliberately trades away. State in the write-up which
      arm was selected on which metric.
- [ ] Evaluate every arm clean, under attack, and under corruption; collect
      risk–coverage, AURC/AUGRC/E-AURC **with `n_operating_points` beside
      them**, calibration, and the WP4 indicators.
- [ ] Build the headline comparison: EV-AT and the two confidence-axis arms
      against the MSP / temperature-scaling baselines, under PGD / AutoAttack /
      A³ / ACE — the robustness–uncertainty trade-off the keystone paper
      reports.
- [ ] Report the minimum-norm column (`bb`, `pdpgd`, `deepfool`, `cw`) as the
      median perturbation norm among **successful** samples, not as an attacked
      accuracy. Confirm the norms agree across the four: a large disagreement
      means one has stalled, and a stalled min-norm attack reads as robustness.
- [ ] Sanity floor: run `src/baselines.py` and read every model accuracy
      against `width == height → fake`, not against 0.5. Then re-run the
      headline on a geometry-controlled subset
      (`datamodule.datamodule.geometry_filter=matched`) and report both.
- [ ] Pick a forensic epsilon and confirm PGD-AT does not collapse at 8/255
      (use `adv_warmup_epochs` or lower `adv_eps`, e.g. 2/255).

## 2. Adaptive evaluation of the confidence-axis defences (the real risk)

`at_conf` and `conf_reg` are currently **non-adaptive** results: they are
measured against a fixed attack, including the one `at_conf` trains on. That is
the standard way a robustness claim dissolves (Athalye et al. 2018), and it is
the first thing a robustness researcher will ask about.

- [ ] Build an adaptive attacker against each: re-tune ACE / over-confidence
      hyperparameters *against the defended model*, and try a confidence attack
      that optimises through the defence's own objective.
- [ ] Report adaptive and non-adaptive numbers side by side. A negative result
      here is publishable — the question is open — but an unlabelled
      non-adaptive number is not.

## 3. Open question for the PI (deferred — immaterial while testing the pipeline)

- [ ] Is the official SID-Set test split available to the group? The manifest
      currently carves "test" from the validation split (the authors withhold
      the real test set). If the real split is reachable, switch the manifest to
      it and drop the "held-out validation slice" provenance wording.

## 4. Method extensions still open

- [ ] **Localization sub-task (optional).** The tampered third of SID-Set
      supports localization (SIDA adds a textual-explanation head) — out of the
      current classification + selective scope, noted as a possible extension.
- [ ] **Independent uncertainty producer (the σ seam).** The uncertainty gate
      always reads the wrapper's own score, so it cannot test an uncertainty
      that is *independent* of the confidence the attack moves — which is the
      one design that survives a confidence attack by construction. Needs a
      producer seam plus a degeneracy check (rank correlation against
      `1 − MSP`; |ρ| ≥ 0.98 means the "new" uncertainty is the old one
      relabelled).

## 5. Housekeeping

- [ ] Optional: send the reference fixes / structure improvements upstream —
      commit 1 is the framework exactly as received, so `git diff` against it is
      a clean patch series. Two are worth offering regardless of the rest: the
      corrected uncertainty-gate invariant, and the `NaN`-not-`0.0` failure
      AUROC guard.
- [ ] The original WP1 repo (`wp1/src/moderation.py:62`) still carries the
      uncertainty-gate claim that was corrected here. Fix it there too.

## Done since TF_01

All of the previous §2 (attacks A³, PDPGD, BB — now native, no new dependency)
and all of §4 except localization: geometry-controlled evaluation, the
common-corruption ladder, AWP, and the IKL EMA rate. §5's merge decision is
resolved (PR #1 merged as `294e2c0`) and the demonstrator notebook is in
`notebooks/`.
