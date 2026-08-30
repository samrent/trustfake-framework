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
- [x] Epsilon is **pinned at 8/255** for every experiment — Madry's setting
      and RobustBench's headline column, so results are comparable to the
      field rather than only to each other. wp1 recorded training collapsing
      onto a constant output at this budget; that is now a claim the runs
      test rather than one the design routes around, and `src/sweep.py`
      flags collapse instead of scoring it. The measured 1-4/255 ladder
      found no collapse at 4/255 (clean 0.7005), so any threshold sits
      above it.
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

### The VLM port (seam 1 landed, unmeasured)

`trustfake.models.torch.clip` presents a CLIP encoder as the wrapper contract,
so the battery runs against it. Nothing has been *run*: the code path is pinned
by tests with a stub encoder, and no real checkpoint has been through it.

- [ ] **Install and smoke the real checkpoint.** `open_clip_torch` is declared
      but not in the lock; build `ViT-B-32/laion2b_s34b_b79k`, confirm clean
      zero-shot accuracy on a labelled transfer set is sane before reading any
      robustness number off it. A broken prompt set looks exactly like a robust
      model at chance.
- [ ] **Fit the temperature, then report it.** At the native `logit_scale` the
      confidence signal is constant and every failure-detection number is
      meaningless. Fit on calib, freeze, report T beside Φ — the whole point is
      that the confidence axis is not readable off an uncalibrated head.
- [ ] **Specify the gate (slot 3) for a non-forgery task.** `metrics/moderation.py`
      hardcodes `p_fake = 1 − P(real)` and `failure_detection` takes a
      `real_class` index. A zero-shot label set has no "real" class, so WP4
      residual risk is undefined until the abstain-vs-auto decision is
      re-specified for the transfer task. Until then, report Φ (FD-AUROC) and
      selective risk only — do **not** report a moderation number.
- [ ] **The anchor comparison.** Run `QueryConfidence` (gradient-free, the only
      measurement the validity law trusts) against the zero-shot head and put
      accuracy-vs-Φ beside the detector's 0.558→0.558 / 0.555→0.0003. Same
      protocol, same five metrics, different component: that IS the transfer
      claim, and it is falsifiable — Φ may well survive here, which is a result.
- [ ] **P2 — encoder propagation (the reason the port exists).** Swap the frozen
      tower for a FARE/TeCoA-robustified one, hold the head and prompts fixed,
      re-run. The prediction: downstream confidence-axis robustness rises with
      *no head retraining*, because encoder robustification flattens the
      geometry the confidence attack exploits. If it holds, the shared encoder
      is the fleet-wide control point for the confidence axis too.
- [ ] **Seam 2, once seam 1 has a number.** Representation-level: confidence
      from feature density / augmentation self-consistency / ensemble
      disagreement, which are decoupled from task logits by construction (the
      σ-seam item above, in its natural habitat). Blocked on a contract that
      does not assume logits — `CLIPZeroShotClassifier.encode` is the entry
      point. Note the honest catch: scoring faithfulness needs a correctness
      label, so a task re-enters through the back door and must be disclosed.

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
