# TrustFake — remaining work

State as of the `port/tier1` branch (PR #1). Everything below is what is *not*
done; what is done is in the README and the PR description. 256 unit tests pass,
but nothing here has been trained or measured on real data.

## 1. Produce the results (blocking — needs GPU + dataset)

The method and baselines are implemented and unit-tested, but no numbers exist
yet. This is the actual deliverable.

- [ ] Fetch SID-Set shards to `${DATA_PATH}/sid_set` (`jobs/download_sidset.sh`).
- [ ] Train each arm on the same protocol, matched on optimiser steps:
      `standard`, `pgd_at`, `trades`, `evidential_adversarial`
      (`experiment.training_pipe=...`).
- [ ] Evaluate every arm clean and under attack (`+attack=fgsm|pgd|autoattack|ace|
      param_ace|evidence_pgd|...`); collect risk–coverage, AURC/AUGRC/E-AURC,
      calibration (ECE/NLL/Brier), and WP4 moderation indicators.
- [ ] Build the headline comparison: EV-AT vs MSP / temperature-scaling
      baselines under PGD / AutoAttack / ACE — the robustness–uncertainty
      trade-off the keystone paper reports.
- [ ] Sanity floor: run `src/baselines.py` and read every model accuracy
      against `width == height → fake`, not against 0.5.
- [ ] Pick a forensic epsilon and confirm PGD-AT does not collapse at 8/255
      (use `adv_warmup_epochs` or lower `adv_eps`, e.g. 2/255).

## 2. Attacks not yet ported (dependency decision)

Each needs a non-PyPI research repo or a new heavyweight dependency, so none was
reimplemented (a weak/wrong attack silently overstates robustness).

- [ ] **BB** (Brendel & Bethge, NeurIPS 2019) — available in `foolbox`. Cleanest
      to add via a foolbox adapter if `foolbox` is accepted as a dependency.
- [ ] **A³** (Adaptive AutoAttack, CVPR 2022) — author repo, not on PyPI.
- [ ] **PDPGD** (Matyasko & Chau 2021) — author repo; primal-dual proximal.

## 3. Open question for the PI

- [ ] Is the official SID-Set test split available to the group? The manifest
      currently carves "test" from the validation split (the authors withhold
      the real test set). If the real split is reachable, switch the manifest to
      it and drop the "held-out validation slice" provenance wording.

## 4. Protocol and method extensions

- [ ] **Geometry-/format-controlled evaluation subset.** The trivial baselines
      show SID-Set is squares-are-fake biased; the honest fix is a controlled
      eval subset (a WP1 protocol change), not a better model. Add a manifest
      option that filters to square-only (or matched-geometry) rows.
- [ ] **Common-corruption evaluation.** The keystone benchmark reports clean +
      adversarial + common-corruption. Port a perturbation ladder (jpeg_q*,
      downscale_*, webp) as evaluation conditions.
- [ ] **AWP** (Adversarial Weight Perturbation). The EV-AT ablation reports AWP
      as an additional, non-substitutable gain on top of the evidential loss +
      REA. Not implemented; would tighten a faithful reproduction.
- [ ] **IKL global weight.** The class-wise global weight uses an EMA buffer
      updated during training; confirm it matches the paper's running-mean
      definition closely enough on real runs, and expose the EMA rate in config.
- [ ] **Localization sub-task (optional).** The tampered third of SID-Set
      supports localization (SIDA adds a textual-explanation head) — out of the
      current classification+selective scope, noted as a possible extension.

## 5. Housekeeping

- [ ] Decide merge of PR #1 into `main` (or keep as a review branch).
- [ ] Optional: send the reference fixes / structure improvements upstream —
      commit 1 is the framework exactly as received, so `git diff` against it is
      a clean patch series.
- [ ] Notebook/demo: a small demonstrator that shows the risk–coverage curve
      clean vs adversarial and the abstention rule breaking under ACE.
