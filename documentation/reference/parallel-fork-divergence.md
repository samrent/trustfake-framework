---
type: reference
title: A parallel fork of the framework — what differs and what it means
status: current
as_of: 2026-08-31
source: "diffed against a snapshot received 2026-08-31; a newer snapshot was announced"
tags: [collaboration]
links: [track-a-and-track-b-are-separate-tables]
---

A **divergent fork**, not a newer version. Deltas run both ways.

**Its SID-Set `test` split IS its `validation` split** — literally
`test_ds = dataset["validation"]`, with a `TODO` acknowledging it. Model selection and reporting
therefore share rows, so its SID-Set numbers are not held-out and are **not comparable** to this
repo's. Its FakeClue and So-Fake-OOD numbers are cleaner, because those are genuinely separate
datasets and sidestep the collapsed split.

**What it has that this repo lacked:** FakeClue, So-Fake-OOD, a combined SID-Set+FakeClue training
datamodule, augmented variants, an analysis/reporting layer (LaTeX tables + plots), and native
APGD/FAB/Square/AutoAttack/C&W implementations instead of the fra31 wrappers.

**What this repo has that it lacks:** the entire confidence axis (no attack taxonomy, no
QueryConfidence, no evidential head or EV-AT), selective classification (437 lines vs 78),
`FailureAUROC` with the NaN-not-0.0 guard (absent there entirely), **any calib split at all**
(`grep calib` returns nothing), `metrics/calibration/`, `moderation.py`, `corruptions/`,
`sweep.py`, `baselines.py`, and the shard-level manifest firewall. Tests: 794 here vs 22 there.

**Correction worth recording:** an early read of the file listing suggested this repo lacked
TRADES/TRADES-AWP. It does not — `TRADESTrainingModule` is in `adversarial_training.py`, selectable
as `training_pipe=trades`, and AWP is a mixin that composes with *every* arm rather than being
bolted to TRADES. This repo also has MART and an AT+KL hybrid, which the fork lacks. **There is
nothing to port on the adversarial-training side.**

**One thing to revisit if adopting `combined.py`:** it maps FakeClue's fakes to SID-Set's *fully
synthetic* class, pushing FF++ deepfakes and edited documents into the class the per-modality
breakout exists to separate them from.
