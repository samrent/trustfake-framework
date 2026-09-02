---
type: reference
title: Shortcut alignment — why SID-trained models "transfer" to So-Fake-OOD but not FakeClue
status: current
as_of: 2026-09-02
source: "headers-only audit of sid_set / so_fake_ood caches + fake_clue test.zip, 2026-09-02"
tags: [curation, g1, evaluation, so-fake-ood, fakeclue]
links: [separability-vocabulary, 2026-09-01-tb-e3-curation-ladder, rejected/clip-backbone-fixes-cross-dataset]
---

Measured 2026-09-02 (5-fold OOF headers-only AUROC, real-vs-fake): **sid_set 1.0000,
so_fake_ood 0.9997, fake_clue test 0.9666.** The per-class anatomy:

| dataset | class | square rate | JPEG share |
|---|---|---:|---:|
| SID-Set | real | 0.042 | 1.00 |
| SID-Set | synthetic | 1.000 | 0.00 |
| SID-Set | tampered | 1.000 | 0.86 |
| So-Fake-OOD | real | 0.057 | 0.94 |
| So-Fake-OOD | synthetic | 0.874 | 0.39 |
| So-Fake-OOD | tampered | 0.966 | 0.00 |
| FakeClue test | fake (label 0) | 0.735 | 0.84 |
| FakeClue test | real | 0.388 | 0.36 |

**The alignment observation.** SID's shortcut directions are square=>fake and PNG=>fake.
So-Fake-OOD carries the SAME directions (fakes square and PNG-heavy, reals non-square JPEG).
FakeClue partially carries the square direction but INVERTS the format direction (its fakes
are JPEG-heavy, its reals PNG-heavy). So a SID-trained model that absorbed the format/geometry
shortcuts — including through implicit conditioning (the 224 resize imprints aspect into
pixels) — would look like it "transfers" to So-Fake-OOD and fail or anti-transfer on
FakeClue, with no representation-quality difference at all.

**What this qualifies.** The rejected-leaf conclusion "a foundation backbone buys real
robustness to shift (So-Fake-OOD) and nothing on FakeClue" is now *consistent with* a second
mechanism: shortcut alignment between SID and So-Fake-OOD. It does NOT overturn TB-E3/E4's
shifted-leg results for the CURATED arms: an arm trained on nuisance-matched data (per-arm G1
~0.51) had no gradient toward the shortcut directions, so its L2 score cannot be riding
alignment it never learned. C0/C1's L2 numbers, and ANY SID-only-trained model's So-Fake-OOD
numbers (ours and the colleague fork's), carry the ceiling-on-evidence caveat.

**Standing instruction.** So-Fake-OOD is itself metadata-separable (0.9997): report every L2
result from a non-matched training arm with that caveat, and prefer matched-training arms (or
a geometry-matched L2 subset) for transfer claims.

## The directed shortcut-transfer matrix (2026-09-02, supersedes the cosine table)

Headers-only model TRAINED on row, detection AUROC on column (25k samples/dataset):

|  | sid | sfo | audits | sagi | tgif | imd | cf | fakeclue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sid_set | 1.000 | 0.864 | 0.796 | 0.357 | 0.407 | 0.282 | 0.773 | 0.805 |
| so_fake_ood | 0.995 | 1.000 | 0.752 | 0.557 | 0.399 | 0.978 | 0.731 | 0.799 |
| audits | 0.746 | 0.144 | 1.000 | 0.595 | 0.438 | 0.503 | 0.542 | 0.756 |
| sagi_d | 0.384 | 0.491 | 0.625 | 1.000 | 0.500 | 0.504 | 0.555 | 0.536 |
| tgif | 0.298 | 0.688 | 0.247 | 0.431 | 0.737 | 0.372 | 0.016 | 0.304 |
| imd2020 | 0.826 | 0.932 | 0.119 | 0.853 | 0.621 | 1.000 | 0.516 | 0.740 |
| cf_small | 0.991 | 0.949 | 0.814 | 0.503 | 0.383 | 0.501 | 0.995 | 0.748 |
| fake_clue | 0.997 | 0.862 | 0.923 | 0.362 | 0.383 | 0.042 | 0.911 | 0.972 |

**Corrections to the first-pass story.** (1) The linear-cosine table was a weak proxy
(sid<->sfo cosine -0.14 yet operational transfer 0.864/0.995): nonlinear shortcut models route
around component inversions, so alignment must be measured as TRANSFER, not correlation.
(2) The FakeClue-inversion prediction was wrong at the model level: SID-headers transfer to
FakeClue at 0.805 (squareness/resolution carry it; the inverted format bit is routed around).
True inversions live elsewhere: tgif->cf 0.016, fake_clue->imd 0.042, imd->audits 0.119,
audits->sfo 0.144 (below-chance cells = anti-aligned pairs, where pixel-model transfer numbers
are CONSERVATIVE).

**The sharpest new claim.** On both historical cross-dataset benchmarks the pure shortcut
model transfers BETTER than every pre-curation pixel model this project trained:
headers 0.864/0.805 (sfo/FakeClue) vs probe 0.727/0.520 and ResNet 0.502/0.488. The
imprinted shortcut travels worse than the explicit one and/or non-transferring content
drowned it — either way, pre-curation "transfer" numbers were bounded above by a two-feature
negative control and never reached it.

**Interpretation discipline for our own results.** Even Arm B's L2 0.786 sits below the L2
headers ceiling (0.864). This does NOT undermine it -- Arm B was trained nuisance-matched
(per-arm G1 ~0.51) and cannot be riding a direction it never learned -- but it means L2
alone cannot prove pixel-forensics superiority over metadata; matched training plus this
matrix are precisely the controls that let 0.786 be read as content-driven. Naive-reliance
caveat: correlating model scores with the shortcut model on a leg where the shortcut is
genuinely predictive conflates reliance with correctness; the proper reliance measure is
label-conditional (within-class) correlation -- open follow-up.
