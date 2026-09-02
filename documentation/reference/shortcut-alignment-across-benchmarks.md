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
