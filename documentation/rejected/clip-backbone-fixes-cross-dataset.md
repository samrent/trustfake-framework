---
type: rejected
title: "A foundation backbone would fix cross-dataset generalisation"
status: current
as_of: 2026-08-31
source: "measured on the box, jobs/track_b_chain.sh, 2026-08-31"
tags: [track-b, generalisation]
links: [2026-08-31-track-b-first-results, track-a-and-track-b-are-separate-tables]
---

**Claim.** SID-Set-trained models are bad off SID-Set because the *representation* was fitted to
one dataset's artefacts. A frozen CLIP encoder was never fitted to them (Ojha et al., CVPR 2023),
so a linear probe on it should generalise where the ResNet does not.

**What killed it.** On FakeClue, zero-shot cross-dataset:

| | CLIP probe | e8_standard ResNet |
|---|---:|---:|
| accuracy | 0.5207 | 0.5148 |
| fd_auroc | 0.5054 | 0.4313 |
| detection_auroc | 0.5198 | 0.4880 |

**Both are at chance** — and below FakeClue's 0.638 majority floor and its 0.690 "square => fake"
metadata rule. Swapping the backbone changed nothing cross-dataset.

**What survives.** In-domain the probe is genuinely better (accuracy 0.8811 vs 0.8249, aurc 0.0316
vs 0.0511) with **1,539 trainable parameters**. So the backbone is a good in-domain lever and not a
generalisation lever.

**Caveat on an earlier number.** A CLIP probe scored 0.879 on FakeClue in an earlier session — but
that probe was *fitted on FakeClue rows* and tested on held-out FakeClue. It was never comparable
to a SID-Set-trained model evaluated zero-shot. The 0.52 above is the fair comparison.

## Qualified the same day by the So-Fake-OOD leg

On FakeClue the rejection stands: both models at chance. But on So-Fake-OOD the backbone **does**
matter — the probe holds `detection_auroc` 0.7266 and `fd_auroc` 0.6665 where the ResNet sits at
0.5023 and 0.4703 (the latter *below* chance, i.e. anti-correlated uncertainty).

So the honest claim is narrower than either the original hypothesis or its first rejection: **a
foundation backbone buys real robustness to distribution shift, and buys nothing on FakeClue.**
Which of those generalises is unresolved with one shift dataset and one cross-dataset benchmark,
and FakeClue's own metadata floor (0.690) makes it a harsh test that may say more about the
benchmark than the model.

**Where that points:** multi-dataset training remains the untested lever, and it is now the
interesting one — the two models fail on *opposite* classes under shift, so combining their
training data is a directed hypothesis rather than a shot in the dark.
