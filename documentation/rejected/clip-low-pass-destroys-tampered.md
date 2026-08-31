---
type: rejected
title: "ViT-B/32's patch-32 low-pass would collapse tampered recall"
status: superseded
as_of: 2026-08-31
source: "measured in-domain 2026-08-31; partially reinstated by the OOD leg the same day"
tags: [track-b, tampered]
links: [2026-08-31-track-b-first-results]
---

**Claim.** ViT-B/32 resizes to 224 and patchifies at 32 px — a heavy low-pass over exactly the
local high-frequency evidence a tampered edit leaves, the destruction `input_mode: crop` exists to
prevent. So a CLIP backbone should be strong on *synthetic* and weak on *tampered*.

**What killed it.** Measured in-domain on SID-Set:

- `detection_auroc_synthetic` = 0.9971, `recall_synthetic` = 0.9972 — near-perfect, as predicted
- `detection_auroc_tampered` = **0.8962**, `recall_tampered` = 0.8418 — no collapse

Tampered is the weaker class, but it holds up well. The directional intuition was right; the
magnitude was wrong, and the prediction of a collapse was wrong.

**Why it's worth keeping.** The reasoning was plausible enough that it shaped the config and the
chain's closing advice ("read the per-modality rows, not the average"). That advice is still
right — it is *how* the prediction got falsified. ViT-B/16 remains the fallback if a finer patch
grid is ever wanted, but there is no measured reason to reach for it yet.


## Reinstated in part, 2026-08-31 (So-Fake-OOD)

The rejection above holds **in-domain only**. Under distribution shift the CLIP probe's
`recall_tampered` collapses to **0.0503** — it finds essentially no tampered images at all — while
`recall_synthetic` stays at 0.7928 and `detection_auroc_synthetic` at 0.8586.

So the original reasoning was right about the *mechanism* and wrong about *where it shows up*: a
patch-32 low-pass does cost the local high-frequency evidence tampering leaves, but in-domain the
probe compensates (0.8962 AUROC, 0.8418 recall) and only under shift does the loss become
catastrophic. Read this note together with
[[snapshots/2026-08-31-track-b-first-results]]; ViT-B/16 now has a measured reason to be tried.
