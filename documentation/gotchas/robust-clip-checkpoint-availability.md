---
type: reference
title: Which robust CLIP encoders actually exist, and at which epsilon
status: current
as_of: 2026-09-01
source: "read from chs20/* open_clip_config.json on the Hub"
tags: [clip, robustness, track-b]
links: [2026-09-01-track-b-full-matrix]
---

The adversarially robustified CLIP encoders (Schlarmann et al., Robust CLIP)
are published at these architectures, and the epsilon knob is NOT uniformly
available:

| checkpoint | arch | layers | width | patch | img | eps |
|---|---|---:|---:|---:|---:|---|
| FARE4-ViT-B-32-laion2B-s34B-b79K | ViT-B/32 | 12 | 768 | 32 | 224 | 4/255 |
| FARE4-ViT-B-16-laion2B-s34B-b88K | ViT-B/16 | 12 | 768 | 16 | 224 | 4/255 |
| TeCoA4-ViT-B-32 / -B-16 | as above | | | | | 4/255 |
| FARE4-convnext_base_w | ConvNeXt | - | - | none | **256** | 4/255 |
| fare2-clip / tecoa2-clip | **ViT-L/14** | 24 | 1024 | 14 | 224 | 2/255 |
| fare4-clip / tecoa4-clip | **ViT-L/14** | 24 | 1024 | 14 | 224 | 4/255 |

**The consequence that matters.** FARE4 at B/32 costs 0.094 clean accuracy
(0.8811 -> 0.7873). The obvious way to recover some of that is a weaker
adversarial budget -- eps=2 instead of eps=4 -- and **that variant does not
exist at B/32 or B/16**. It exists only at ViT-L/14, which is a different
architecture (2x the depth), so taking it means starting a new column rather
than extending the grid, and its query-attack cells are far more expensive.

So the accuracy/robustness tradeoff cannot be tuned off the shelf at the size
currently being run. The routes to a cheaper tradeoff are: move to L/14 and
accept a new column, or adversarially fine-tune an encoder ourselves at a
chosen epsilon, which is the only genuinely "custom" option.

`FARE4-convnext_base_w` is worth running for a different reason: it has no
patch projection, so it tests whether the patchified low-pass is really the
mechanism behind the tampered collapse. Note it expects image_size 256, which
is a protocol difference and belongs in the table caption.
