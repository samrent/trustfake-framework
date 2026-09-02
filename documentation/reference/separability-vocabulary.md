---
type: reference
title: How to phrase separability findings — the vocabulary that makes them land
status: current
as_of: 2026-09-02
source: "Samuel's phrasing discipline, 2026-09-02, folded into the brain verbatim in spirit"
tags: [reporting, curation, g1]
links: [2026-09-01-tb-e3-curation-ladder, fakeclue-dataset-properties]
---

Every G1-type finding is read by someone whose numbers it undermines. The framing below is the
difference between a finding that lands and one that gets rebutted.

**The terms.** *Dataset separability / confounded benchmark* — the measured property: "AUDITS is
metadata-separable at 0.9998 AUROC", never "the model cheats"; the dataset was measured, not the
model. *Shortcut / spurious correlation* — the mechanism a model could exploit. *Unfalsifiable
result* — the epistemic consequence and the strongest correct claim: on such a benchmark a
genuine forgery detector and a shortcut detector produce indistinguishable scores, so the number
cannot confirm forgery detection. *Negative control* — what the headers-only classifier is: a
model that provably sees no forensic evidence sets the floor; a pixel model scoring at or below
it is explainable without any tampering signal. *Implicit conditioning* — the resize pathway:
metadata never enters the model as a feature but is imprinted into the pixel distribution (a
fixed 224x224 resize squashes real 4:3 photos anisotropically; square fakes pass untouched).

**Two framing rules.**
1. State it as a **ceiling on evidence, not an accusation**: "the result is consistent with
   forgery detection but equally consistent with reading the aspect ratio." That sentence is
   exactly what 0.98 separability means, so it cannot be rebutted.
2. Name the fix as the **claim's missing control, not extra work**: "the number becomes
   trustworthy only under a metadata-matched evaluation" — and point at the one clean subset
   where the score already IS falsifiable (canonical exemplar: the ff++ slice of FakeClue,
   where headers score ~0.50). You are not saying everything is worthless; you are saying here
   is the one place the benchmark is clean, and everywhere else needs the same property before
   its number means anything.

**One-liner template** (meeting version): "A classifier that never decodes a pixel — width and
height only — gets 0.98 on SID-Set. Every fake is a 1024x1024 square; 96% of real photos are
not. So 'is the image square' *is* the label."

**Careful template** (report version): "We do not claim the trained model uses metadata — its
inputs are pixels only. We claim something weaker but sufficient: the benchmark cannot detect
whether it does."
