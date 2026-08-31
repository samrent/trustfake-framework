---
type: decision
title: Track A and Track B are separate comparisons, never one table
status: current
as_of: 2026-08-31
source: "session 2026-08-31"
tags: [experiments, reporting]
links: [clip-backbone-fixes-cross-dataset]
---

**Decision.** Three distinct uses of CLIP exist in this repo and are named apart in the
docstrings, because "we use CLIP" would otherwise mean three things in one paper:

1. **Audit target** — `CLIPZeroShotClassifier`, seam 1 of the confidence-axis port. CLIP is the
   thing being attacked.
2. **Measurement instrument** — an ad-hoc probe used to get a non-degenerate model fast so Phi was
   measurable. Scaffolding, never a proposed detector.
3. **Detector backbone** — `CLIPProbeClassifier`, Track B.

Track A (robustness on the ResNet-18 EV-AT ladder) and Track B (generalisation with a foundation
backbone) are **separate tables**. A different backbone is a different model, so Track B starts a
new comparison rather than extending the seven trained arms and their ~90 evaluations.

**Why.** Swapping the backbone inside Track A would invalidate comparability with everything
already run, for no gain — the robustness results were already sufficient for the workshop.
