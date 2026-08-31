---
type: gotcha
title: Apply the binary fold after load_from_checkpoint, not before
status: current
as_of: 2026-08-31
source: "found while wiring jobs/track_b_chain.sh"
tags: [models, evaluation, fakeclue]
links: [fakeclue-label-convention-is-inverted]
---

**Trap.** `BinaryFoldClassifier` wraps a 3-class model to score it on a binary benchmark. Wrapping
the module *before* `ClassificationEvaluationModule.load_from_checkpoint` prefixes every state_dict
key with `inner.`, so the checkpoint fails to match.

**Fix.** `src/test.py` applies the fold **after** the load, by replacing
`eval_module.model.model`. A test pins that post-load wrapping preserves the loaded weights.

**Related:** the fold was implemented before it was *wired*. For a while the chain would have
scored a 3-class model against binary labels with no collapse at all — no error, just wrong
numbers. It is now driven by `+binary_fold=true`, and refuses to run unless the datamodule reports
exactly 2 classes, so the two cannot silently disagree.
