---
type: gotcha
title: CLIP models normalize internally; the datamodule must not
status: current
as_of: 2026-08-31
source: "designed into trustfake.models.torch.clip, pinned by test"
tags: [models, clip]
links: [freezing-a-backbone-must-not-use-no-grad]
---

**Trap.** Everywhere else in this framework the datamodule owns normalization and the wrapper
applies it before the model. A CLIP encoder is **not** free to be normalized any way you like — it
was trained under its own mean/std. The datamodule's ImageNet `Normalize` on top of a CLIP model
raises nothing, returns the right shapes, and returns different numbers. The L_inf ball is
unaffected either way (attacks still perturb raw `[0,1]` pixels), which is precisely why nothing
downstream can detect it.

**Fix.** Both CLIP classes apply their own preprocessing inside the module, from buffers that
travel with the checkpoint. **Set `datamodule.datamodule.normalization_layer=null`** with any CLIP
model; the chain script and both model configs say so.

**Second CLIP trap:** `logit_scale`. The shipped scale (~100) can saturate `1 - MSP` into a
constant, which makes failure-detection AUROC rank nothing while accuracy is bit-identical
(temperature is monotone). Measured as input-dependent: 6.85e-05 on unambiguous inputs but
3.32e-02 (max 0.27) on ambiguous ones, and on real CIFAR-10 the native scale barely mattered
(Phi 0.9142 vs 0.9126 calibrated). Check it against the real distribution; don't assume it bites.
