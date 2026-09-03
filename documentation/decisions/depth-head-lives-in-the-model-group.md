---
type: decision
title: The depth head is an opt-in kwarg of the ResNet, reached only by a separate forward path
status: current
as_of: 2026-09-03
source: "Track C design, 2026-09-03; alternatives measured against the checkpoint and attack contracts"
tags: [track-c, models, checkpoints, evaluation]
links: [specs/track-c-depth-auxiliary, track-a-and-track-b-are-separate-tables]
---

**Decision.** `ResNet(depth_head=True)` builds a `DepthHead` submodule under the flat key
`depth_head.*`; `model=resnet18_depth` is a separate model config with its own `name` (so its own
output tree); `ResNet.forward` is untouched and bit-identical, and the head is reachable only via
`forward_with_depth`, exposed on `TrustFakeWrapper` with normalisation applied. At evaluation
the head is dropped by construction: nothing that calls `forward` can see it.

**Why.** Three contracts pinned it. (1) `src/test.py` rebuilds the wrapper from the `model` group
and loads the checkpoint strictly, so the eval-time module tree must equal the training-time one
key for key; a checkpoint from `resnet18_depth` refuses to load into `resnet18` and vice versa,
loudly. (2) The optimiser is bound to the model-group module's parameters before wrapping, so a
head owned by the pipe or the wrapper would never train. (3) Every attack reads `model(x)[0]`;
a second forward path costs the attacks nothing.

**Three scorings of one checkpoint are three evaluation runs**, differing only in `wrapper` and
`uncertainty_score`: `base`+`multiclass_max_probability`, `depth`+`depth_consistency`,
`depth`+`depth_combined`. The `depth` wrapper with a probability score IS the base wrapper op
for op; with a depth-aware score it owns the frozen teacher (kept out of the module tree, so out
of the checkpoint) and returns `None` from `outputs_from_logits` so the pipe re-forwards the
perturbed batch and the teacher sees the attacked pixels.

**What the attack sees is an explicit setting** (`depth_attack_scoring`): `white_box` (the score
itself, teacher gradient included) or `transfer` (`1 − MSP` during the attack). Neither is
silent; the run's `experiment_config.yaml` records it and the collator keys cells by it.

**Alternatives rejected.** *Composition (`backbone.*` prefix)* — renames every key, ImageNet
loading needs remapping, and it was falsely believed to load into `resnet18`. *Strip head keys
on load (`strict=False`)* — also swallows a genuinely mismatched checkpoint. *A score class that
receives the input* — every existing score takes probabilities only, and the calib gate,
temperature fit and `test_step` all read `model(x)[3]`; a wrapper threads the score through all
three for free. *Grey-box as the default* — a rejection score whose robustness is only ever
measured against an attack that does not see it would read as robust while trivially attackable;
the honest default is white-box, transfer is the named cheap protocol.
