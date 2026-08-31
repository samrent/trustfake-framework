---
type: gotcha
title: Freeze a backbone with requires_grad_(False), never no_grad
status: current
as_of: 2026-08-31
source: "caught while writing CLIPProbeClassifier"
tags: [models, attacks]
links: [track-a-and-track-b-are-separate-tables]
---

**Trap.** Wrapping a frozen backbone's forward in `torch.no_grad()` looks like a free speed-up for
a linear probe — the encoder's weights never update anyway. But `no_grad` also cuts the gradient
with respect to the **input**, and every gradient attack in this repo differentiates the logits
w.r.t. the image. A probe built that way reports **perfect PGD robustness while being trivially
attackable** — the exact failure mode the harness exists to catch.

**Fix.** `requires_grad_(False)` on the parameters. It stops parameter gradients while leaving
input gradients intact. `tests/models/test_clip_probe.py` pins both: input gradients flow, and PGD
actually moves the model.
