---
type: snapshot
title: Track A arm inventory and evaluation coverage
status: current
as_of: 2026-08-31
source: "read from _runs/out on the box"
tags: [track-a, results]
links: []
---

Seven arms, all ResNet-18 seed 1, all trained 2026-08-27, 12 epochs each (0–11), two checkpoints
apiece, 1.89 GB total.

| arm | val_accuracy | best epoch |
|---|---:|---|
| e8_ev_only | 0.8261 | 05 |
| e8_standard | 0.8206 | 04 |
| e8_ev_at_b0 | 0.7686 | 11 |
| e8_ev_at_awp | 0.7560 | 10 |
| e8_ev_at | 0.7499 | 07 |
| e8_ev_at_l2 | 0.7332 | 09 |
| e8_ev_at_kl | 0.6937 | 01 |

**Coverage gap.** The five EV-AT arms have 14 conditions each but **no `query_overconf` /
`query_underconf`** — the gradient-free confidence attack, and the only measurement the project's
own validity law trusts. `e8_ev_only` has both; `e8_standard` has *only* nat + the two query
attacks (3 conditions total). `jobs/premise_test.sh` does exactly these cells and appears to have
run for two arms then stopped.

**Findings visible in the existing data:**
- Gradient masking on the prediction axis, as documented: `e8_ev_only` leaves 0.5285 accuracy under
  PGD but **0.0040** under Square.
- The beta term is load-bearing: full `e8_ev_at` survives Square at 0.5897; `e8_ev_at_b0` collapses
  to 0.0707.
- The anchor pattern reproduces: `e8_ev_only` under `ace_uint8` holds accuracy at 0.8389 —
  identical to clean — while fd_auroc falls 0.8195 -> 0.1203.
- **Unexplained:** on the confidence axis the *gradient* attack is far stronger (ACE -> 0.1203)
  than the gradient-free one (query_overconf -> 0.7889), the reverse of the prediction axis. Worth
  understanding before writing the masking narrative.
- The min-norm column is near-vacuous on four arms: bb and cw identical to 4 decimals, within
  ~0.005 of clean accuracy — reporting "attack failed", not "model robust".
