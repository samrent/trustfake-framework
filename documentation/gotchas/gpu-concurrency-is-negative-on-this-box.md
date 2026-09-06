---
type: gotcha
title: Running evaluations concurrently on the 3090 is slower than sequential
status: current
as_of: 2026-08-31
source: "benchmarked on the box 2026-08-31"
tags: [infrastructure, gpu]
links: []
---

**Trap.** The obvious way to use idle GPU is to run several evaluation jobs at once. Measured, it
is worse:

| concurrent jobs | wall | aggregate throughput |
|---|---:|---:|
| 1 | 5.5 s | 3467 img·q/s |
| 2 | 11.7 s | 2256 |
| 3 | 16.5 s | 2224 |
| 4 | 21.5 s | 2187 |

Wall time scales **linearly** and aggregate throughput drops 35% — two jobs together are slower
than one alone. Consumer GeForce cards have no MPS, so concurrent CUDA contexts time-slice rather
than overlap, and each process costs 600–1000 MiB of context (N=6 OOM'd).

**Fix.** Run sequentially. `jobs/track_b_chain.sh` does, and waits on `sweep.py` /
`premise_test.sh` so it will not contend with another run on the same GPU.

**Also measured, so nobody re-optimises the wrong thing:** batch size buys almost nothing (raw
model 1.24x from batch 16 to 256), and a ResNet-18 eval peaks at ~439 MiB — so the ~18 GB of local
LLM servers resident on this card are irrelevant to it. Evaluation is time-bound, not memory-bound,
and the cost is dominated by query budgets (Square/AutoAttack at 5000 queries).
