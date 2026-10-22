---
type: decision
title: Track C white-box query cells run on L1 only, depth score only, and at their own eval batch
status: current
as_of: 2026-09-04
source: "PI decision 2026-09-03 during the first run, after the first white-box query cell measured 8,229 s"
tags: [track-c, protocol, cost, depth]
links: [specs/track-c-depth-auxiliary, snapshots/2026-09-04-track-c-first-results, gotchas/gpu-concurrency-is-negative-on-this-box]
---

# Track C white-box query cells: L1 only, depth score only

**Decision.** `jobs/track_c_depth.sh` schedules `depth_wb` for `query_underconf` and
`query_overconf` on SID-Set only, at the full `limit_test=1000`. `combined_wb` and every
So-Fake-OOD `_wb` cell are dropped. `WB_DATASETS` / `WB_SCORES` restore the full matrix.

**Why.** A white-box query cell is 400 queries × (1000/8) batches, each a Depth Anything V2
forward at 518 px: 8,229 s measured on the 3090. The pre-registered 16 cells were ~37 GPU-hours
for numbers the decision rules do not read — H(d) is defined on `depth_wb` vs `depth_tr` on the
query attacks, and the combined score is read second by design; L2 sits at chance for every
arm, so a white-box number there would not be readable anyway. H(d) is therefore an in-domain
number and the snapshot says so.

**Companion, not a protocol change.** Evaluation runs at `EVAL_BATCH=8`, separate from the
recipe's training batch 32: a white-box gradient through the fp32 teacher costs ~0.85 GB per
image, so batch 32 (~27 GB) does not fit a 24 GB card at all, and the no-grad calib pass alone
(~9 GB) OOMed beside the box's resident processes. Every attack in the chain reduces per-sample
(sum, sign, per-sample ε-halving, per-sample query acceptance), so the eval batch moves no
reported number.
