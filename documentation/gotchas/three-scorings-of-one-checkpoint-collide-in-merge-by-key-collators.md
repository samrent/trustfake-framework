---
type: gotcha
title: Three scorings of one checkpoint log identical metric keys; merge-by-key collators overwrite them
status: current
as_of: 2026-09-03
source: "Track C design review, 2026-09-03"
tags: [evaluation, reporting, track-c]
links: [specs/track-c-depth-auxiliary]
---

**Trap.** Track C scores one checkpoint three ways (max-prob, depth consistency, combined) in
three `src/test.py` runs. All three log the SAME keys (`nat_fd_auroc`, `pgd_fd_auroc`, ...) into
different `test_lightning_logs/version_N` dirs under the same `output_dir`, and
`storage_<key>.pt` has no score axis. `sweep._latest_test_metrics` merges every `metrics.csv`
under an experiment by key, last write wins — so through `sweep.rank`, or any `rglob` collator,
the depth score silently becomes whichever scoring ran last, and the three columns of the table
read as one.

**Fix.** Track C is never ranked through `sweep.py`. `jobs/summarise_track_c.py` resolves each
cell from ITS OWN version dir (the path its log names), keys cells `<arm>__<dataset>__<cond>__<score>`,
and refuses a cell whose `experiment_config.yaml` records a different `uncertainty_score._target_`
than the key claims. `depth_calib_gate.json` in the same dir carries the G2 correlation.
