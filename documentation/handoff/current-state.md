---
type: handoff
title: Track A / Track B state, and how to resume
status: current
as_of: 2026-08-31
source: "run live on the 3090 box 2026-08-31"
tags: [experiments, track-a, track-b]
links: [ood-thresholds-come-from-in-domain-calib, track-a-and-track-b-are-separate-tables, clip-backbone-fixes-cross-dataset, 2026-08-31-track-b-first-results]
---

# Where we are

**Track A (robustness, ResNet-18 EV-AT ladder) is being run by colleagues**, not here. Seven
trained arms exist in `_runs/out/` with ~90 evaluation conditions. Its one real gap is that
`query_overconf`/`query_underconf` were never run on the five EV-AT arms — `jobs/premise_test.sh`
does exactly that and appears to have stopped after two arms.

**Track B (does a foundation backbone generalise?) is the active work here.** Built and merged
to `main` on 2026-08-31: a FakeClue datamodule with an identity firewall, a So-Fake-OOD
datamodule, `CLIPProbeClassifier`, `BinaryFoldClassifier`, and `jobs/track_b_chain.sh`.

## What changed

- `main` is at the merge of the Track B work; both the Mac and the box are in sync via
  `origin/main`. 794 tests, ruff clean, verified on both machines.
- `open_clip_torch` is installed in the box's `.venv` (torch stayed 2.5.1+cu124).
- Evaluation data is on the box: FakeClue (1.2 GB) and two So-Fake-OOD shards (5.5 GB).
- Results are backed up: `_runs/backups/out-20260831-1856.tar.gz` (1.7 GB, 1246 files,
  integrity-checked). `_runs/` is **untracked by git** — that archive is the only copy besides
  the live tree, and it sits on the same ZFS pool.

## Next steps

1. The chain was relaunched after the calib fix; the four completed steps skip via `.done`
   markers. Confirm the two So-Fake-OOD legs finished and read their numbers.
2. Read `detection_auroc_tampered` vs `detection_auroc_synthetic` — **not** mean accuracy. Only
   the So-Fake-OOD leg carries that breakout; FakeClue's binary fold destroys it.
3. The open research question has moved: since a better backbone did **not** fix cross-dataset
   performance ([[clip-backbone-fixes-cross-dataset]]), the next lever is multi-dataset training
   — which is where the colleague's instinct pointed. No combined datamodule exists yet.

## Blockers

None hard. Multi-dataset training needs a combined datamodule plus a decision on how FakeClue's
binary fakes map into the 3-class space; the colleague's `combined.py` folds them into
*synthetic*, which muddies the tampered/synthetic separation this project works to preserve.

## How to resume, from the box, without SSH

Claude Code 2.1.233 is installed at `~/.local/bin/claude` and authenticated. From
`~/Desktop/FILES/PROJECTS/trustfake/framework`:

```bash
git pull                      # pick up this brain and any fixes
~/.local/bin/claude           # interactive, with this documentation/ as context
```

Chain control:

```bash
LOGS=$(grep ^LOGS_PATH .env | cut -d= -f2)/track_b
cat $LOGS/chain.log           # step-by-step progress
ls $LOGS/*.done               # completed steps
bash jobs/track_b_chain.sh    # resume; completed steps skip
rm $LOGS/<step>.done          # force ONE step to re-run
```

`.done` markers make the chain resumable, but they also mean a step that *succeeded with a bad
number* will be skipped on rerun — delete its marker to redo it.
