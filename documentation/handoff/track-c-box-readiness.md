---
type: handoff
title: Track C — the box is ready, the branch is not here
status: current
as_of: 2026-09-03
source: "checked live on the 3090 box 2026-09-03 from worktree claude/track-c-depth-auxiliary-944efa"
tags: [track-c, depth, experiments, blocker]
links: [handoff/current-state, gotchas/gpu-concurrency-is-negative-on-this-box, decisions/track-a-and-track-b-are-separate-tables]
---

# Track C — the box is ready, the branch is not here

**Blocker.** The Track C code (branch `claude/depth-auxiliary-robustness-track-b98707`, seven
commits on `faa69c1`, built on the Mac) exists neither on the box nor on `origin`. Checked
2026-09-03: `git ls-remote --heads origin` lists only `main`, `port/tier1`, and two `claude/*`
branches; `git log --all` on the box has no commit mentioning depth; no bundle or patch lives
under `~/Desktop` or `~/Downloads`. Every step of the Track C runbook after "install the deps"
needs `src/precompute_depth.py`, `jobs/track_c_depth.sh` and `documentation/specs/track-c-depth-auxiliary.md`, so nothing past
step 1 was run. **Push the branch from the Mac** (`git push -u origin claude/depth-auxiliary-robustness-track-b98707`),
then resume at step 2 below.

## What was done on the box (step 1, complete)

- `transformers==5.16.1` installed into the shared `.venv` with `uv pip install`; torch stayed
  `2.5.1+cu124`. Like `open_clip_torch`, it is declared in the branch's pyproject but not in the lock,
  so a `uv sync` would drop it again.
- The teacher `depth-anything/Depth-Anything-V2-Small-hf` is cached, snapshot
  `5426e4f0f36572d16453bbda7a8389317b1bef99`. The branch pins a revision; if the pin differs the
  first run fetches once more (~99 MB, 4 s here). The box is not air-gapped.
- Proven on the GPU with that install: the model loads through `AutoModelForDepthEstimation`,
  a 518×518 batch of two gives a finite `(2, 518, 518)` float32 map, and an fp32 backward through the
  teacher to the input yields a finite gradient. That is the online-teacher path the branch's
  `jobs/track_c_smoke_teacher_grad.py` checks; the framework-side half (the wrapper keeping
  `autocast=False`) still needs the branch.
- **The Hugging Face cache is not in `$HOME`.** `~/.cache/huggingface` is a symlink to
  `/scratch/models/huggingface`; nothing in `.env` or the shell rc sets `HF_HOME`. Pre-seeding
  into a literal `~/.cache/huggingface/hub` works only because of that link.
- Test suite (794) and ruff pass from a worktree wired per the usual links; `.venv` is *not*
  gitignored in a worktree, only `.env` is, so do not `git add -A` there.

## GPU state to plan around

Nothing from this project was training on 2026-09-03. TB-E3 reached `ATTACKS_COMPLETE` on
2026-09-01 21:25 and the backbone grid's last log is from 2026-09-01. Two unrelated resident
processes hold about 12.5 GB of the 24 GB (a llama.cpp server and the sotto app), leaving ~11 GB.
`pgd_at_depth` runs the ResNet plus the fp32 teacher online at 518 px with seven inner steps; if
the smoke OOMs, those two processes are the first thing to stop, not the batch size. Sequential
only, per [[gotchas/gpu-concurrency-is-negative-on-this-box]].

## Resume (from `~/Desktop/FILES/PROJECTS/trustfake/framework`, after the push)

```bash
git fetch && git checkout claude/depth-auxiliary-robustness-track-b98707   # or merge to main and pull
LIMIT_SHARDS=1 bash jobs/precompute_depth_targets.sh                       # timing probe: read img/s
setsid bash jobs/track_c_depth.sh                                          # smoke first; fix before the store
python3 jobs/summarise_track_c.py $LOGS_PATH/track_c_depth > RESULTS_track_c.md
```

Local `main` on the box was behind `origin/main` by the PR #5 merge on 2026-09-03; a plain
`git pull` fast-forwards it. The decisions in the seed (eps 8/255, own baselines, calib from
`sid_set`, white-box default, no `sweep.rank`) stand unchanged; the spec itself arrives with the branch.
