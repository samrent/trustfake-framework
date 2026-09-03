---
type: handoff
title: Track A / B / C state, and how to resume
status: current
as_of: 2026-09-03
source: "Track B state from the box 2026-08-31/09-01; Track C built on the Mac 2026-09-03, not run"
tags: [experiments, track-a, track-b, track-c]
links: [ood-thresholds-come-from-in-domain-calib, track-a-and-track-b-are-separate-tables, depth-head-lives-in-the-model-group, specs/track-c-depth-auxiliary, 2026-09-01-track-b-full-matrix, 2026-09-01-backbone-grid]
---

# Where we are

**Track A (robustness, ResNet-18 EV-AT ladder) is being run by colleagues**, not here. Seven
trained arms exist in `_runs/out/` with ~90 evaluation conditions; the `query_*` coverage gap
on the five EV-AT arms is still open (`jobs/premise_test.sh`).

**Track B (does a foundation backbone generalise?)** is collated: both a CLIP probe and the
trained ResNet sit at chance cross-dataset (`rejected/clip-backbone-fixes-cross-dataset`), FARE
stabilises Φ under the query attack at a clean-accuracy cost, and patch size does not recover OOD
tampered recall (`snapshots/2026-09-01-*`). TB-E3, the curation ladder, is registered and
waiting (`specs/tb-e3-curation-ladder.md`).

**Track C (does monocular depth add value?) is BUILT, TESTED ON THE MAC, and NOT RUN.** Branch
`claude/depth-auxiliary-robustness-track-b98707`, commits in reviewable steps; 922 tests and
ruff green. The spec with pre-registered decision rules is `specs/track-c-depth-auxiliary.md`;
the executable runbook is `jobs/track_c_depth.sh`. Design decisions in
`decisions/depth-head-lives-in-the-model-group.md`.

## What Track C consists of

- `trustfake.losses.depth` — the shared zero-median/unit-MAD frame and the SSI-L1 loss.
- `trustfake.models.torch.DepthHead`; `ResNet(depth_head=True)` with `forward_with_depth`;
  `configs/training/model/resnet18_depth.yaml`. `ResNet.forward` is bit-identical to before.
- `trustfake.depth` — the frozen Depth Anything V2 small teacher (lazy `transformers` import,
  its own preprocessing, pinned revision) and a stub for tests.
- `trustfake.data.depth_targets` + `src/precompute_depth.py` — per-shard float16 store keyed by
  manifest uid with a settings manifest; `datamodule.datamodule.depth_targets_dir` opt-in.
- `trustfake.pipes.train.depth_auxiliary` — `standard_depth`, `pgd_at_depth`, `trades_depth`;
  `depth_lambda`; head supervised on x_adv against the clean target; AWP ascends the joint loss.
- `trustfake.metrics.uncertainty.depth` + `trustfake.models.wrapper.DepthConsistencyWrapper`
  (`wrapper=depth`) — `depth_consistency` and `depth_combined` scores; `depth_attack_scoring`
  (`white_box` | `transfer`); `src/test.py` fits the combined reference and writes the σ-seam
  gate `depth_calib_gate.json` on the in-domain calib split.
- `jobs/precompute_depth_targets.sh`, `jobs/track_c_depth.sh`, `jobs/summarise_track_c.py`.

## Next steps (on the box)

1. `git pull`; `uv pip install transformers` into `.venv` (declared in pyproject, not in the
   lock, like open_clip); make sure the box can fetch the ~99 MB teacher once, or pre-seed
   `~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf`.
2. `LIMIT_SHARDS=1 bash jobs/precompute_depth_targets.sh` — read img/s from the probe; the
   full `train` profile is 30 shards.
3. `setsid bash jobs/track_c_depth.sh` — its first step is a two-batch GPU smoke of
   `pgd_at_depth` and one depth-score evaluation: the two CUDA-only failure modes
   (`gotchas/deterministic-mode-throws-on-median-and-bilinear-backward`) were fixed blind on
   the Mac and this is where they get proven. If the smoke fails, fix before the store is built.
4. Collate with `python3 jobs/summarise_track_c.py $LOGS_PATH/track_c_depth`; write the
   snapshot; answer H(a)–H(d) from the spec; update this leaf.

## The box, as checked live on 2026-09-03 (from the other session's readiness note)

- Branch state: `claude/track-c-depth-auxiliary` is on `origin` (PR #6 into `main`). The
  earlier note that "the depth branch was never pushed" (branch
  `claude/track-c-depth-auxiliary-944efa`, one doc commit) is superseded by this leaf.
- `transformers==5.16.1` is installed in the shared `.venv` with `uv pip install`; torch stayed
  `2.5.1+cu124`. It is declared in pyproject but not in the lock, so a `uv sync` drops it again.
- The teacher `depth-anything/Depth-Anything-V2-Small-hf` is cached at the pinned snapshot
  `5426e4f0f36572d16453bbda7a8389317b1bef99`; loading through `AutoModelForDepthEstimation`, a
  518x518 forward and an fp32 backward to the input were all proven finite on the GPU.
- **The HF cache is not in `$HOME`:** `~/.cache/huggingface` is a symlink to
  `/scratch/models/huggingface`; nothing sets `HF_HOME`. Pre-seeding works only via that link.
- GPU: nothing from this project was training; two unrelated resident processes (a llama.cpp
  server and the sotto app) hold ~12.5 GB of 24 GB, leaving ~11 GB. `pgd_at_depth` runs the
  ResNet plus the fp32 teacher at 518 px with seven inner steps -- if the smoke OOMs, those two
  processes are the first thing to stop, not the batch size. Sequential only.
- In a box worktree `.venv` is not gitignored (only `.env` is): never `git add -A` there.
- Local `main` on the box is behind `origin/main`; a plain `git pull` fast-forwards it.

## Blockers

None hard. Unverified until the smoke runs: the CUDA determinism fixes on the framework side
(the teacher path alone was proven), the teacher's throughput at 518, and memory with the
resident processes up.

## How to resume, from the box, without SSH

Claude Code is installed at `~/.local/bin/claude` and authenticated. From
`~/Desktop/FILES/PROJECTS/trustfake/framework`:

```bash
git pull                      # pick up this brain and any fixes
~/.local/bin/claude           # interactive, with this documentation/ as context
```

Chain control (`.done` markers make every chain resumable; a step that succeeded with a bad
number is skipped on rerun -- delete its marker to redo it):

```bash
LOGS=$(grep ^LOGS_PATH .env | cut -d= -f2)/track_c_depth
cat $LOGS/chain.log; ls $LOGS/*.done
bash jobs/track_c_depth.sh
```
