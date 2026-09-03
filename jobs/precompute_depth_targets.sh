#!/usr/bin/env bash
# Track C, step 1: precompute the depth targets with the frozen teacher.
#
# Runs Depth Anything V2 (small) ONCE over the fit shards of a profile and
# stores each image's relative depth at the head's grid (112x112 for the
# 224 view, float16, zero-median / unit-MAD frame) under a store the
# datamodule reads back by manifest uid. Resumable per shard: re-running
# skips what exists and refuses to extend a store computed under other
# settings. The store is what `datamodule.datamodule.depth_targets_dir`
# points at for every Track C training run.
#
# One-time needs on the box: `uv pip install transformers` into .venv, and
# network for the ~99 MB safetensors fetch into ~/.cache/huggingface (set
# HF_HUB_OFFLINE=1 afterwards if the box is air-gapped). Time a first run
# with LIMIT_SHARDS=1 before committing the GPU for the rest; nothing here
# is a reported number, so the stub teacher (STUB=1) is for a plumbing
# check only -- never train a reported arm on it.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

TAG="${TAG:-track_c}"
PROFILE="${PROFILE:-train}"
DEPTH_DIR="${DEPTH_DIR:-${DATA_PATH}/sid_set_depth/dav2_small_518_224}"
TEACHER_INPUT="${TEACHER_INPUT:-518}"   # the checkpoint's own rule; recorded in the store
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH="${BATCH:-32}"
LIMIT_SHARDS="${LIMIT_SHARDS:-}"
STUB="${STUB:-0}"

LOG="${LOGS_PATH}/${TAG}_precompute"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/precompute.log"; }

# Anchored on the interpreter, never the bare path (a bare pattern matches
# any watcher that mentions the script, including itself).
WAIT_DEADLINE=$((SECONDS + 10800))
for other in "bash jobs/track_b_backbone_grid.sh" "bash jobs/track_b_convnext.sh" "src/sweep.py --strategy"; do
  while pgrep -f "$other" | grep -qv "^$$$"; do
    [ $SECONDS -gt $WAIT_DEADLINE ] && { say "gave up waiting on $other"; break; }
    say "waiting on: $other"; sleep 120
  done
done

step() {
  local key="$1"; shift
  if [ -f "$LOG/${key}.done" ]; then say "SKIP  $key"; return 0; fi
  say "START $key"
  local t0=$SECONDS
  if "$@" >"$LOG/${key}.log" 2>&1; then
    touch "$LOG/${key}.done"; say "  ok  $key ($((SECONDS - t0))s)"; return 0
  fi
  say "  FAILED $key -- see $LOG/${key}.log"; return 1
}

ARGS=(
  --profile "$PROFILE"
  --out-dir "$DEPTH_DIR"
  --image-size "$IMAGE_SIZE"
  --teacher-input-size "$TEACHER_INPUT"
  --batch "$BATCH"
)
[ -n "$LIMIT_SHARDS" ] && ARGS+=(--limit-shards "$LIMIT_SHARDS")
[ "$STUB" = "1" ] && ARGS+=(--stub-teacher)

# A limited run is a timing probe, not a completed step: it never writes the
# .done marker, so the full run still happens.
if [ -n "$LIMIT_SHARDS" ]; then
  say "timing probe: ${LIMIT_SHARDS} shard(s) -> $DEPTH_DIR"
  $PY src/precompute_depth.py "${ARGS[@]}" 2>&1 | tee -a "$LOG/probe.log"
else
  step "precompute__${PROFILE}" $PY src/precompute_depth.py "${ARGS[@]}"
fi

say "store: $DEPTH_DIR (manifest.json records teacher, revision, input size, grid, view)"
say "READ: the SUMMARY_JSON block in the step log -- rows written, shards skipped"
