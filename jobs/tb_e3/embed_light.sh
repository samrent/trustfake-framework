#!/usr/bin/env bash
# TB-E3 runbook step 3, last leg: embed the extracted file-tree datasets.
# Waits for embed_heavy (one GPU, sequential -- the gotcha) and for each
# dataset's extraction marker; skips a dataset whose marker never appears
# rather than failing the whole chain (record it, don't retry blindly).
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv/bin/python
export PYTHONPATH="$PWD/src"

LOG="${LOGS_PATH}/tb_e3/embed"; mkdir -p "$LOG"
FETCH="${LOGS_PATH}/tb_e3/fetch"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/embed.log"; }

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

while pgrep -f "jobs/tb_e3/embed_heav[y].sh" > /dev/null; do
  say "waiting on embed_heavy.sh"; sleep 300
done

wait_extract() {
  local marker="$FETCH/extract_$1.done" deadline=$((SECONDS + 21600))
  until [ -f "$marker" ]; do
    [ $SECONDS -gt $deadline ] && { say "gave up waiting on extract_$1"; return 1; }
    say "waiting on extract_$1"; sleep 300
  done
}

wait_extract synthbuster  && step synthbuster $PY src/curate.py embed --dataset synthbuster
wait_extract imd2020      && step imd2020     $PY src/curate.py embed --dataset imd2020

# VISION has no extraction step -- wait on its fetch marker directly.
DEADLINE=$((SECONDS + 21600))
until [ -f "$FETCH/vision_images.done" ]; do
  [ $SECONDS -gt $DEADLINE ] && { say "gave up waiting on vision fetch"; break; }
  say "waiting on vision fetch"; sleep 300
done
[ -f "$FETCH/vision_images.done" ] && step vision $PY src/curate.py embed --dataset vision

wait_extract tgif_tars    && wait_extract tgif2_random && \
  step tgif $PY src/curate.py embed --dataset tgif

say "embed_light complete"
