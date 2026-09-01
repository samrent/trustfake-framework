#!/usr/bin/env bash
# TB-E3 runbook step 3 (local half): embed SID-Set and So-Fake-OOD once with
# the frozen standard ViT-B/16. Shard-level resume lives in embed.py; the
# .done marker per dataset follows the chain convention. Sequential on the
# GPU (gotchas/gpu-concurrency-is-negative-on-this-box.md).
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv/bin/python
export PYTHONPATH="$PWD/src"

LOG="${LOGS_PATH}/tb_e3/embed"; mkdir -p "$LOG"
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

step so_fake_ood $PY src/curate.py embed --dataset so_fake_ood
step sid_set     $PY src/curate.py embed --dataset sid_set

say "embed_local complete"
