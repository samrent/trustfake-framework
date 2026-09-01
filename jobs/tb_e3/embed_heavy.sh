#!/usr/bin/env bash
# TB-E3 runbook step 3, heavy half: embed the fetched training environments.
# Sequential on the GPU, one .done marker per dataset, shard-level resume
# inside embed.py. The audits step waits for its zip extraction to finish.
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

# Wait on any still-running embed chain: two embedders would contend for
# the one GPU (gotchas/gpu-concurrency-is-negative-on-this-box.md).
while pgrep -f "jobs/tb_e3/embed_local[.]sh" > /dev/null; do
  say "waiting on embed_local.sh"; sleep 60
done

step community_forensics_small $PY src/curate.py embed --dataset community_forensics_small

DEADLINE=$((SECONDS + 7200))
until grep -q EXTRACT_OK "${DATA_PATH}/audits/extract.log" 2>/dev/null; do
  [ $SECONDS -gt $DEADLINE ] && { say "gave up waiting on audits extraction"; exit 1; }
  say "waiting on audits extraction"; sleep 120
done
step audits $PY src/curate.py embed --dataset audits

say "embed_heavy complete"
