#!/usr/bin/env bash
# TB-E4: embed the pair cache, then assemble + Arm A, once staging lands.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
export PYTHONPATH="$PWD/src"
LOG="${LOGS_PATH}/tb_e4"; mkdir -p "$LOG"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }

DEADLINE=$((SECONDS + 21600))
until [ -f "${DATA_PATH}/tb_e4_qf85/pairs_manifest.parquet" ]; do
  [ $SECONDS -gt $DEADLINE ] && { say "gave up waiting on pairs manifest"; exit 1; }
  sleep 120
done

step() {
  local key="$1"; shift
  [ -f "$LOG/${key}.done" ] && { say "SKIP $key"; return 0; }
  say "START $key"
  if "$@" >"$LOG/${key}.log" 2>&1; then
    touch "$LOG/${key}.done"; say "  ok $key"; return 0
  fi
  say "  FAILED $key -- see $LOG/${key}.log"; return 1
}

step embed_pairs .venv/bin/python jobs/tb_e4/embed_pairs.py
step assemble_probe .venv/bin/python jobs/tb_e4/assemble_and_probe.py
say "stage23 complete"
