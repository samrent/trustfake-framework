#!/usr/bin/env bash
# TB-E4: pack pixels then fine-tune, once assemble+probe lands.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
export PYTHONPATH="$PWD/src"
LOG="${LOGS_PATH}/tb_e4"; mkdir -p "$LOG"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }

DEADLINE=$((SECONDS + 28800))
until [ -f "$LOG/assemble_probe.done" ]; do
  [ $SECONDS -gt $DEADLINE ] && { say "gave up waiting on assemble_probe"; exit 1; }
  sleep 180
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

step pack_pixels .venv/bin/python jobs/tb_e4/pack_pixels.py
step armB .venv/bin/python jobs/tb_e4/finetune_b.py
say "stage5 complete"
