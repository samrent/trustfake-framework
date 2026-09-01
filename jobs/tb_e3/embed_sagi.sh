#!/usr/bin/env bash
# TB-E3: embed SAGI-D once fetched+extracted; waits for the other embed
# chains (one GPU, sequential -- the gotcha).
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv/bin/python
export PYTHONPATH="$PWD/src"

LOG="${LOGS_PATH}/tb_e3/embed"; mkdir -p "$LOG"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/embed.log"; }

DEADLINE=$((SECONDS + 28800))
until [ -f "${LOGS_PATH}/tb_e3/fetch/sagi_d.done" ]; do
  [ $SECONDS -gt $DEADLINE ] && { say "gave up waiting on sagi_d fetch"; exit 1; }
  sleep 300
done
while pgrep -f "jobs/tb_e3/embed_heav[y].sh" > /dev/null || pgrep -f "jobs/tb_e3/embed_ligh[t].sh" > /dev/null; do
  say "embed_sagi waiting on GPU chains"; sleep 300
done

if [ -f "$LOG/sagi_d.done" ]; then say "SKIP  sagi_d"; exit 0; fi
say "START sagi_d"
t0=$SECONDS
if $PY src/curate.py embed --dataset sagi_d >"$LOG/sagi_d_embed.log" 2>&1; then
  touch "$LOG/sagi_d.done"; say "  ok  sagi_d ($((SECONDS - t0))s)"
else
  say "  FAILED sagi_d -- see $LOG/sagi_d_embed.log"
fi
