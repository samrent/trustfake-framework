#!/usr/bin/env bash
# TB-E3: fetch SAGI-D (giakop/sagi-d, ~192 GB) once Kaggle credentials
# exist. Separate from fetch_rest.sh on purpose -- that script had a live
# instance when this was written, and bash reads scripts lazily.
#
# The kaggle CLI does not resume a broken download, so this retries whole
# (Kaggle's CDN is fast enough that a restart is cheaper than plumbing
# signed-URL ranges). Extraction skips mask/ trees; original/ trees stay:
# sagid.csv's src_path references them, so they are shipped rows, and the
# original-vs-edit pairs are the tampered training signal.
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a

LOG="${LOGS_PATH}/tb_e3/fetch"; mkdir -p "$LOG"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/fetch_rest.log"; }

DEST="${DATA_PATH}/sagi_d"
MARKER="$LOG/sagi_d.done"
[ -f "$MARKER" ] && { say "SKIP  sagi_d"; exit 0; }

say "START sagi_d"
t0=$SECONDS
tries=0
until .venv/bin/kaggle datasets download giakop/sagi-d -p "$DEST" >>"$LOG/sagi_d.log" 2>&1; do
  tries=$((tries + 1))
  [ $tries -ge 5 ] && { say "  FAILED sagi_d after $tries attempts -- see $LOG/sagi_d.log"; exit 1; }
  say "  retry $tries sagi_d"; sleep 60
done

say "  download ok, extracting (masks skipped)"
if unzip -qn "$DEST/sagi-d.zip" -x '*/mask/*' -d "$DEST" >>"$LOG/sagi_d.log" 2>&1; then
  touch "$MARKER"; say "  ok  sagi_d ($((SECONDS - t0))s)"
else
  say "  FAILED sagi_d extraction -- see $LOG/sagi_d.log"; exit 1
fi
