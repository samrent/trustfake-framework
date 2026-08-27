#!/usr/bin/env bash
# The decisive experiment: gradient-free confidence attack vs the finished
# arms. Eval-only (all checkpoints exist). Waits for the sweep so the two do
# not halve each other's GPU throughput.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
while pgrep -f "src/sweep.py --strategy" > /dev/null 2>&1; do sleep 60; done
LOG="$LOGS_PATH/premise"; mkdir -p "$LOG"
ARMS="e8_standard e8_pgd_at e8_trades e8_ev_only e8_ev_at e8_ev_at_awp"
for arm in $ARMS; do
  for cond in query_overconf query_underconf; do
    echo "[$(date -u +%H:%M:%S)] $arm $cond"
    .venv/bin/python src/test.py experiment.name=$arm \
      datamodule.datamodule.profile=train dataloaders.batch_size=32 \
      dataloaders.num_workers=6 datamodule.datamodule.limit_test=1000 \
      +attack=$cond > "$LOG/${arm}_${cond}.log" 2>&1 \
      && echo "  ok" || echo "  FAILED"
  done
done
echo "[$(date -u +%H:%M:%S)] premise test complete"
