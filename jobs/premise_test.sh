#!/usr/bin/env bash
# The decisive EV-AT experiment: gradient-free confidence attack against the
# EV-AT ladder, with `standard` as the positive control the attack must be
# able to damage. Eval-only; waits for the training sweep so the two do not
# halve each other's GPU throughput.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
while pgrep -f "src/sweep.py --strategy" > /dev/null 2>&1; do sleep 60; done
LOG="$LOGS_PATH/premise"; mkdir -p "$LOG"
ARMS="e8_standard e8_ev_only e8_ev_at_b0 e8_ev_at e8_ev_at_awp e8_ev_at_kl e8_ev_at_l2"
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
echo "[$(date -u +%H:%M:%S)] EV-AT premise test complete"
