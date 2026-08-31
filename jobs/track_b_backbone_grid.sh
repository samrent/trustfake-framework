#!/usr/bin/env bash
# Exhaust the zero-training robust-encoder options, as a controlled grid.
#
# The question is whether an adversarially robustified encoder buys anything
# -- and if so, whether it buys it for Sournac's goal (a usable detector that
# generalises) or only for the confidence axis. The first FARE data point
# already showed a real cost in-domain: accuracy 0.8811 -> 0.7873, Phi 0.8284
# -> 0.7793, recall_tampered 0.8418 -> 0.7342. So robustification is not free,
# and it has to win somewhere else to be worth having.
#
# The grid separates the two mechanisms that were confounded in the
# frozen-vs-fitted analysis:
#
#              standard                      FARE4 (adv. fine-tuned)
#   B/32       clip_vit_b32_probe [done]     clip_vit_b32_fare_probe [running]
#   B/16       clip_vit_b16_probe            clip_vit_b16_fare_probe
#
# Both B/16 arms share the laion2B-s34B-b88K base; both B/32 arms share
# b79K. So each column is controlled, and the standard-B/16 arm is what stops
# a B/16 result being read as a robustness result. Plus TeCoA4 at B/32: a
# different robustification objective (supervised, against text prototypes)
# at the same architecture, to tell "adversarial fine-tuning does this" from
# "FARE does this".
#
# Cheapest arm first, so an overnight failure costs the least.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

TAG="${TAG:-track_b}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
PROFILE="${PROFILE:-train}"
EPOCHS="${EPOCHS:-8}"
LIMIT_TEST="${LIMIT_TEST:-1000}"

LOG="${LOGS_PATH}/${TAG}_grid"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/grid.log"; }

# Anchored on the interpreter, never the bare path -- a bare pattern also
# matches any watcher that merely mentions the script, including one that
# matches itself and can therefore never exit.
WAIT_DEADLINE=$((SECONDS + 10800))
for other in "bash jobs/track_b_fare_and_crop.sh" "bash jobs/track_b_matrix.sh" "src/sweep.py --strategy"; do
  while pgrep -f "$other" | grep -qv "^$$$"; do
    [ $SECONDS -gt $WAIT_DEADLINE ] && { say "gave up waiting on $other"; break; }
    say "waiting on: $other"; sleep 120
  done
done

COMMON=(
  "wrapper=base"
  "uncertainty_score=multiclass_max_probability"
  "datamodule.datamodule.normalization_layer=null"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
)

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

arm() {  # arm <model-config>
  local model="$1"
  local exp="${TAG}_${model}"

  step "${model}__train" \
    $PY src/train.py experiment.name="$exp" model="$model" "${COMMON[@]}" \
      experiment.training_pipe=standard \
      datamodule.datamodule.profile="${PROFILE}" \
      trainer.trainer.max_epochs="${EPOCHS}" \
    || { say "  training failed for $model -- skipping its evaluations"; return 0; }

  # In-domain clean, then the confidence attack that actually bites
  # (query_underconf took the undefended probe's Phi to 0.4837, chance),
  # then the shift condition with in-domain thresholds.
  step "${model}__sid_set__clean" \
    $PY src/test.py experiment.name="$exp" model="$model" "${COMMON[@]}" \
      datamodule.datamodule.profile="${PROFILE}" \
      datamodule.datamodule.limit_test="${LIMIT_TEST}"

  step "${model}__sid_set__query_underconf" \
    $PY src/test.py experiment.name="$exp" model="$model" "${COMMON[@]}" \
      datamodule.datamodule.profile="${PROFILE}" \
      datamodule.datamodule.limit_test="${LIMIT_TEST}" \
      +attack=query_underconf

  step "${model}__so_fake_ood__clean" \
    $PY src/test.py experiment.name="$exp" model="$model" "${COMMON[@]}" \
      datamodule=so_fake_ood calib_datamodule=sid_set \
      datamodule.datamodule.limit_test="${LIMIT_TEST}"
}

# Cheapest first. B/16 has 4x the tokens of B/32, so its query cells are the
# expensive ones and they run last.
arm clip_vit_b32_tecoa_probe
arm clip_vit_b16_probe
arm clip_vit_b16_fare_probe

say "grid complete"
say "READ 1: does robustification shrink the Phi collapse (0.8284 -> 0.4837 undefended)?"
say "READ 2: does B/16 recover recall_tampered, and does it need FARE to do it?"
say "READ 3: do TeCoA and FARE behave alike -- method, or adversarial training in general?"
