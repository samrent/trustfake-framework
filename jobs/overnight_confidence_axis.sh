#!/usr/bin/env bash
# Overnight run: does a classical label-axis defence already fix the
# CONFIDENCE axis?
#
# That is the one question whose answer changes what this project is. Ledda
# et al. 2025 -- the proposal's own reference [7] -- reports that adversarial
# training gives some uncertainty robustness. If pgd_at/trades already restore
# failure-AUROC under ACE, then at_conf/conf_reg have no room and the
# contribution is the measurement harness, not a new defence. If they do not,
# the confidence-axis arms are the contribution. If nothing restores it, that
# is the strongest negative result and the most publishable framing, because
# it is the question the field has not answered.
#
# Five arms, matched on optimiser steps (same max_epochs, same schedule, same
# backbone, same data protocol) -- the proposal requires that, and without it
# "our method wins" can just mean "our method got more training".
#
# Deliberately NOT here:
#   mart, at_kl        -- SWEEP-STRATEGIES.md Strategy B, which that document
#                         recommends skipping; implemented, but not what the
#                         night should buy.
#   evidential_adversarial -- the PIs' own method. Benchmark it once these
#                         harness numbers are trusted, not before.
#   autoattack, square -- a run-once-at-the-end overnight job in their own
#                         right (BUILD.md:213). Queue separately.
#
# Forensic epsilon (2/255) with warm-up: an 8/255 ball erases the
# small-amplitude high-frequency evidence a deepfake detector reads and
# collapses training onto a constant output, silently.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

PROFILE="${PROFILE:-train}"
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
ADV_EPS="${ADV_EPS:-0.00784}"      # 2/255
ADV_STEPS="${ADV_STEPS:-7}"
WARMUP="${WARMUP:-4}"
TAG="${TAG:-night}"

LOGDIR="${LOGS_PATH}/${TAG}"
mkdir -p "$LOGDIR"

# Every arm is selected on the SAME clean metric and additionally logs
# val_robust_accuracy. Selecting each arm on "what it optimises" is
# defensible but makes the table a comparison of selection protocols; one
# shared metric plus both numbers recorded keeps that confound out of the
# first pass.
COMMON=(
  "datamodule.datamodule.profile=${PROFILE}"
  "trainer.trainer.max_epochs=${EPOCHS}"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
  "adv_eps=${ADV_EPS}"
  "adv_steps=${ADV_STEPS}"
  "adv_warmup_epochs=${WARMUP}"
  "robust_val_steps=3"
  "selection_metric=val_f1_score"
  "selection_mode=max"
)

ARMS=(standard pgd_at trades at_conf conf_reg)
# Conditions: the headline confidence attack in its realisable form, the
# label-free confidence attack, the label-axis control, and one corruption
# for spec compliance ("compression, resizing, re-encoding" is named in the
# proposal). Clean is implicit in every run.
CONDITIONS=("" "+attack=ace_uint8" "+attack=overconf" "+attack=pgd" "+corruption=jpeg")

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say()   { echo "[$(stamp)] $*" | tee -a "$LOGDIR/driver.log"; }

# CALIBRATE=1 times one epoch of the cheapest and the most expensive arm and
# projects the whole grid, so EPOCHS is chosen from a measurement instead of
# an guess. The budget is the constraint here, not the dataset: there is no
# prize for touching every shard, and an overnight run that does not finish
# produces nothing at all.
if [ "${CALIBRATE:-0}" = "1" ]; then
  say "=== calibration | profile=$PROFILE ==="
  declare -A per_epoch
  for arm in standard pgd_at; do
    t0=$(date +%s)
    $PY src/train.py experiment.name="cal_${arm}" experiment.training_pipe="$arm" \
        "${COMMON[@]}" trainer.trainer.max_epochs=1 \
        > "$LOGDIR/cal_${arm}.log" 2>&1
    t1=$(date +%s); per_epoch[$arm]=$((t1 - t0))
    say "calibration: $arm 1 epoch = ${per_epoch[$arm]}s"
  done
  s=${per_epoch[standard]}; a=${per_epoch[pgd_at]}
  # standard + conf_reg are cheap; pgd_at + trades + at_conf carry the attack.
  grid=$(( 2 * s + 3 * a ))
  say "calibration: one epoch across all five arms = ${grid}s"
  budget_s=$(( ${BUDGET_HOURS:-5} * 3600 ))
  fit=$(( budget_s / (grid > 0 ? grid : 1) ))
  say "calibration: ${BUDGET_HOURS:-5}h of training fits ~${fit} epochs -> rerun with EPOCHS=${fit}"
  exit 0
fi

say "=== overnight run start | profile=$PROFILE epochs=$EPOCHS eps=$ADV_EPS steps=$ADV_STEPS ==="

for arm in "${ARMS[@]}"; do
  name="${TAG}_${arm}"
  ckpt_glob="${OUTPUT_PATH}/${name}"
  if compgen -G "${ckpt_glob}/**/*.ckpt" > /dev/null 2>&1; then
    say "TRAIN $arm -- checkpoint exists, skipping"
    continue
  fi
  say "TRAIN $arm -- start"
  if $PY src/train.py experiment.name="$name" experiment.training_pipe="$arm" \
        "${COMMON[@]}" > "$LOGDIR/train_${arm}.log" 2>&1; then
    say "TRAIN $arm -- done"
  else
    say "TRAIN $arm -- FAILED (see $LOGDIR/train_${arm}.log)"
  fi
done

for arm in "${ARMS[@]}"; do
  name="${TAG}_${arm}"
  for cond in "${CONDITIONS[@]}"; do
    label="${cond:-clean}"; label="${label//[+=]/_}"
    say "EVAL $arm $label -- start"
    if [ -z "$cond" ]; then
      $PY src/test.py experiment.name="$name" \
          "datamodule.datamodule.profile=${PROFILE}" \
          "dataloaders.batch_size=${BATCH}" "dataloaders.num_workers=${WORKERS}" \
          > "$LOGDIR/eval_${arm}_${label}.log" 2>&1
    else
      $PY src/test.py experiment.name="$name" \
          "datamodule.datamodule.profile=${PROFILE}" \
          "dataloaders.batch_size=${BATCH}" "dataloaders.num_workers=${WORKERS}" \
          "$cond" > "$LOGDIR/eval_${arm}_${label}.log" 2>&1
    fi
    [ $? -eq 0 ] && say "EVAL $arm $label -- done" \
                 || say "EVAL $arm $label -- FAILED"
  done
done

say "=== overnight run complete ==="
