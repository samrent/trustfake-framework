#!/usr/bin/env bash
# Two experiments, both aimed at a specific term in the Track B analysis.
#
# A. THE ENCODER SWAP (P2). FARE4 is the same ViT-B/32 architecture and the
#    same LAION-2B pretraining as the probe already measured, adversarially
#    fine-tuned at eps=4/255. Same head, same data, same protocol -- so any
#    change in the confidence axis is attributable to the encoder alone.
#
#    Undefended probe, in-domain: accuracy 0.8811 -> 0.8811 (bit-identical)
#    while Phi 0.8284 -> 0.4837 under query_underconf, a collapse to chance
#    and 4.5x the ResNet's. If FARE shrinks that collapse, the confidence axis
#    is repairable AT THE ENCODER with no head retraining. If it does not, the
#    confidence axis is not reducible to input-space robustness -- which is a
#    sharper claim than the literature makes, and equally publishable.
#
# B. THE CROP DIAGNOSTIC. The tampered collapse under shift (recall 0.0503)
#    was attributed to ViT-B/32's patch-32 projection. But the resize to 224
#    happens FIRST, and this project measured that as a ~4.6x low-pass on
#    1024px sources. `input_mode: crop` removes the resize and keeps the patch
#    grid, so it separates the two: if tampered recovers, the resize dominates
#    and ViT-B/16 -- which removes the rank bound but NOT the resize -- buys
#    much less than the frozen-vs-fitted analysis implies. Evaluation only, no
#    retraining: it re-scores the existing checkpoint under a different input
#    pipeline, which is a protocol change and must be reported as one.
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
FARE_EXP="${FARE_EXP:-${TAG}_fare_probe}"
CLIP_EXP="${CLIP_EXP:-${TAG}_clip_probe}"

LOG="${LOGS_PATH}/${TAG}_fare"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/run.log"; }

# Anchored on the interpreter so a watcher mentioning the path cannot match.
WAIT_DEADLINE=$((SECONDS + 1800))
for other in "src/sweep.py --strategy" "bash jobs/track_b_matrix.sh" "bash jobs/premise_test.sh"; do
  while pgrep -f "$other" | grep -qv "^$$$"; do
    [ $SECONDS -gt $WAIT_DEADLINE ] && { say "gave up waiting on $other"; break; }
    say "waiting on: $other"; sleep 60
  done
done

COMMON=(
  "wrapper=base"
  "uncertainty_score=multiclass_max_probability"
  "datamodule.datamodule.normalization_layer=null"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
)

step() {  # step <key> <command...>
  local key="$1"; shift
  if [ -f "$LOG/${key}.done" ]; then say "SKIP  $key"; return 0; fi
  say "START $key"
  local t0=$SECONDS
  if "$@" >"$LOG/${key}.log" 2>&1; then
    touch "$LOG/${key}.done"; say "  ok  $key ($((SECONDS - t0))s)"
  else
    say "  FAILED $key -- see $LOG/${key}.log"; return 1
  fi
}

# ---------------------------------------------------------------- A. FARE ---
step fare_train \
  $PY src/train.py experiment.name="${FARE_EXP}" model=clip_vit_b32_fare_probe \
    "${COMMON[@]}" experiment.training_pipe=standard \
    datamodule.datamodule.profile="${PROFILE}" \
    trainer.trainer.max_epochs="${EPOCHS}" \
  || { say "FARE training failed; skipping its evaluations"; }

if [ -f "$LOG/fare_train.done" ]; then
  for cond in clean query_underconf query_overconf pgd; do
    extra=(); [ "$cond" != "clean" ] && extra=("+attack=${cond}")
    step "fare_sid_set_${cond}" \
      $PY src/test.py experiment.name="${FARE_EXP}" model=clip_vit_b32_fare_probe \
        "${COMMON[@]}" datamodule.datamodule.profile="${PROFILE}" \
        datamodule.datamodule.limit_test="${LIMIT_TEST}" "${extra[@]}"
  done
  for cond in clean query_underconf; do
    extra=(); [ "$cond" != "clean" ] && extra=("+attack=${cond}")
    step "fare_so_fake_ood_${cond}" \
      $PY src/test.py experiment.name="${FARE_EXP}" model=clip_vit_b32_fare_probe \
        "${COMMON[@]}" datamodule=so_fake_ood calib_datamodule=sid_set \
        datamodule.datamodule.limit_test="${LIMIT_TEST}" "${extra[@]}"
  done
fi

# ---------------------------------------------------------- B. crop diag ---
# The existing (standard-backbone) checkpoint, re-scored with the resize
# removed. A protocol change, so these numbers sit in their own column and
# never beside resize-mode numbers without saying so.
step crop_sid_set_clean \
  $PY src/test.py experiment.name="${CLIP_EXP}" model=clip_vit_b32_probe \
    "${COMMON[@]}" datamodule.datamodule.profile="${PROFILE}" \
    datamodule.datamodule.input_mode=crop \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

step crop_so_fake_ood_clean \
  $PY src/test.py experiment.name="${CLIP_EXP}" model=clip_vit_b32_probe \
    "${COMMON[@]}" datamodule=so_fake_ood calib_datamodule=sid_set \
    datamodule.datamodule.input_mode=crop \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

say "done"
say "READ (A): Phi clean -> query_underconf for FARE vs 0.8284 -> 0.4837 undefended"
say "READ (B): recall_tampered under crop vs 0.8418 in-domain / 0.0503 shifted"
