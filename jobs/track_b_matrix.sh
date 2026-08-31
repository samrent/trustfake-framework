#!/usr/bin/env bash
# Track B, the FULL array: does a foundation backbone hold up on BOTH axes,
# in-domain and under shift?
#
# jobs/track_b_chain.sh ran only the clean column -- which says a CLIP probe
# beats a ResNet in-domain and holds signal under shift, but says nothing at
# all about robustness. This is the rest of the array.
#
#   models      x  datasets              x  conditions
#   ---------      --------                 ----------
#   clip_probe     sid_set   (in-domain)    clean            (done by the chain)
#   e8_standard    fake_clue (cross)        pgd              (prediction axis)
#                  so_fake_ood (shift)      query_overconf   (confidence axis)
#                                           query_underconf  (confidence axis)
#
# 2 x 3 x 4 = 24 cells, 5 of them already done. Every cell is resumable via its
# own .done marker, and the order is cheap-first so a crash costs the least.
#
# WHY THESE CONDITIONS. `pgd` moves the prediction axis; `query_overconf` and
# `query_underconf` are gradient-free and move the CONFIDENCE axis while the
# argmax is held fixed, which is the only measurement this project's validity
# law trusts (a defence that survives gradients and folds to a query attack was
# masking gradients). Reporting them under one "robustness" heading is exactly
# what the attack taxonomy in attacks/abc.py exists to prevent.
#
# Square/AutoAttack are deliberately NOT here: 5000 queries each is a run of
# its own, and the cells that answer the question are the 400-query ones.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

TAG="${TAG:-track_b}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
PROFILE="${PROFILE:-train}"
LIMIT_TEST="${LIMIT_TEST:-1000}"
CLIP_EXP="${CLIP_EXP:-${TAG}_clip_probe}"
RESNET_EXP="${RESNET_EXP:-e8_standard}"

LOG="${LOGS_PATH}/${TAG}_matrix"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/matrix.log"; }

# Sequential only. Concurrency is measurably NEGATIVE on this card: two jobs
# ran slower than one and aggregate throughput fell 35%, because consumer
# GeForce has no MPS and CUDA contexts time-slice instead of overlapping.
for other in "src/sweep.py --strategy" "jobs/premise_test.sh" "jobs/track_b_chain.sh"; do
  while pgrep -f "$other" >/dev/null 2>&1; do say "waiting on: $other"; sleep 60; done
done

cell() {  # cell <model> <dataset> <condition>
  local model="$1" dataset="$2" cond="$3"
  local name="${model}/${dataset}/${cond}"
  local key="${model}__${dataset}__${cond}"
  if [ -f "$LOG/${key}.done" ]; then say "SKIP  $name"; return 0; fi

  local -a args=(
    "wrapper=base"
    "uncertainty_score=multiclass_max_probability"
    "dataloaders.batch_size=${BATCH}"
    "dataloaders.num_workers=${WORKERS}"
  )
  case "$model" in
    clip_probe) args+=("experiment.name=${CLIP_EXP}" "model=clip_vit_b32_probe"
                       # the encoder normalizes internally; a datamodule
                       # Normalize on top double-normalizes silently
                       "datamodule.datamodule.normalization_layer=null") ;;
    resnet)     args+=("experiment.name=${RESNET_EXP}" "model=resnet18") ;;
    *) say "unknown model $model"; return 1 ;;
  esac
  case "$dataset" in
    sid_set)     args+=("datamodule.datamodule.profile=${PROFILE}") ;;
    # FakeClue is binary; the models are 3-class. The fold merges synthetic
    # and tampered -- unavoidable, its ground truth does not separate them.
    fake_clue)   args+=("datamodule=fake_clue" "+binary_fold=true") ;;
    # Thresholds come from IN-DOMAIN calib, never from the shifted data.
    so_fake_ood) args+=("datamodule=so_fake_ood" "calib_datamodule=sid_set") ;;
    *) say "unknown dataset $dataset"; return 1 ;;
  esac
  args+=("datamodule.datamodule.limit_test=${LIMIT_TEST}")
  [ "$cond" != "clean" ] && args+=("+attack=${cond}")

  say "START $name"
  local t0=$SECONDS
  if $PY src/test.py "${args[@]}" >"$LOG/${key}.log" 2>&1; then
    touch "$LOG/${key}.done"; say "  ok  $name  ($((SECONDS - t0))s)"
  else
    say "  FAILED $name -- see $LOG/${key}.log"
  fi
}

# Cheap first: prediction-axis attacks are ~10 gradient steps, the query
# attacks are 400 forward passes each.
for cond in clean pgd query_overconf query_underconf; do
  for model in clip_probe resnet; do
    for dataset in sid_set fake_clue so_fake_ood; do
      cell "$model" "$dataset" "$cond"
    done
  done
done

say "matrix complete"
say "READ: accuracy AND fd_auroc per cell. A confidence attack that leaves"
say "accuracy bit-identical while fd_auroc falls is the whole point -- an"
say "accuracy-only table cannot see it."
