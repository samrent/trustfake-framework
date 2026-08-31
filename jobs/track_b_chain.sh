#!/usr/bin/env bash
# Track B: does a foundation-model backbone give a detector that GENERALISES?
#
# Track A (robustness on the ResNet-18 EV-AT ladder) is a separate table and
# is running elsewhere. Nothing here is comparable to it: a different backbone
# is a different model, so these numbers start a new comparison rather than
# extending the old one. Say so in any report that shows both.
#
# The question, in one line: Nicolas measured that SID-Set-trained models are
# bad off SID-Set. Is that a training-recipe problem or a REPRESENTATION
# problem? A frozen CLIP encoder was never fitted to SID-Set's artefacts, so
# if a linear probe on it generalises where the ResNet does not, the answer is
# representation -- and the fix is the backbone, not more epochs.
#
# The chain is ordered so each step is only run if the one it depends on
# produced something, and so the cheapest information arrives first.
#
#   1. train  clip_probe on SID-Set        (1.5k trainable params -- minutes)
#   2. eval   in-domain SID-Set test        <- is it even a working detector?
#   3. eval   FakeClue (cross-dataset)      <- the generalisation question
#   4. eval   So-Fake-OOD (shift)           <- Nicolas's stated target
#   5. eval   the ResNet baseline on 3+4    <- the head-to-head, no retraining
#
# THE NUMBER TO READ IS NOT MEAN ACCURACY. ViT-B/32 resizes to 224 and
# patchifies at 32 px, a heavy low-pass over exactly the local
# high-frequency evidence the tampered class lives in. The prediction is
# strong detection_auroc_synthetic and weak detection_auroc_tampered. A
# backbone that lifts the average while dropping tampered recall is a
# regression dressed as an improvement, so step 6 prints the per-modality
# rows rather than the average.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

TAG="${TAG:-track_b}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
PROFILE="${PROFILE:-train}"
LIMIT_TEST="${LIMIT_TEST:-1000}"
BASELINE_ARM="${BASELINE_ARM:-e8_standard}"

LOG="${LOGS_PATH}/${TAG}"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }

# Don't fight another job for the GPU. Concurrency is measurably NEGATIVE
# here -- two eval jobs together ran slower than one alone and aggregate
# throughput dropped 35%, because consumer cards time-slice CUDA contexts
# instead of overlapping them.
while pgrep -f "src/sweep.py --strategy" >/dev/null 2>&1; do sleep 60; done
while pgrep -f "jobs/premise_test.sh" >/dev/null 2>&1; do sleep 60; done

# The CLIP encoder carries its own preprocessing; the datamodule must not
# normalize on top or the input is normalized twice, silently.
CLIP_COMMON=(
  "model=clip_vit_b32_probe"
  "wrapper=base"
  "uncertainty_score=multiclass_max_probability"
  "datamodule.datamodule.normalization_layer=null"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
)

step() {  # step <name> <logfile> <command...>
  local name="$1"; shift
  local logfile="$1"; shift
  if [ -f "$LOG/${logfile}.done" ]; then say "SKIP  $name (already done)"; return 0; fi
  say "START $name"
  if "$@" >"$LOG/${logfile}.log" 2>&1; then
    touch "$LOG/${logfile}.done"; say "  ok  $name"; return 0
  fi
  say "  FAILED $name -- see $LOG/${logfile}.log"; return 1
}

# --- 1. train ---------------------------------------------------------------
step "train clip_probe on SID-Set" train \
  $PY src/train.py experiment.name="${TAG}_clip_probe" \
    "${CLIP_COMMON[@]}" \
    experiment.training_pipe=standard \
    datamodule.datamodule.profile="${PROFILE}" \
    trainer.trainer.max_epochs="${EPOCHS}" \
  || { say "training failed -- nothing downstream can run"; exit 1; }

# --- 2. in-domain -----------------------------------------------------------
step "eval clip_probe : SID-Set (in-domain)" eval_sidset \
  $PY src/test.py experiment.name="${TAG}_clip_probe" \
    "${CLIP_COMMON[@]}" \
    datamodule.datamodule.profile="${PROFILE}" \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

# --- 3. cross-dataset -------------------------------------------------------
# FakeClue is binary; the model is 3-class. BinaryFoldClassifier collapses
# p_fake = 1 - P(real). That fold merges synthetic and tampered -- unavoidable
# here, because FakeClue's ground truth does not distinguish them.
step "eval clip_probe : FakeClue (cross-dataset)" eval_fakeclue \
  $PY src/test.py experiment.name="${TAG}_clip_probe" \
    "${CLIP_COMMON[@]}" \
    datamodule=fake_clue \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

# --- 4. shift ---------------------------------------------------------------
# So-Fake-OOD keeps the 3-class labels, so the per-modality breakout SURVIVES
# here and this is the leg that can answer "does tampered degrade faster under
# shift than synthetic". Thresholds come from the in-domain calib split; the
# datamodule refuses to serve one of its own.
step "eval clip_probe : So-Fake-OOD (shift)" eval_ood \
  $PY src/test.py experiment.name="${TAG}_clip_probe" \
    "${CLIP_COMMON[@]}" \
    datamodule=so_fake_ood \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

# --- 5. the head-to-head ----------------------------------------------------
# No retraining: the ResNet arm already exists. Same two shifted conditions,
# same limit_test, so the comparison is a backbone comparison.
RESNET_COMMON=(
  "model=resnet18"
  "wrapper=base"
  "uncertainty_score=multiclass_max_probability"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
)
step "eval ${BASELINE_ARM} : FakeClue" baseline_fakeclue \
  $PY src/test.py experiment.name="${BASELINE_ARM}" \
    "${RESNET_COMMON[@]}" \
    datamodule=fake_clue \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

step "eval ${BASELINE_ARM} : So-Fake-OOD" baseline_ood \
  $PY src/test.py experiment.name="${BASELINE_ARM}" \
    "${RESNET_COMMON[@]}" \
    datamodule=so_fake_ood \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

say "chain complete -- logs in $LOG"
say "READ: detection_auroc_tampered vs detection_auroc_synthetic, NOT mean accuracy"
