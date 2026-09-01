#!/usr/bin/env bash
# Is the ViT patch projection what destroys the tampered class under shift?
#
# The open question after the backbone grid: OOD recall_tampered sits at
# 0.0476-0.0503 for both standard ViTs and moves to 0.24-0.29 for the
# robustified ones -- but OOD detection_auroc_tampered is 0.576-0.593
# everywhere, so nothing is DISCRIMINATING tampered better. Finer patches
# (B/16) did not help. That leaves the input path itself as the suspect:
# ViT-B/32 resizes to 224 and patchifies at 32 px, a low-pass over exactly
# the local high-frequency evidence a tampered edit leaves.
#
# FARE4-convnext_base_w is the test. It is the only NON-ViT robust CLIP
# encoder chs20 published, so it is adversarially robustified like the FARE
# ViTs but has NO patch projection at all.
#
#   tampered HOLDS here  -> the patch grid is the mechanism
#   tampered COLLAPSES   -> it is not, and the lever is elsewhere
#     (a crop-TRAINED arm, or unfreezing the encoder)
#
# Either answer closes a question the grid left open, which is why this runs
# before any partial-unfreeze machinery gets built.
#
# ---------------------------------------------------------------------------
# TWO PROTOCOL DIFFERENCES, state them in any table this appears in:
#
#   1. image_size 256, not 224 (this encoder's native resolution). That is a
#      different input protocol, the same way input_mode=crop is -- so this
#      arm is a DIAGNOSTIC, not a table-mate for the ViT arms.
#   2. embed_dim 640, not 512. clip_probe reads visual.output_dim so the head
#      sizes itself; nothing to configure, but do not assume 512 anywhere.
#
# CALIB FIX, new here: configs/training/calib_datamodule/sid_set.yaml carries
# its own ImageNet normalization_layer, and no previous job script nulled it.
# In-domain cells were unaffected (calib falls back to the eval datamodule,
# which IS nulled), but every So-Fake-OOD cell run so far fitted temperature
# and the moderation thresholds on DOUBLE-NORMALIZED calib data -- CLIP
# normalizes internally, see gotchas/clip-normalization-lives-in-the-model.
# Argmax quantities (accuracy, recall_*) are unaffected; anything downstream
# of temperature (fd_auroc, detection_auroc, every moderation indicator) is
# suspect in those cells. This script nulls it on both sides, and the calib
# split is served at 256 too, so calibration and test share one protocol.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

TAG="${TAG:-track_b}"
MODEL="${MODEL:-clip_convnext_fare_probe}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
PROFILE="${PROFILE:-train}"
EPOCHS="${EPOCHS:-8}"
LIMIT_TEST="${LIMIT_TEST:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

LOG="${LOGS_PATH}/${TAG}_convnext"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/convnext.log"; }

# Concurrency on this box is measurably NEGATIVE (see
# gotchas/gpu-concurrency-is-negative-on-this-box), so queue behind anything
# already holding the GPU. Patterns are anchored on the interpreter, never a
# bare script path -- a bare pattern also matches a watcher that merely
# mentions the script, including one that matches itself and so never exits.
WAIT_DEADLINE=$((SECONDS + 21600))
for other in "bash jobs/premise_test.sh" "bash jobs/track_b_backbone_grid.sh" "src/sweep.py --strategy"; do
  while pgrep -f "$other" | grep -qv "^$$$"; do
    [ $SECONDS -gt $WAIT_DEADLINE ] && { say "gave up waiting on $other"; break; }
    say "waiting on: $other"; sleep 120
  done
done

# normalization_layer=null: the CLIP encoder normalizes internally and an
# ImageNet Normalize on top double-normalizes with no error and no shape
# change. image_size=256 is this encoder's native resolution.
COMMON=(
  "model=${MODEL}"
  "wrapper=base"
  "uncertainty_score=multiclass_max_probability"
  "datamodule.datamodule.normalization_layer=null"
  "datamodule.datamodule.image_size=${IMAGE_SIZE}"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
)

# Only for steps that pass calib_datamodule=sid_set. In-domain steps leave
# calib_datamodule=none, where calib falls back to the eval datamodule and is
# already covered by COMMON; overriding a group set to `none` would error.
CALIB_FIX=(
  "calib_datamodule=sid_set"
  "calib_datamodule.datamodule.normalization_layer=null"
  "calib_datamodule.datamodule.image_size=${IMAGE_SIZE}"
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

EXP="${TAG}_${MODEL}"

step "${MODEL}__train" \
  $PY src/train.py experiment.name="$EXP" "${COMMON[@]}" \
    experiment.training_pipe=standard \
    datamodule.datamodule.profile="${PROFILE}" \
    trainer.trainer.max_epochs="${EPOCHS}" \
  || { say "training failed -- skipping evaluations"; exit 1; }

step "${MODEL}__sid_set__clean" \
  $PY src/test.py experiment.name="$EXP" "${COMMON[@]}" \
    datamodule.datamodule.profile="${PROFILE}" \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

step "${MODEL}__sid_set__query_underconf" \
  $PY src/test.py experiment.name="$EXP" "${COMMON[@]}" \
    datamodule.datamodule.profile="${PROFILE}" \
    datamodule.datamodule.limit_test="${LIMIT_TEST}" \
    +attack=query_underconf

# THE step this script exists for.
step "${MODEL}__so_fake_ood__clean" \
  $PY src/test.py experiment.name="$EXP" "${COMMON[@]}" \
    datamodule=so_fake_ood "${CALIB_FIX[@]}" \
    datamodule.datamodule.limit_test="${LIMIT_TEST}"

say "convnext arm complete"
say "READ 1: OOD recall_tampered AND detection_auroc_tampered vs 0.0476-0.0503 / 0.576-0.593"
say "        at B/32 and B/16. AUROC is the one that says 'discrimination'; recall alone"
say "        is an operating point and can be bought with a prior shift."
say "READ 2: in-domain accuracy sits under a 256 protocol -- not comparable to the ViT arms."
say "READ 3: this arm's OOD cell is the FIRST with correctly-normalized calib. If its"
say "        fd_auroc/detection_auroc look off-trend, suspect the fix, not the encoder."
