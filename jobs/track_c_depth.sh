#!/usr/bin/env bash
# Track C: does monocular depth add value to synthetic-image detection?
#
# Two questions on two axes, using the repo's own machinery:
#   (b) Mao et al. (ECCV 2020): an auxiliary depth head trained alongside
#       classification -- the attacker must fool two objectives through one
#       backbone -- should buy a modest robust-accuracy bump. Head dropped at
#       inference; zero added attack surface.
#   (c) The residual between the model's depth head and a frozen teacher run
#       on the same input, as a REJECTION SCORE: does it detect attacked or
#       misclassified inputs better than 1 - max prob?
# Plus (a) clean accuracy. Matrix: {standard, pgd_at} x {baseline, +depth},
# scored three ways: max prob, depth consistency, combined. TRADES is an
# optional second adversarial arm (WITH_TRADES=1).
#
# Self-contained: the baselines are trained HERE, under the same settings as
# the depth arms, so the comparison never depends on how the e8_* arms were
# fitted. Track C is its own table -- never merged with Track A or B.
#
# Policy constants (recorded decisions): eps = 8/255 (adv_eps=0.03137) with
# 7 inner steps and a 2-epoch warm-up, as the Track A protocol; thresholds
# and the combined score's reference come from the IN-DOMAIN calib split
# (calib_datamodule=sid_set on the shifted set); FakeClue excluded.
#
# What an attack sees when the score is depth-aware is explicit per cell:
#   *_tr  transfer  -- the attack optimises 1 - max prob, the depth score is
#                      computed on the final perturbed batch ("an attack
#                      crafted against the classifier, scored by depth"; the
#                      standard-attack protocol, the cheap one).
#   *_wb  white_box -- the attack optimises the depth score itself, teacher
#                      included (the adaptive number; query_* attacks make
#                      it a gradient-free adaptive attack for free).
# Prediction-axis attacks (pgd, corruptions) never read the score, so for
# them transfer == white_box at a fraction of the cost and only `_tr` runs.
# CAVEAT for every depth-score number: a gradient adaptive attack whose loss
# also minimises the residual is a follow-up, not this chain.
#
# Steps are resumable (.done markers). Cheapest first; a training failure
# short-circuits its evaluations. Sequential only (the GPU-concurrency gotcha).
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

TAG="${TAG:-track_c}"
PROFILE="${PROFILE:-train}"
EPOCHS="${EPOCHS:-12}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-6}"
LIMIT_TEST="${LIMIT_TEST:-1000}"
LAMBDAS="${LAMBDAS:-1.0}"            # e.g. "0.1 0.3 1.0" for the sweep
WITH_TRADES="${WITH_TRADES:-0}"
CONDS="${CONDS:-clean pgd query_underconf query_overconf ace_uint8}"
CORRUPTIONS="${CORRUPTIONS:-jpeg}"
DATASETS="${DATASETS:-sid_set so_fake_ood}"
DEPTH_DIR="${DEPTH_DIR:-${DATA_PATH}/sid_set_depth/dav2_small_518_224}"
TEACHER_INPUT="${TEACHER_INPUT:-518}"  # MUST equal the store's input_size
ADV_EPS="${ADV_EPS:-0.03137}"          # 8/255, pinned
ADV_STEPS="${ADV_STEPS:-7}"
WARMUP="${WARMUP:-2}"

LOG="${LOGS_PATH}/${TAG}_depth"; mkdir -p "$LOG"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }

WAIT_DEADLINE=$((SECONDS + 21600))
for other in "bash jobs/track_b_backbone_grid.sh" "bash jobs/track_b_convnext.sh" "bash jobs/track_b_matrix.sh" "src/sweep.py --strategy"; do
  while pgrep -f "$other" | grep -qv "^$$$"; do
    [ $SECONDS -gt $WAIT_DEADLINE ] && { say "gave up waiting on $other"; break; }
    say "waiting on: $other"; sleep 120
  done
done

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

TRAIN_COMMON=(
  "datamodule.datamodule.profile=${PROFILE}"
  "trainer.trainer.max_epochs=${EPOCHS}"
  "dataloaders.batch_size=${BATCH}"
  "dataloaders.num_workers=${WORKERS}"
  "robust_val_steps=3"
  "selection_metric=val_f1_score"
  "selection_mode=max"
)
ADV=("adv_eps=${ADV_EPS}" "adv_steps=${ADV_STEPS}" "adv_warmup_epochs=${WARMUP}")
DEPTH=("model=resnet18_depth" "datamodule.datamodule.depth_targets_dir=${DEPTH_DIR}")

# ---------------------------------------------------------------------------
# 0. Smoke: the CUDA-only failure modes (deterministic-mode ops, bf16, the
#    teacher gradient) BEFORE any GPU time is spent on the real store: two
#    training batches, one full evaluation on the smoke profile (capped test
#    split; the calib pass and the white-box FGSM through the teacher run in
#    full), and the teacher-gradient check. Real teacher, tiny store.
# ---------------------------------------------------------------------------
SMOKE_DIR="${DATA_PATH}/sid_set_depth/smoke_${TEACHER_INPUT}_224"
step "smoke__precompute" \
  $PY src/precompute_depth.py --profile smoke --out-dir "$SMOKE_DIR" \
    --teacher-input-size "$TEACHER_INPUT" --batch "$BATCH"
step "smoke__pgd_at_depth" \
  $PY src/train.py experiment.name="${TAG}_smoke" experiment.training_pipe=pgd_at_depth \
    model=resnet18_depth "datamodule.datamodule.depth_targets_dir=${SMOKE_DIR}" \
    datamodule.datamodule.profile=smoke trainer.trainer.max_epochs=1 \
    +trainer.trainer.limit_train_batches=2 +trainer.trainer.limit_val_batches=1 \
    "dataloaders.batch_size=${BATCH}" "dataloaders.num_workers=${WORKERS}" \
    "${ADV[@]}" depth_lambda=1.0 resume=false \
  || { say "smoke FAILED: fix before spending GPU time on the store"; exit 1; }
step "smoke__depth_score" \
  $PY src/test.py experiment.name="${TAG}_smoke" model=resnet18_depth wrapper=depth \
    uncertainty_score=depth_combined datamodule.datamodule.profile=smoke \
    "datamodule.datamodule.limit_test=${SMOKE_LIMIT:-64}" \
    "depth_teacher_input_size=${TEACHER_INPUT}" depth_attack_scoring=white_box \
    +attack=fgsm "dataloaders.batch_size=${BATCH}" "dataloaders.num_workers=${WORKERS}" \
  || { say "smoke eval FAILED: fix before the chain"; exit 1; }
# The CPU suite proves the white-box gradient in fp32 only; this proves it
# on the device the numbers will come from.
step "smoke__teacher_grad" \
  $PY jobs/track_c_smoke_teacher_grad.py --input-size "$TEACHER_INPUT" \
  || { say "smoke teacher-gradient check FAILED: the _wb cells would be mislabelled"; exit 1; }

# ---------------------------------------------------------------------------
# 1. The store for the training profile.
# ---------------------------------------------------------------------------
step "precompute__${PROFILE}" \
  $PY src/precompute_depth.py --profile "$PROFILE" --out-dir "$DEPTH_DIR" \
    --teacher-input-size "$TEACHER_INPUT" --batch "$BATCH" \
  || { say "precompute FAILED"; exit 1; }

# ---------------------------------------------------------------------------
# 2. Arms. Each pair shares every setting but the head and its loss term.
# ---------------------------------------------------------------------------
train() {  # train <arm-name> <pipe> <extra overrides...>
  local arm="$1" pipe="$2"; shift 2
  step "${arm}__train" \
    $PY src/train.py experiment.name="${TAG}_${arm}" experiment.training_pipe="$pipe" \
      "${TRAIN_COMMON[@]}" "$@"
}

ARMS=()
train standard standard model=resnet18 && ARMS+=("standard:resnet18")
train pgd_at pgd_at model=resnet18 "${ADV[@]}" && ARMS+=("pgd_at:resnet18")
[ "$WITH_TRADES" = "1" ] && { train trades trades model=resnet18 "${ADV[@]}" && ARMS+=("trades:resnet18"); }
for L in $LAMBDAS; do
  train "standard_depth_l${L}" standard_depth "${DEPTH[@]}" "depth_lambda=${L}" \
    && ARMS+=("standard_depth_l${L}:resnet18_depth")
  train "pgd_at_depth_l${L}" pgd_at_depth "${DEPTH[@]}" "${ADV[@]}" "depth_lambda=${L}" \
    && ARMS+=("pgd_at_depth_l${L}:resnet18_depth")
  [ "$WITH_TRADES" = "1" ] && { train "trades_depth_l${L}" trades_depth "${DEPTH[@]}" "${ADV[@]}" "depth_lambda=${L}" \
    && ARMS+=("trades_depth_l${L}:resnet18_depth"); }
done

# ---------------------------------------------------------------------------
# 3. Evaluation cells: <arm>__<dataset>__<cond>__<score>
# ---------------------------------------------------------------------------
is_confidence_axis() { case "$1" in query_*|ace*|overconf|underconf|uncertainty_fgsm) return 0;; *) return 1;; esac; }

cell() {  # cell <arm> <model> <dataset> <cond> <score-key> <wrapper> <score-yaml> [attack-scoring]
  local arm="$1" model="$2" dataset="$3" cond="$4" score="$5" wrapper="$6" yaml="$7" mode="${8:-}"
  local key="${arm}__${dataset}__${cond}__${score}"
  local args=(
    "experiment.name=${TAG}_${arm}" "model=${model}" "wrapper=${wrapper}"
    "uncertainty_score=${yaml}" "datamodule.datamodule.limit_test=${LIMIT_TEST}"
    "dataloaders.batch_size=${BATCH}" "dataloaders.num_workers=${WORKERS}"
  )
  case "$dataset" in
    sid_set)     args+=("datamodule.datamodule.profile=${PROFILE}");;
    so_fake_ood) args+=("datamodule=so_fake_ood" "calib_datamodule=sid_set");;
  esac
  case "$cond" in
    clean) ;;
    corruption_*) args+=("+corruption=${cond#corruption_}");;
    *) args+=("+attack=${cond}");;
  esac
  if [ "$wrapper" = "depth" ]; then
    args+=("depth_teacher_input_size=${TEACHER_INPUT}" "depth_attack_scoring=${mode:-transfer}")
  fi
  step "$key" $PY src/test.py "${args[@]}"
}

for cond in $CONDS $(for c in $CORRUPTIONS; do echo "corruption_$c"; done); do
  for entry in "${ARMS[@]}"; do
    arm="${entry%%:*}"; model="${entry##*:}"
    for dataset in $DATASETS; do
      cell "$arm" "$model" "$dataset" "$cond" msp base multiclass_max_probability
      [ "$model" = "resnet18_depth" ] || continue
      if is_confidence_axis "$cond"; then
        cell "$arm" "$model" "$dataset" "$cond" depth_tr depth depth_consistency transfer
        cell "$arm" "$model" "$dataset" "$cond" combined_tr depth depth_combined transfer
        cell "$arm" "$model" "$dataset" "$cond" depth_wb depth depth_consistency white_box
        cell "$arm" "$model" "$dataset" "$cond" combined_wb depth depth_combined white_box
      else
        cell "$arm" "$model" "$dataset" "$cond" depth depth depth_consistency transfer
        cell "$arm" "$model" "$dataset" "$cond" combined depth depth_combined transfer
      fi
    done
  done
done

say "chain complete -- collate with: python3 jobs/summarise_track_c.py $LOG > RESULTS_track_c.md"
say "READ (a): clean accuracy, standard vs standard_depth -- does the head help or cost?"
say "READ (b): pgd accuracy, pgd_at vs pgd_at_depth -- the Mao et al. bump, if any"
say "READ (c): fd_auroc under attack, msp vs depth vs combined on the SAME arm; GATE G2 |rho| < 0.98 first"
say "READ (d): depth_tr vs depth_wb on the query attacks -- how much of (c) survives the adaptive attack"
