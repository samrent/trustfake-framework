#!/usr/bin/env bash
# TB-E3: extract fetched archives into the trees the readers expect.
# Waits on each fetch marker; one .done per extraction; masks never land
# in reader-visible trees (TGIF masks stay in their own subtree).
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a

LOG="${LOGS_PATH}/tb_e3/fetch"; mkdir -p "$LOG"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/extract.log"; }

step() {
  local key="$1"; shift
  if [ -f "$LOG/extract_${key}.done" ]; then say "SKIP  extract $key"; return 0; fi
  say "START extract $key"
  if "$@" >>"$LOG/extract_${key}.log" 2>&1; then
    touch "$LOG/extract_${key}.done"; say "  ok  extract $key"; return 0
  fi
  say "  FAILED extract $key"; return 1
}

wait_marker() {
  local marker="$LOG/$1.done" deadline=$((SECONDS + 21600))
  until [ -f "$marker" ]; do
    [ $SECONDS -gt $deadline ] && { say "gave up waiting on $1"; return 1; }
    sleep 120
  done
}

synthbuster() {
  cd "${DATA_PATH}/synthbuster" && unzip -qn synthbuster.zip && cd - >/dev/null
}

imd2020() {
  local d="${DATA_PATH}/imd2020"
  mkdir -p "$d/real_life" "$d/camera_real" "$d/gan_inpaint"
  unzip -qn "$d/IMD2020.zip" -d "$d/real_life" || return 1
  for i in 01 02 03 04; do
    unzip -qn "$d/IMD2020_real_$i.zip" -d "$d/camera_real" || return 1
  done
  for i in 01 02 03 04 05 06 07; do
    unzip -qn "$d/IMD2020_Generative_Image_Inpainting_yu2018_$i.zip" -d "$d/gan_inpaint" || return 1
  done
}

tgif_tars() {
  local d="${DATA_PATH}/tgif"
  for tool in orig ps-sp; do
    mkdir -p "$d/extracted/$tool"
    for split in training validation testing; do
      tar xzf "$d/$tool/${tool}_${split}.tar.gz" -C "$d/extracted/$tool" || return 1
    done
  done
}

tgif2_random() {
  local d="${DATA_PATH}/tgif"
  # Per-tool tars fetched via DAV (the share-zip endpoint returns an empty
  # body); each unpacks a {split}/{category}/ tree into extracted/<tool>-rnd/.
  for tarball in "$d"/tgif2_random/*.tar.gz; do
    tool="$(basename "$tarball" | sed 's/_\(training\|validation\|testing\)\.tar\.gz$//')"
    mkdir -p "$d/extracted/${tool}-rnd"
    tar xzf "$tarball" -C "$d/extracted/${tool}-rnd" || return 1
  done
}

wait_marker synthbuster   && step synthbuster synthbuster
wait_marker imd2020       && step imd2020 imd2020
wait_marker tgif_orig     && wait_marker tgif_ps_sp && step tgif_tars tgif_tars
wait_marker tgif2_random  && step tgif2_random tgif2_random

say "extract_archives run done"
