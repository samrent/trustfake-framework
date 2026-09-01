#!/usr/bin/env bash
# TB-E3 R2: fetch the non-HF shortlist (spec documentation/specs/tb-e3-curation-ladder.md).
#
# Sources verified live 2026-09-01 (see reference/dataset-availability-tb-e3.md
# and the session's URL-resolution pass). Smallest first, sequential, one
# .done marker per step -- delete a marker to force a re-fetch. Every fetch
# is resumable (wget -c / curl -C - / snapshot_download).
#
# Deliberate selections, recorded here rather than implied:
#   * TGIF: originals + masks + ps-sp only. ps-sp (Photoshop/Firefly
#     generative fill) is the L4 held-out-tool candidate -- eval-only, so it
#     must exist locally, but the SD2/SDXL semantic-mask forgeries are
#     superseded by TGIF2's random-mask share (boundary-cheating control).
#   * TGIF2: the random-mask share (sd2/sdxl/flux1 + masks + metadata,
#     78 GB), not the 118 GB semantic-mask FLUX share.
#   * VISION: images only (~40 GB), native + FB/WA re-shares; videos are out
#     of scope for a still-image detector.
#   * GenImage: NOT fetched -- 654 GB monolithic (Dataverse) or Drive-quota
#     gdown; deferred, recorded in the spec change log as an open item.
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv/bin/python

LOG="${LOGS_PATH}/tb_e3/fetch"; mkdir -p "$LOG"
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/fetch_rest.log"; }

step() {
  local key="$1"; shift
  if [ -f "$LOG/${key}.done" ]; then say "SKIP  $key"; return 0; fi
  say "START $key"
  local t0=$SECONDS
  if "$@" >>"$LOG/${key}.log" 2>&1; then
    touch "$LOG/${key}.done"; say "  ok  $key ($((SECONDS - t0))s)"; return 0
  fi
  say "  FAILED $key -- see $LOG/${key}.log"; return 1
}

# --- Synthbuster: Zenodo 10066460, 12.4 GB, CC BY-NC-SA ---------------------
synthbuster() {
  local dest="${DATA_PATH}/synthbuster"; mkdir -p "$dest"
  wget -c -q --show-progress --progress=dot:giga \
    'https://zenodo.org/records/10066460/files/synthbuster.zip?download=1' \
    -O "$dest/synthbuster.zip"
}

# --- IMD2020: staff.utia.cas.cz, broken TLS on purpose (-k) -----------------
imd2020() {
  local dest="${DATA_PATH}/imd2020"; mkdir -p "$dest"
  local base="https://staff.utia.cas.cz/novozada/db"
  local files=(IMD2020.zip IMD2020_real_01.zip IMD2020_real_02.zip
               IMD2020_real_03.zip IMD2020_real_04.zip)
  for i in 01 02 03 04 05 06 07; do
    files+=("IMD2020_Generative_Image_Inpainting_yu2018_${i}.zip")
  done
  files+=(IMD2020_Generative_Image_Inpainting_yu2018_mask.zip)
  for f in "${files[@]}"; do
    wget -c -q --no-check-certificate "$base/$f" -O "$dest/$f" || return 1
  done
}

# --- AUDITS: DivyaApp/AUDITS on HF, 37.2 GB, MIT card -----------------------
audits() {
  local dest="${DATA_PATH}/audits"
  $PY - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("DivyaApp/AUDITS", repo_type="dataset", local_dir="$dest", max_workers=2)
EOF
}

# --- VISION: images only, per-device tree preserved -------------------------
vision_images() {
  local dest="${DATA_PATH}/vision"; mkdir -p "$dest"
  wget -c -q https://lesc.dinfo.unifi.it/VISION/VISION_files.txt -O "$dest/VISION_files.txt"
  grep '/images/' "$dest/VISION_files.txt" > "$dest/vision_images.txt"
  say "  vision: $(wc -l < "$dest/vision_images.txt") image files"
  # -x keeps the device/collection tree; -nH drops the hostname prefix.
  (cd "$dest" && wget -c -x -nH --cut-dirs=1 -q -i vision_images.txt)
}

# --- TGIF share 1: originals + masks + ps-sp (L4 candidate) -----------------
_tgif_fetch() {  # _tgif_fetch <share-token> <folder> <dest-subdir>
  local token="$1" folder="$2" dest="${DATA_PATH}/tgif/$3"; mkdir -p "$dest"
  for split in training validation testing; do
    curl -sS -C - -H 'X-Requested-With: XMLHttpRequest' \
      -o "$dest/${folder}_${split}.tar.gz" \
      "https://cloud.ilabt.imec.be/public.php/dav/files/${token}/${folder}/${folder}_${split}.tar.gz" \
      || return 1
  done
}

tgif_orig()  { _tgif_fetch xEeAzrY7ES9KA8o orig  orig; }
tgif_masks() { _tgif_fetch xEeAzrY7ES9KA8o masks masks; }
tgif_ps_sp() { _tgif_fetch xEeAzrY7ES9KA8o ps-sp ps-sp; }

# --- TGIF2 random-mask share: whole share as one zip (78 GB) ----------------
tgif2_random() {
  local dest="${DATA_PATH}/tgif/tgif2_random"; mkdir -p "$dest"
  curl -sS -C - -H 'X-Requested-With: XMLHttpRequest' \
    -o "$dest/tgif2_random.zip" \
    'https://cloud.ilabt.imec.be/index.php/s/GDGewtTFcHccaNj/download?accept=zip'
}

# One process per HOST is the useful parallelism (Zenodo throttles a single
# stream to ~1 MB/s; queueing the HF and unifi fetches behind it would be
# the mistake). Pass step names to run a subset:
#   bash jobs/tb_e3/fetch_rest.sh synthbuster
#   bash jobs/tb_e3/fetch_rest.sh tgif_orig tgif_masks tgif_ps_sp tgif2_random
ALL=(synthbuster imd2020 audits vision_images tgif_orig tgif_masks tgif_ps_sp tgif2_random)
for key in "${@:-${ALL[@]}}"; do
  step "$key" "$key"
done

say "fetch_rest run done: ${*:-all}"
