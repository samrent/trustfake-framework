#!/usr/bin/env bash
# TB-E3 R2: fetch Community Forensics (Small) -- the primary synthetic
# environment (spec: documentation/specs/tb-e3-curation-ladder.md).
#
# The -Small variant on purpose: it is the redistributable one, its reals
# (FFHQ / VISION / COCO / LHQ) ship inside the parquet rather than as
# pointers, and R1 sized it at 278 GB against the base's 1.08 TB. Per-image
# `model_name` (4,803 generators) is what makes the L3 held-out-generator
# legs choosable at ingestion.
#
# Resumable: snapshot_download skips complete files, and the .done marker
# follows the chain convention -- delete it to force a re-verify.
set -uo pipefail

cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a
PY=.venv/bin/python

LOG="${LOGS_PATH}/tb_e3/fetch"; mkdir -p "$LOG"
DEST="${DATA_PATH}/community_forensics_small"
MARKER="$LOG/community_forensics_small.done"

if [ -f "$MARKER" ]; then echo "SKIP community_forensics_small"; exit 0; fi

echo "[$(date -u +%F' '%H:%M:%S)] START community_forensics_small -> $DEST"
if $PY - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    "OwensLab/CommunityForensics-Small",
    repo_type="dataset",
    local_dir="$DEST",
    max_workers=4,
)
EOF
then
  touch "$MARKER"
  echo "[$(date -u +%F' '%H:%M:%S)] ok community_forensics_small"
else
  echo "[$(date -u +%F' '%H:%M:%S)] FAILED community_forensics_small"
  exit 1
fi
