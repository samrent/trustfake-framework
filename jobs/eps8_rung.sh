#!/usr/bin/env bash
# Wait out the in-flight sweep, then run only the 8/255 rung.
# Sequential by design: two training processes on one 24GB card that is also
# carrying the sotto daemons would contend for memory and make both timings
# meaningless.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
while pgrep -f "src/sweep.py --strategy AC" > /dev/null 2>&1; do sleep 60; done
exec .venv/bin/python src/sweep.py --strategy A --epochs 12 --batch 32 \
     --workers 6 --clean-floor 0.75 --only e8
