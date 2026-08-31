#!/usr/bin/env bash
# Fetch the two Track B evaluation datasets.
#
# Neither is needed for training -- both are evaluation-only here, so the
# 28 GB FakeClue train.zip is deliberately NOT downloaded. What is fetched:
#
#   FakeClue     test.zip (1.3 GB) + data_json/test.json  -- cross-dataset
#   So-Fake-OOD  N test shards (~3 GB each, 46 total)     -- distribution shift
#
# So-Fake-OOD is ~135 GB in full. Shards are the unit of sampling and the
# choice is seeded by manifest_seed, so a partial download is a
# protocol-defined sample rather than an accident of what finished -- see
# trustfake.data.so_fake_ood.select_shards. Set N_SHARDS to control it.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

N_SHARDS="${N_SHARDS:-2}"
FC_DIR="${DATA_PATH}/fake_clue"
OOD_DIR="${DATA_PATH}/so_fake_ood"

echo "FakeClue  -> $FC_DIR"
echo "So-Fake-OOD ($N_SHARDS shards) -> $OOD_DIR"

.venv/bin/python - "$FC_DIR" "$OOD_DIR" "$N_SHARDS" <<'PY'
import sys, time
from huggingface_hub import hf_hub_download, HfApi

fc_dir, ood_dir, n_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])

for remote in ("data_json/test.json", "test.zip"):
    t0 = time.time()
    hf_hub_download("lingcco/FakeClue", remote, repo_type="dataset", local_dir=fc_dir)
    print(f"  FakeClue {remote}  {time.time()-t0:.0f}s", flush=True)

# Take the SAME shards select_shards would pick, so what lands on disk is the
# protocol's sample rather than the first N alphabetically.
import torch
info = HfApi().repo_info("saberzl/So-Fake-OOD", repo_type="dataset")
shards = sorted(s.rfilename for s in info.siblings
                if s.rfilename.endswith(".parquet") and "test" in s.rfilename)
g = torch.Generator().manual_seed(0)          # DEFAULT_MANIFEST_SEED
picked = torch.randperm(len(shards), generator=g)[:n_shards].tolist()
for i in sorted(picked):
    t0 = time.time()
    hf_hub_download("saberzl/So-Fake-OOD", shards[i], repo_type="dataset",
                    local_dir=ood_dir)
    print(f"  So-Fake-OOD {shards[i]}  {time.time()-t0:.0f}s", flush=True)
print("done")
PY
