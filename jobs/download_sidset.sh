#!/bin/bash

# ------------------------------------------------------------------------------
# Slurm directives
# ------------------------------------------------------------------------------

#SBATCH --job-name=download_sidset
#SBATCH --output=lucia_out/%j_%x.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=p_ariac_multitel

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
module purge
module load EasyBuild/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load virtualenv/20.23.1-GCCcore-12.3.0
module list

source ~/TrustFake/.venv/bin/activate
cd ../

TARGET_DIR="/gpfs/projects/multitel/p_ariac_multitel/TReC26/Trustfake/sid_set"
mkdir -p "$TARGET_DIR"

# ------------------------------------------------------------------------------
# Running the code
# ------------------------------------------------------------------------------

echo "--------------- Running the code ---------------"

echo -n "This run started on: "
date

python -c "
from datasets import load_dataset

load_dataset('saberzl/SID_Set', cache_dir='$TARGET_DIR')
"

echo -n "This run completed on: "
date
