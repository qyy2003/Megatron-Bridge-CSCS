#!/bin/bash
# Spectra of the momentum in checkpoints (tools/ckpt_spectra.py) on one GPU node, in the container:
#   CKPTS="<name>=<iter dir> ..." OUT=<dir> [LAYERS=0,8,16,24,31] [STATE=momentum_buffer] sbatch tools/run_ckpt_spectra.sbatch
#SBATCH --job-name=mbcscs-ckpt-spectra
#SBATCH --account=prep23
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
: "${CKPTS:?}" "${OUT:?}"
LAYERS=${LAYERS:-0,8,16,24,31}; STATE=${STATE-momentum_buffer}
WORKDIR=$(pwd)
EDF=$WORKDIR/logs/edf-ckpt-spectra-$SLURM_JOB_ID.toml
cat > "$EDF" <<TOML
image = "/capstor/scratch/cscs/yqin/mbcscs/images/nemo-26.08.01-alps-mbcscs.sqsh"
mounts = ["/capstor", "/iopsstor", "/users"]
workdir = "$WORKDIR"
writable = true
entrypoint = true

[env]
PMIX_MCA_psec = "native"

[annotations]
com.hooks.cxi.enabled = "false"
TOML
echo "START $(date) job=$SLURM_JOB_ID"
export CKPTS OUT LAYERS STATE
srun --mpi=pmix --network=disable_rdzv_get --environment="$EDF" bash -c '
    for item in $CKPTS; do
        name=${item%%=*}; ckpt=${item#*=}
        echo "== $name: $ckpt"
        python tools/ckpt_spectra.py --ckpt "$ckpt" --out "$OUT/$name" --layers "$LAYERS" --state "$STATE"
    done'
echo "END $(date)"
