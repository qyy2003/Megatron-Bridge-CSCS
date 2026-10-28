#!/bin/bash
# Unit tests of the Dion patch set on one node (4 GPUs):
#   SNAP=<code snapshot with patch sets muon-zero1,dion> sbatch tests/run_dion_tests.sbatch
#SBATCH --job-name=mbcscs-dion-unit
#SBATCH --account=prep23
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
: "${SNAP:?set SNAP to a code snapshot built with PATCHES=muon-zero1,dion}"
WORKDIR=$(pwd)
EDF=$WORKDIR/logs/edf-dion-unit-$SLURM_JOB_ID.toml
cat > "$EDF" <<TOML
image = "/capstor/scratch/cscs/yqin/mbcscs/images/nemo-26.08.01-alps-mbcscs.sqsh"
mounts = ["/capstor", "/iopsstor", "/users", "$SNAP:/opt/Megatron-Bridge"]
workdir = "$WORKDIR"
writable = true
entrypoint = true

[env]
PMIX_MCA_psec = "native"

[annotations]
com.hooks.cxi.enabled = "false"
TOML
echo "START $(date) job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST snapshot=$SNAP"
srun --mpi=pmix --network=disable_rdzv_get --environment="$EDF" bash -c '
    export PYTHONPATH=/opt/Megatron-Bridge/3rdparty/Megatron-LM:/opt/Megatron-Bridge/src
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0   # the test switches TF32 itself (on for the timing)
    export TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
    python -c "import megatron.core.optimizer.dion as d; print(\"dion from\", d.__file__)"
    torchrun --nproc_per_node=4 --master_port=29511 tests/test_dion.py'
echo "END $(date)"
