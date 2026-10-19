#!/bin/bash
#SBATCH --job-name=mbcscs-infra
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --partition=debug
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#
# Fabric + logging sanity check of the training image (nccl-tests ship with the Alps stack):
#   1. all_reduce_perf, one node (NVLink) and all nodes (Slingshot)
#   2. alltoall_perf,   one node (the EP=4 MoE dispatch) and all nodes (EP spanning nodes)
#   3. the same inter-node all-to-all with the NCCL inspector on: plugin loads, records appear
# CSCS reference for large all-reduce on GH200: ~340 GB/s busbw intra-node, ~90 GB/s inter-node.
#
#   source config.sh && sbatch tools/test-infra.sbatch
set -uo pipefail

WORKDIR=${WORKDIR:-$SLURM_SUBMIT_DIR}
source "$WORKDIR/env/versions.env"
OUT=$WORKDIR/logs/infra-$SLURM_JOB_ID
mkdir -p "$OUT"
EDF=$OUT/edf.toml
cat > "$EDF" <<EOF
image = "$IMAGE_DIR/$IMAGE_NAME"
mounts = ["/capstor", "/iopsstor", "/users"]
workdir = "$WORKDIR"
writable = true
entrypoint = true
[env]
PMIX_MCA_psec = "native"
[annotations]
com.hooks.cxi.enabled = "false"
EOF
RUN=(srun --mpi=pmix --network=disable_rdzv_get --environment="$EDF" --cpus-per-task=72)
ARGS=(-b 8M -e 4G -f 4 -g 1 -w 5 -n 20)

echo "=== NCCL init (network plugin, version)"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET "${RUN[@]}" -N "$SLURM_NNODES" --ntasks-per-node=4 \
    all_reduce_perf -b 1M -e 1M -n 1 -w 1 -g 1 2>&1 | grep -E "NCCL version|Using network|NET/OFI|Plugin|GDR" | sort | uniq -c | head -20

echo "=== all_reduce intra-node (4 GPUs, NVLink)"
"${RUN[@]}" -N 1 --ntasks-per-node=4 all_reduce_perf "${ARGS[@]}"
echo "=== all_reduce inter-node ($((SLURM_NNODES * 4)) GPUs)"
"${RUN[@]}" -N "$SLURM_NNODES" --ntasks-per-node=4 all_reduce_perf "${ARGS[@]}"
echo "=== alltoall intra-node (EP=4 pattern)"
"${RUN[@]}" -N 1 --ntasks-per-node=4 alltoall_perf "${ARGS[@]}"
echo "=== alltoall inter-node"
"${RUN[@]}" -N "$SLURM_NNODES" --ntasks-per-node=4 alltoall_perf "${ARGS[@]}"

echo "=== alltoall inter-node with the NCCL inspector"
NCCL_PROFILER_PLUGIN=/usr/local/lib/libnccl-profiler-inspector.so NCCL_INSPECTOR_ENABLE=1 \
NCCL_INSPECTOR_DUMP_DIR=$OUT/nccl-inspector NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=1000000 \
NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES=0 NCCL_INSPECTOR_ENABLE_P2P=1 NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0 \
    "${RUN[@]}" -N "$SLURM_NNODES" --ntasks-per-node=4 alltoall_perf -b 64M -e 64M -n 10 -w 2 -g 1
ls -la "$OUT/nccl-inspector" | head
f=$(ls "$OUT"/nccl-inspector/*.log 2>/dev/null | head -1)
[ -n "$f" ] && { echo "--- first records of $f"; head -c 1500 "$f"; echo; }
python3 "$WORKDIR/tools/summarize_netlogs.py" "$OUT" 2>&1 | head -40
