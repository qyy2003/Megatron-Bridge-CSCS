#!/bin/bash
# Per-rank launcher: srun starts one copy per GPU inside the container.
#
#   - NUMA:   GPU i is attached to Grace CPU / NUMA node i (cores 72i..72i+71). Bridge uses
#             cuda:$SLURM_LOCALID, so bind the rank's CPUs to that node and keep host memory
#             on the CPU nodes 0-3 (the GPU HBM is also exposed as NUMA nodes 4/12/20/28 and
#             must never receive host allocations).
#   - netmon: local rank 0 of every node samples NIC/NVLink counters in the background
#             (NETLOG >= 1).
#   - nsys:   ranks listed in NSYS_RANKS run under `nsys profile`; the trainer opens and closes
#             the capture window with cudaProfilerStart/Stop (profiling.profile_step_*).
#
# Usage: rank_wrapper.sh python train/pretrain.py ...
set -uo pipefail

L=${SLURM_LOCALID:?not launched by srun}
R=${SLURM_PROCID:?not launched by srun}

NETMON_PID=
if [ "$L" = 0 ] && [ "${NETLOG:-0}" -ge 1 ]; then
    mkdir -p "$NETLOG_DIR/netmon"
    python3 "${SRC:-$WORKDIR}/tools/netmon.py" \
        --out "$NETLOG_DIR/netmon/$(hostname).jsonl" \
        --interval "${NETMON_INTERVAL:-1}" --nvlink-every "${NETMON_NVLINK_EVERY:-5}" &
    NETMON_PID=$!
fi

CMD=("$@")
if [ "${NSYS:-0}" = 1 ] && [[ ",${NSYS_RANKS:-0}," == *",$R,"* ]]; then
    mkdir -p "$NSYS_DIR"
    CMD=(nsys profile
        --trace="${NSYS_TRACE:-cuda,nvtx}"
        --capture-range=cudaProfilerApi --capture-range-end=stop
        --cuda-graph-trace=node
        --sample=none --cpuctxsw=none
        --force-overwrite=true
        --output="$NSYS_DIR/rank${R}_$(hostname)_${SLURM_JOB_ID}.${SLURM_STEP_ID:-0}"
        "${CMD[@]}")
fi

numactl --cpunodebind="$L" --membind=0-3 "${CMD[@]}"
RC=$?

if [ -n "$NETMON_PID" ]; then
    kill "$NETMON_PID" 2>/dev/null
    wait "$NETMON_PID" 2>/dev/null
fi
exit $RC
