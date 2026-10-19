#!/bin/bash
#
# Usage: ./launch.sh <mode> [nodes] [steps]
#
# Modes:
#   smoke       Pipeline check (2 nodes, debug partition): the real Qwen3.5 architecture
#               (Gated DeltaNet + gated attention + MoE + MTP) shrunk to one 4-layer block and
#               32 experts, mock data, 20 steps, one eval and one checkpoint save, network
#               logging (NETLOG=2) and an nsys capture of steps 10-12 on rank 0.
#   throughput  Full Qwen3.5-35B-A3B, default 8 nodes, 30 steps, no eval, no checkpoints.
#               Mock data unless BLEND is set.
#   train       Full model on the tokenized Lightning blend (BLEND), checkpoints every
#               SAVE_INTERVAL steps, auto-resume and auto-resubmit until <steps> is reached.
#
# Examples:
#   ./launch.sh smoke
#   ./launch.sh throughput 8
#   EP=8 GRAD_FP32=1 ./launch.sh throughput 16 50
#   NSYS=1 NSYS_STEPS=20:22 NSYS_RANKS=0,4 ./launch.sh throughput 8 30
#   ./launch.sh train 16 200000
#   DRYRUN=1 ./launch.sh train 16 200000      # resolve + print the config (1-node debug job)
#   SUBMIT=0 ./launch.sh train 16 200000      # only write logs/<job>.sbatch
#
# Knobs (environment variables, [default]):
#   EP [4] TP [1] PP [2] CP [1] ETP [1]   parallel sizes; EP=4 keeps the expert all-to-all
#                                         inside a node (NVLink), only DP/PP traffic crosses Slingshot;
#                                         PP=2 halves the static memory (PP=1 does not fit, WORKLOG.md)
#   MBS [1] GBS [512] SEQ_LEN [4096]      micro/global batch, sequence length (recipe values)
#   GRAD_FP32 [0]       1 = fp32 gradient reduction as in the GB200 recipe (+~20 GB/GPU at EP=4);
#                       0 = bf16, as NVIDIA's Hopper MoE recipes (Qwen3-30B-A3B H100) do
#   CUDA_GRAPHS [1]     0 = disable the recipe's TE-scoped CUDA graphs (attn, moe_router, moe_preprocess)
#   DISPATCHER [alltoall]  MoE token dispatcher: alltoall (NCCL) | hybridep (the GB200 recipe's
#                       fused NVLink dispatcher; only with the expert group inside a node, EP*ETP <= 4)
#   RECOMPUTE [none]    selective = recompute moe_act + layernorm (activation memory)
#   LR MIN_LR WARMUP    learning-rate schedule [recipe: 3e-4 -> 3e-5 cosine, 500 warmup steps]
#   BLEND               blend JSON from data/build_blend.py [train: $DATA_ROOT/blends/lightning35_qwen35.json]
#   NETLOG [2]          0 off | 1 NIC + NVLink counters and per-step W&B net metrics
#                       | 2 + NCCL inspector, every collective and P2P op | 3 + per-op timestamps
#   NSYS [0]            1 = raw nsys timeline of NSYS_STEPS [10:12] on NSYS_RANKS [0]
#   SAVE_INTERVAL [500] EVAL_INTERVAL [500] EVAL_ITERS [32]
#   WALLTIME PARTITION MAX_RESUBMITS [50] EXP_SUFFIX
#   EXTRA               more KEY=VALUE overrides, e.g. EXTRA="optimizer.weight_decay=0.05"
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$WORKDIR/config.sh" ] || { echo "copy config.sh.example to config.sh first"; exit 1; }
source "$WORKDIR/config.sh"
source "$WORKDIR/env/versions.env"

USAGE="Usage: ./launch.sh <smoke|throughput|train> [nodes] [steps]"
MODE=${1:?$USAGE}

hms_to_mins() {
    local h m s
    IFS=: read -r h m s <<< "$1"
    echo $((10#$h * 60 + 10#$m + (10#${s:-0} + 59) / 60))
}

EP=${EP:-4}; TP=${TP:-1}; PP=${PP:-2}; CP=${CP:-1}; ETP=${ETP:-1}
MBS=${MBS:-1}; SEQ_LEN=${SEQ_LEN:-4096}
GRAD_FP32=${GRAD_FP32:-0}; CUDA_GRAPHS=${CUDA_GRAPHS:-1}; RECOMPUTE=${RECOMPUTE:-none}
DISPATCHER=${DISPATCHER:-alltoall}
NSYS_STEPS=${NSYS_STEPS:-10:12}; NSYS_RANKS=${NSYS_RANKS:-0}; NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
EVAL_ITERS=${EVAL_ITERS:-32}
SUMMARIZE=1
OVERRIDES=()

################ Mode ################
case $MODE in
    smoke)
        NODES=${2:-2}; STEPS=${3:-20}
        GBS=${GBS:-32}
        PARTITION=${PARTITION:-debug}; WALLTIME=${WALLTIME:-00:40:00}
        NETLOG=${NETLOG:-2}; NSYS=${NSYS:-1}
        export NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV}
        # One 4-layer block of the real pattern (3x Gated DeltaNet + 1 gated attention): the
        # bridge stores the 40-entry per-layer pattern, so replace it by its period.
        OVERRIDES+=(model.num_layers=4 model.linear_attention_freq=4 model.num_moe_experts=32
                    validation.eval_interval=10 validation.eval_iters=2
                    scheduler.lr_warmup_iters=5
                    checkpoint.save_interval=15)
        CKPT=save; RESUBMIT=false; MAX_RESUBMITS=0
        ;;
    throughput)
        NODES=${2:-8}; STEPS=${3:-30}
        GBS=${GBS:-512}
        PARTITION=${PARTITION:-normal}; WALLTIME=${WALLTIME:-01:00:00}
        NETLOG=${NETLOG:-2}; NSYS=${NSYS:-0}
        OVERRIDES+=(validation.eval_interval=null scheduler.lr_warmup_iters=10)
        CKPT=none; RESUBMIT=false; MAX_RESUBMITS=0
        ;;
    train)
        NODES=${2:?$USAGE}; STEPS=${3:?$USAGE}
        GBS=${GBS:-512}
        PARTITION=${PARTITION:-normal}; WALLTIME=${WALLTIME:-12:00:00}
        NETLOG=${NETLOG:-2}; NSYS=${NSYS:-0}
        BLEND=${BLEND:-$DATA_ROOT/blends/lightning35_qwen35.json}
        OVERRIDES+=(validation.eval_interval=${EVAL_INTERVAL:-500} validation.eval_iters=$EVAL_ITERS)
        CKPT=save_load; RESUBMIT=true; MAX_RESUBMITS=${MAX_RESUBMITS:-50}
        SUMMARIZE=0
        ;;
    *) echo "$USAGE"; exit 1 ;;
esac

################ Sanity checks (cheap, before anything is queued) ################
WORLD=$((NODES * 4))
MP=$((TP * PP * CP))
if (( WORLD % MP != 0 )); then echo "world size $WORLD not divisible by TP*PP*CP=$MP"; exit 1; fi
DP=$((WORLD / MP))
if (( WORLD % (EP * ETP * PP) != 0 )); then echo "world size $WORLD not divisible by EP*ETP*PP"; exit 1; fi
if (( GBS % (MBS * DP) != 0 )); then echo "GBS=$GBS must be a multiple of MBS*DP=$((MBS * DP))"; exit 1; fi
[ -f "$IMAGE_DIR/$IMAGE_NAME" ] || { echo "image $IMAGE_DIR/$IMAGE_NAME missing: sbatch env/build_image.sbatch"; exit 1; }
[ -f "$HF_HOME/.prefetched-${HF_MODEL_ID//\//--}" ] || { echo "run tools/prefetch_hf.sh first (Qwen3.5 config/tokenizer for offline jobs)"; exit 1; }
if [ -n "${BLEND:-}" ] && [ ! -f "$BLEND" ]; then echo "BLEND=$BLEND does not exist (data/build_blend.py)"; exit 1; fi
for d in Megatron-Bridge Megatron-Bridge/3rdparty/Megatron-LM; do
    git -C "$WORKDIR/$d" cat-file -e "$( [ "$d" = Megatron-Bridge ] && echo "$MBRIDGE_COMMIT" || echo "$MLM_COMMIT")^{commit}" \
        || { echo "$d: pinned commit missing; git submodule update --init --recursive"; exit 1; }
done
# Patches must apply to the pinned commits (that is what the job snapshots), whatever state
# the submodule working trees are in: check them against a throw-away index.
check_patches() {  # <repo> <commit> <patch dir>
    local idx p
    idx=$(mktemp)
    GIT_INDEX_FILE=$idx git -C "$1" read-tree "$2"
    for p in "$3"/*.patch; do
        [ -e "$p" ] || continue
        p=$(realpath "$p")
        GIT_INDEX_FILE=$idx git -C "$1" apply --cached "$p" || { rm -f "$idx"; echo "patch does not apply: $p"; return 1; }
    done
    rm -f "$idx"
}
check_patches "$WORKDIR/Megatron-Bridge" "$MBRIDGE_COMMIT" "$WORKDIR/patches/megatron-bridge"
check_patches "$WORKDIR/Megatron-Bridge/3rdparty/Megatron-LM" "$MLM_COMMIT" "$WORKDIR/patches/megatron-lm"

################ Config overrides ################
OVERRIDES+=(
    model.tensor_model_parallel_size=$TP
    model.pipeline_model_parallel_size=$PP
    model.context_parallel_size=$CP
    model.expert_model_parallel_size=$EP
    model.expert_tensor_parallel_size=$ETP
    train.micro_batch_size=$MBS
    train.global_batch_size=$GBS
    train.train_iters=$STEPS
    dataset.seq_length=$SEQ_LEN
    dataset.path_to_cache=$CACHE_ROOT/dataset_index
    logger.tensorboard_dir=RUN_DIR/tensorboard
)
(( TP > 1 )) && OVERRIDES+=(model.sequence_parallel=true)
if [ "$GRAD_FP32" = 1 ]; then GF=true; else GF=false; fi
OVERRIDES+=(mixed_precision.grad_reduce_in_fp32=$GF ddp.grad_reduce_in_fp32=$GF)
[ "$CUDA_GRAPHS" = 0 ] && OVERRIDES+=(model.cuda_graph_impl=none)
case $DISPATCHER in
    alltoall) ;;
    hybridep)
        # HybridEP's NVLink path only: across nodes it needs InfiniBand GPU-initiated RDMA.
        # Expert groups are consecutive ranks (ETP innermost), so EP*ETP <= 4 stays in a node.
        (( EP * ETP <= 4 )) || { echo "DISPATCHER=hybridep needs EP*ETP <= 4 (expert group inside one node)"; exit 1; }
        OVERRIDES+=(model.moe_token_dispatcher_type=flex model.moe_flex_dispatcher_backend=hybridep) ;;
    *) echo "DISPATCHER must be alltoall or hybridep"; exit 1 ;;
esac
case $RECOMPUTE in
    none) ;;
    selective) OVERRIDES+=(model.recompute_granularity=selective "model.recompute_modules=[moe_act,layernorm]") ;;
    *) echo "RECOMPUTE must be none or selective"; exit 1 ;;
esac
[ -n "${LR:-}" ] && OVERRIDES+=(optimizer.lr=$LR)
[ -n "${MIN_LR:-}" ] && OVERRIDES+=(optimizer.min_lr=$MIN_LR)
[ -n "${WARMUP:-}" ] && OVERRIDES+=(scheduler.lr_warmup_iters=$WARMUP)
case $CKPT in
    none) OVERRIDES+=(checkpoint.save=null checkpoint.load=null) ;;
    save) OVERRIDES+=(checkpoint.save=RUN_DIR/checkpoints checkpoint.load=null) ;;
    save_load)
        EXIT_MINS=$(( $(hms_to_mins "$WALLTIME") - 15 ))
        OVERRIDES+=(checkpoint.save=RUN_DIR/checkpoints checkpoint.load=RUN_DIR/checkpoints
                    checkpoint.save_interval=${SAVE_INTERVAL:-500} train.exit_duration_in_mins=$EXIT_MINS) ;;
esac
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ] && OVERRIDES+=($EXTRA)

################ Names and paths ################
TAG="q35a3b-${NODES}n-ep${EP}"
(( TP > 1 )) && TAG+="-tp$TP"; (( PP > 1 )) && TAG+="-pp$PP"; (( CP > 1 )) && TAG+="-cp$CP"
TAG+="-mbs${MBS}-gbs${GBS}-sl${SEQ_LEN}"
[ "$GRAD_FP32" = 1 ] && TAG+="-gf32"; [ "$CUDA_GRAPHS" = 0 ] && TAG+="-nocg"; [ "$RECOMPUTE" != none ] && TAG+="-rc"
[ "$DISPATCHER" != alltoall ] && TAG+="-$DISPATCHER"
if [ "$MODE" = train ]; then
    # Stable across resubmissions and node counts: torch_dist checkpoints reshard on load
    # across data-parallel sizes, so only data-semantic settings are in the name. The
    # optimizer state is saved DP-reshardable (Megatron default), so TP/PP/CP/EP/ETP must
    # stay fixed for the life of a run (checked below) unless it was saved with
    # checkpoint.dist_ckpt_optim_fully_reshardable=true.
    EXP_NAME="train-q35a3b-gbs${GBS}-sl${SEQ_LEN}${EXP_SUFFIX:+-$EXP_SUFFIX}"
else
    EXP_NAME="${MODE}-${TAG}${EXP_SUFFIX:+-$EXP_SUFFIX}"
fi
JOB_NAME="mbcscs-${EXP_NAME}"
RUN_DIR=$RUN_ROOT/$EXP_NAME
SCRIPT="$WORKDIR/logs/${JOB_NAME}.sbatch"
mkdir -p "$WORKDIR/logs"
if [ "$MODE" = train ] && [ "${DRYRUN:-0}" != 1 ]; then
    LAYOUT="TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP"
    OLD_LAYOUT=$(cat "$RUN_DIR/layout" 2>/dev/null || true)
    if [ -s "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ] && [ -n "$OLD_LAYOUT" ] \
            && [ "$OLD_LAYOUT" != "$LAYOUT" ] && [ "${FORCE_LAYOUT:-0}" != 1 ]; then
        echo "$RUN_DIR has checkpoints written with $OLD_LAYOUT; requested $LAYOUT."
        echo "The optimizer state is only DP-reshardable: keep the layout (the node count may change),"
        echo "or FORCE_LAYOUT=1 if the checkpoints were saved with checkpoint.dist_ckpt_optim_fully_reshardable=true."
        exit 1
    fi
    [ "${SUBMIT:-1}" = 1 ] && mkdir -p "$RUN_DIR" && echo "$LAYOUT" > "$RUN_DIR/layout"
fi

TRAIN_ARGS=(--recipe "${RECIPE:-qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config}"
            --profile "$WORKDIR/conf/qwen35_35b_a3b_gh200.yaml")
[ -n "${BLEND:-}" ] && TRAIN_ARGS+=(--blend "$BLEND")
[ "${DRYRUN:-0}" = 1 ] && TRAIN_ARGS+=(--dry-run)
for o in "${OVERRIDES[@]}"; do TRAIN_ARGS+=("${o//RUN_DIR/$RUN_DIR}"); done

if [ "${DRYRUN:-0}" = 1 ]; then
    NODES=1; PARTITION=debug; WALLTIME=00:15:00; NETLOG=0; NSYS=0; RESUBMIT=false; SUMMARIZE=0
    JOB_NAME+="-dryrun"; SCRIPT="$WORKDIR/logs/${JOB_NAME}.sbatch"
    # The dry run resolves the config for the real world size on a single node.
    export WORLD_SIZE=$WORLD
fi

################ Generate the self-contained sbatch ################
{
cat <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALLTIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=$([ "${DRYRUN:-0}" = 1 ] && echo 1 || echo 4)
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --output=${WORKDIR}/logs/%x-%j.log
#SBATCH --error=${WORKDIR}/logs/%x-%j.log
#SBATCH --no-requeue
#
# Generated by launch.sh on $(date -Iseconds) -- self-contained, resubmittable as is.

################ Settings ################
WORKDIR=$(printf %q "$WORKDIR")
SCRIPT_PATH=$(printf %q "$SCRIPT")
MODE=$MODE
EXP_NAME=$EXP_NAME
RUN_DIR=$(printf %q "$RUN_DIR")
IMAGE=$(printf %q "$IMAGE_DIR/$IMAGE_NAME")
CODE_ROOT=$(printf %q "$MBCSCS_IOPS/code")
CACHE_ROOT=$(printf %q "$CACHE_ROOT")
HF_HOME=$(printf %q "$HF_HOME")
MBRIDGE_COMMIT=$MBRIDGE_COMMIT
MLM_COMMIT=$MLM_COMMIT
NETLOG=$NETLOG
NETMON_INTERVAL=${NETMON_INTERVAL:-1}
NETMON_NVLINK_EVERY=${NETMON_NVLINK_EVERY:-5}
NSYS=$NSYS
NSYS_STEPS=$NSYS_STEPS
NSYS_RANKS=$NSYS_RANKS
NSYS_TRACE=$NSYS_TRACE
SUMMARIZE=$SUMMARIZE
RESUBMIT=$RESUBMIT
MAX_RESUBMITS=$MAX_RESUBMITS
TRAIN_ITERS=$STEPS
WANDB_PROJECT=$(printf %q "${WANDB_PROJECT:-}")
WANDB_ENTITY=$(printf %q "${WANDB_ENTITY:-}")
WANDB_MODE=$(printf %q "${WANDB_MODE:-}")
EOF
[ -n "${NCCL_DEBUG:-}" ] && echo "export NCCL_DEBUG=$NCCL_DEBUG"
[ -n "${NCCL_DEBUG_SUBSYS:-}" ] && echo "export NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
[ -n "${WORLD_SIZE:-}" ] && echo "export WORLD_SIZE=$WORLD_SIZE RANK=0"
printf 'TRAIN_ARGS=('; printf ' %q' "${TRAIN_ARGS[@]}"; printf ' )\n\n'
cat "$WORKDIR/train/job_body.sh"
} > "$SCRIPT"
chmod +x "$SCRIPT"
bash -n "$SCRIPT"

echo "Generated: $SCRIPT"
echo "  run dir: $RUN_DIR"
echo "  layout : nodes=$NODES world=$WORLD TP=$TP PP=$PP CP=$CP EP=$EP DP=$DP MBS=$MBS GBS=$GBS seq=$SEQ_LEN"
echo "  logging: NETLOG=$NETLOG NSYS=$NSYS W&B=$([ -n "${WANDB_API_KEY:-}" ] && echo "on${WANDB_MODE:+ ($WANDB_MODE)}" || { [ "${WANDB_MODE:-}" = offline ] && echo "offline (no key)" || echo off; })"
if [ "${SUBMIT:-1}" = 1 ]; then
    sbatch "$SCRIPT"
fi
