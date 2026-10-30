#!/bin/bash
#
# Usage: ./launch.sh <mode> [nodes] [steps | tokens]
#
# Modes:
#   smoke       Pipeline check (2 nodes, debug partition): the real Qwen3.5 architecture
#               (Gated DeltaNet + gated attention + MoE + MTP) shrunk to one 4-layer block and
#               32 experts, mock data, 20 steps, one eval and one checkpoint save, network
#               logging (NETLOG=2) and an nsys capture of steps 10-12 on rank 0.
#   throughput  Full Qwen3.5-35B-A3B, default 8 nodes, 30 steps, no eval, no checkpoints.
#               Mock data unless BLEND is set.
#   train       Full model on the tokenized Lightning blend (BLEND) for a token budget (1T, 500B,
#               2.5e12; a plain integer is an iteration count), checkpoints every SAVE_INTERVAL
#               steps, auto-resume and auto-resubmit until done. The global batch size ramps up
#               DeepSeek-style from 2.1M to 10.5M tokens over the first 3% of the run (GBS_RAMP).
#
# Examples:
#   ./launch.sh smoke
#   ./launch.sh throughput 8
#   EP=8 GRAD_FP32=1 ./launch.sh throughput 16 50
#   NSYS=1 NSYS_STEPS=20:22 NSYS_RANKS=0,4 ./launch.sh throughput 8 30
#   ./launch.sh train 16 1T                   # batch 512 -> 2560 sequences over the first 30B tokens
#   GBS_RAMP=none GBS=1024 ./launch.sh train 16 500B   # constant global batch size
#   DRYRUN=1 ./launch.sh train 16 1T          # resolve + print the config (1-node debug job)
#   SUBMIT=0 ./launch.sh train 16 1T          # only write logs/<job>.sbatch
#
# Knobs (environment variables, [default]):
#   MODEL [35b-a3b]     35b-a3b: Qwen3.5-35B-A3B (MoE, the official recipe) | 0.8b | 2b | 4b | 9b: Qwen3.5
#                       dense (hybrid Gated DeltaNet / gated attention, MTP; recipe = NVIDIA's GB200
#                       Qwen3.5-9B text recipe built from the size's own HF config, train/pretrain.py).
#                       Dense defaults: EP=1, PP=1 (pure data parallelism), no MoE knobs; run names
#                       start with q35-<size> instead of q35a3b. GLOBAL_DP and every OPTIMIZER work.
#                       W&B project: qwen35-<MODEL>-cscs (WANDB_PROJECT_FIXED=1: the one in config.sh)
#   EP [4] TP [1] PP [2] CP [1] ETP [1]   parallel sizes; EP=4 keeps the expert all-to-all
#                                         inside a node (NVLink), only DP/PP traffic crosses Slingshot;
#                                         PP=2 halves the static memory (PP=1 does not fit, WORKLOG.md).
#                                         A train run may continue on another node count (data-parallel
#                                         size); the layout itself is fixed once checkpoints exist
#   VPP [none]          virtual pipeline stages per PP rank (interleaved 1F1B: bubble / VPP)
#   PP_LAYOUT [PP=2: Et*21|t*19mL]  uneven pipeline split (Megatron layout string: E embedding,
#                       t layer, m MTP, L loss, stages split by |, PP*VPP stages); the last stage also
#                       carries both 248k-vocab output heads and the MTP layer. "" = even split
#   GLOBAL_DP [1]       >1 = split the nodes into GLOBAL_DP "datacenters", each a full model replica
#                       (inter-datacenter training): pipeline, expert all-to-all and the gradient
#                       reduce-scatter / parameter all-gather stay inside a datacenter, and once per
#                       step the gradient shards are all-reduced between datacenters
#                       (ddp.num_distributed_optimizer_instances). With PP=2 on 8 nodes and GLOBAL_DP=2:
#                       nodes 1,2,5,6 | 3,4,7,8. Each datacenter keeps its own optimizer state, so that
#                       state takes GLOBAL_DP x the memory (~+13 GB/GPU on stage 0 at 8 nodes)
#   EP_OVERLAP [0]      1 = overlap the MoE dispatch/combine of one microbatch with another's compute
#                       (comm_overlap.overlap_moe_expert_parallel_comm; needs VPP when PP > 1)
#   MBS [1] GBS [512] SEQ_LEN [4096]      micro/global batch, sequence length (recipe values)
#   GBS_RAMP [train: 512:2560:4:3%, else none]   START:FINAL:STEPS:LENGTH: the global batch grows
#                       from START to FINAL sequences in STEPS equal steps over the first LENGTH
#                       tokens (or % of the run), like DeepSeek-V2/V3; none = constant GBS.
#                       Each step ends the training process at a checkpoint and restarts it in the
#                       same job (CUDA graphs are captured per microbatch count).
#   GBS_SCHEDULE        explicit schedule instead: "TOKENS:BATCH TOKENS:BATCH ..." (first at 0)
#   WARMUP_TOKENS [1.05B]  LR warmup with a batch schedule (= the recipe's 500 steps x 512 x 4096)
#   GRAD_FP32 [0]       1 = fp32 gradient reduction as in the GB200 recipe (+~20 GB/GPU at EP=4);
#                       0 = bf16, as NVIDIA's Hopper MoE recipes (Qwen3-30B-A3B H100) do
#   CUDA_GRAPHS [1]     0 = disable the recipe's TE-scoped CUDA graphs (attn, moe_router, moe_preprocess)
#   DISPATCHER [auto]   MoE token dispatcher: hybridep (the GB200 recipe's fused NVLink dispatcher,
#                       +10% at 8 nodes; only with the expert group inside a node, EP*ETP <= 4) |
#                       alltoall (NCCL) | auto = hybridep if EP*ETP <= 4, else alltoall
#   RECOMPUTE [none]    selective = recompute moe_act + layernorm (activation memory)
#   LR MIN_LR WARMUP    learning-rate schedule [recipe: 3e-4 -> 3e-5 cosine, 500 warmup steps;
#                       with a batch schedule the warmup is WARMUP_TOKENS and the cosine spans the run]
#   BLEND               blend JSON from data/build_blend.py [train: $DATA_ROOT/blends/lightning35_qwen35.json]
#   NETLOG [2]          0 off | 1 NIC + NVLink counters and per-step W&B net metrics
#                       | 2 + NCCL inspector, every collective and P2P op | 3 + per-op timestamps
#   NSYS [0]            1 = raw nsys timeline of NSYS_STEPS [10:12] on NSYS_RANKS [0]
#   SAVE_INTERVAL [train: 100]  checkpoint every N steps (~77 min at GBS 2560 on 8 nodes; a save blocks
#                       ~25-105 s, median ~50 s). Jobs on Clariden get killed by node failures every few
#                       hours to days, and all steps since the last checkpoint are lost (WORKLOG.md)
#   EVAL_INTERVAL [500] EVAL_ITERS [16]
#   EXCLUDE_FLAKY [1]   1 = submit (and resubmit) with --exclude= the nodes that recently dropped out
#                       of Slurm ("Not responding", which kills the job): tools/flaky_nodes.sh
#   CRASH_RETRIES [2]   train mode: after a crashed job (NCCL timeout, GPU error, ...) resubmit from the last
#                       checkpoint, at most this many times without a new checkpoint in between
#   WALLTIME PARTITION MAX_RESUBMITS [200] EXP_SUFFIX
#   OPTIMIZER [adam]    adam (recipe: distributed AdamW, ZeRO-1) | muon: Megatron-Core's layer-wise
#                       distributed Muon (dist_muon; Muon for weight matrices, Adam for embeddings,
#                       norms, router) with Kimi-K2's settings (momentum 0.95, Nesterov, spectral scaling,
#                       5 Newton-Schulz steps); same LR schedule, weight decay and bf16 gradient reduction
#                       as adam. Muon cannot use ZeRO-1, so it also turns off ddp/optimizer
#                       use_distributed_optimizer and overlap_param_gather. Run name "-muon"
#                       | muon-zero1: the same Muon with ZeRO-1 (DeepSeek-V4 style): gradients
#                       reduce-scattered into shard-aligned buffers so each rank holds whole Muon
#                       matrices, non-Muon params in a regular distributed optimizer. Applies the
#                       optional patch set muon-zero1 (which also supports GLOBAL_DP>1). Run name
#                       "-muon-zero1"
#                       | dion-zero1: Dion (Ahn et al. 2025; patch set dion, own implementation) for the
#                       weight matrices on the same ZeRO-1 layout, Adam for the rest. With GLOBAL_DP>1
#                       the matrix gradients are not all-reduced between datacenters: each keeps its
#                       own Dion momentum and only the rank-r factors (m+n)r per matrix cross, exactly
#                       equivalent to Dion on the global gradient. Update scaled like Muon (full rank
#                       = Muon's update size). Run name "-dion-zero1-r<fraction>"
#   DION_RANK_FRACTION [0.25]  Dion rank r = fraction x min(rows, cols) of each matrix
#   DION_COMM_DTYPE [bf16]     bf16 | fp32: dtype of the factors summed across datacenters
#   DION_SYNC_CHECK [0]        every N steps compare the Dion matrices across datacenters (log)
#   DION_SCALE [1]             multiply Dion's update (e.g. 2 = Muon's update RMS at r=1/4); run name "-x<s>"
#   MUON_SPECTRAL [ns]  muon / muon-zero1 only (patch set muon-spectral): how the momentum becomes the
#                       update. ns = Newton-Schulz (Muon); with an exact SVD, rescaled to Muon's update
#                       RMS: polar (all singular values 1), power:<p> (singular values ** p; 1 = normalized
#                       gradient), top:<f> (top f of the singular directions only, the rest dropped).
#                       Run name "-spec-<mode><param>"
#   FINAL_EVAL [1; 0 with MAX_STEPS]  1 = Bridge's validation + test passes after the run's last step
#                       (a MAX_STEPS run already evaluates at its last step with EVAL_INTERVAL=MAX_STEPS)
#   MAX_STEPS           stop the run at this step while the batch-size schedule and the LR schedule stay
#                       those of the full token budget (a short run identical to the start of the long
#                       one); train runs get "-max<N>" in their name, so they never share the full run's
#                       checkpoints
#   PATCHES             optional patch sets for this job only, comma-separated:
#                       patches/optional/<set>/{megatron-bridge,megatron-lm}/*.patch, applied after the
#                       always-on patches. Each combination gets its own code snapshot and the run its own
#                       name ("-p<set>"), so jobs with different sets can be queued together
#   EXTRA               more KEY=VALUE overrides, e.g. EXTRA="optimizer.weight_decay=0.05"
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$WORKDIR/config.sh" ] || { echo "copy config.sh.example to config.sh first"; exit 1; }
source "$WORKDIR/config.sh"
source "$WORKDIR/env/versions.env"

USAGE="Usage: ./launch.sh <smoke|throughput|train> [nodes] [steps]"
MODE=${1:?$USAGE}

MODEL=${MODEL:-35b-a3b}
case $MODEL in
    35b-a3b) MOE=1; MODEL_LAYERS=40; MTAG=q35a3b; DEFAULT_RECIPE=qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config ;;
    0.8b|2b) MOE=0; MODEL_LAYERS=24 ;;
    4b|9b)   MOE=0; MODEL_LAYERS=32 ;;
    *) echo "MODEL must be 35b-a3b, 0.8b, 2b, 4b or 9b"; exit 1 ;;
esac
if [ "$MOE" = 0 ]; then
    MTAG=q35-$MODEL; DEFAULT_RECIPE="qwen35_text_dense:${MODEL^^}"
fi
# One W&B project per model: qwen35-<MODEL>-cscs (qwen35-35b-a3b-cscs, qwen35-9b-cscs, ...), so runs of
# different models never share charts. WANDB_PROJECT_FIXED=1 keeps WANDB_PROJECT from config.sh.
[ "${WANDB_PROJECT_FIXED:-0}" = 1 ] || WANDB_PROJECT="qwen35-${MODEL}-cscs"

model_layers() {  # transformer layers of this run (the smoke model has 4)
    if [ "$MODE" = smoke ]; then echo 4; else echo "$MODEL_LAYERS"; fi
}

hms_to_mins() {
    local h m s
    IFS=: read -r h m s <<< "$1"
    echo $((10#$h * 60 + 10#$m + (10#${s:-0} + 59) / 60))
}

if [ "$MOE" = 1 ]; then EP=${EP:-4}; PP=${PP:-2}; else EP=${EP:-1}; PP=${PP:-1}; fi
TP=${TP:-1}; CP=${CP:-1}; ETP=${ETP:-1}
MBS=${MBS:-1}; SEQ_LEN=${SEQ_LEN:-4096}
GRAD_FP32=${GRAD_FP32:-0}; CUDA_GRAPHS=${CUDA_GRAPHS:-1}; RECOMPUTE=${RECOMPUTE:-none}
DISPATCHER=${DISPATCHER:-auto}
EXCLUDE_FLAKY=${EXCLUDE_FLAKY:-1}
PATCHES=${PATCHES:-}
GLOBAL_DP=${GLOBAL_DP:-1}
OPTIMIZER=${OPTIMIZER:-adam}
case $OPTIMIZER in adam|muon|muon-zero1|dion-zero1) ;; *) echo "OPTIMIZER must be adam, muon, muon-zero1 or dion-zero1"; exit 1 ;; esac
IMPLIED_SETS=""   # optional patch sets an OPTIMIZER needs (in this order, not repeated in the run name)
case $OPTIMIZER in muon-zero1) IMPLIED_SETS="muon-zero1" ;; dion-zero1) IMPLIED_SETS="muon-zero1 dion" ;; esac
MUON_SPECTRAL=${MUON_SPECTRAL:-ns}
case $MUON_SPECTRAL in
    ns) SPEC_MODE=ns; SPEC_PARAM=0 ;;
    polar) SPEC_MODE=polar; SPEC_PARAM=0 ;;
    power:*|top:*) SPEC_MODE=${MUON_SPECTRAL%%:*}; SPEC_PARAM=${MUON_SPECTRAL#*:} ;;
    *) echo "MUON_SPECTRAL must be ns, polar, power:<p> or top:<f>"; exit 1 ;;
esac
if [ "$SPEC_MODE" != ns ]; then
    case $OPTIMIZER in muon|muon-zero1) ;; *) echo "MUON_SPECTRAL needs OPTIMIZER=muon or muon-zero1"; exit 1 ;; esac
    IMPLIED_SETS="$IMPLIED_SETS muon-spectral"
fi
DION_SCALE=${DION_SCALE:-1}
for set in $IMPLIED_SETS; do
    [[ ",$PATCHES," == *,$set,* ]] || PATCHES=${PATCHES:+$PATCHES,}$set
done
DION_RANK_FRACTION=${DION_RANK_FRACTION:-0.25}
DION_COMM_DTYPE=${DION_COMM_DTYPE:-bf16}
DION_SYNC_CHECK=${DION_SYNC_CHECK:-0}
case $DION_COMM_DTYPE in bf16|fp32) ;; *) echo "DION_COMM_DTYPE must be bf16 or fp32"; exit 1 ;; esac
for set in ${PATCHES//,/ }; do
    [ -d "$WORKDIR/patches/optional/$set" ] || { echo "PATCHES: no patch set patches/optional/$set"; exit 1; }
done
VPP=${VPP:-}; EP_OVERLAP=${EP_OVERLAP:-0}
# Default uneven split for PP=2: the last stage also runs both 248k-vocab output heads and the MTP
# layer. On real data 21/19 is as fast as 23/17 (the mock-data optimum, which assumes balanced
# expert routing) and leaves the fullest GPU 11 GB of HBM instead of 6.6 GB (WORKLOG.md).
# PP_LAYOUT="" keeps Megatron's even split.
if [ -z "${PP_LAYOUT+set}" ] && [ "$PP" = 2 ] && [ -z "$VPP" ] && [ "$MOE" = 1 ]; then
    if [ "$MODE" = smoke ]; then PP_LAYOUT="Et*3|tmL"; else PP_LAYOUT="Et*21|t*19mL"; fi
fi
PP_LAYOUT=${PP_LAYOUT:-}
if [ "$MOE" = 0 ]; then
    { [ "$DISPATCHER" = auto ] || [ "$DISPATCHER" = none ]; } || { echo "DISPATCHER: MODEL=$MODEL has no experts"; exit 1; }
    (( EP == 1 && ETP == 1 )) || { echo "EP/ETP: MODEL=$MODEL has no experts"; exit 1; }
    [ "$EP_OVERLAP" = 0 ] || { echo "EP_OVERLAP: MODEL=$MODEL has no experts"; exit 1; }
    DISPATCHER=none
elif [ "$DISPATCHER" = auto ]; then
    if (( EP * ETP <= 4 )); then DISPATCHER=hybridep; else DISPATCHER=alltoall; fi
fi
NSYS_STEPS=${NSYS_STEPS:-10:12}; NSYS_RANKS=${NSYS_RANKS:-0}; NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
EVAL_ITERS=${EVAL_ITERS:-16}   # 16 x GBS 2560 = 168M tokens per evaluation (~1% of the run time at 8 nodes)
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
        OVERRIDES+=(model.num_layers=4 model.linear_attention_freq=4
                    validation.eval_interval=10 validation.eval_iters=2)
        [ "$MOE" = 1 ] && OVERRIDES+=(model.num_moe_experts=32)
        WARMUP=${WARMUP:-5}; SAVE_INTERVAL=${SAVE_INTERVAL:-15}
        CKPT=save; RESUBMIT=false; MAX_RESUBMITS=0
        ;;
    throughput)
        NODES=${2:-8}; STEPS=${3:-30}
        GBS=${GBS:-512}
        PARTITION=${PARTITION:-normal}; WALLTIME=${WALLTIME:-01:00:00}
        NETLOG=${NETLOG:-2}; NSYS=${NSYS:-0}
        # No evaluation at all: Bridge would otherwise run the full validation + test sets after the
        # last step (32 iterations each at the global batch size, ~3 min each at GBS 512 on 8 nodes).
        OVERRIDES+=(validation.eval_interval=null validation.eval_iters=0 scheduler.lr_warmup_iters=10)
        CKPT=none; RESUBMIT=false; MAX_RESUBMITS=0
        ;;
    train)
        NODES=${2:?$USAGE}; BUDGET=${3:?$USAGE}
        if [[ $BUDGET =~ ^[0-9]+$ ]]; then STEPS=$BUDGET; else TRAIN_TOKENS_ARG=$BUDGET; STEPS=0; fi
        GBS=${GBS:-512}; GBS_RAMP=${GBS_RAMP:-512:2560:4:3%}
        PARTITION=${PARTITION:-normal}; WALLTIME=${WALLTIME:-12:00:00}
        NETLOG=${NETLOG:-2}; NSYS=${NSYS:-0}
        BLEND=${BLEND:-$DATA_ROOT/blends/lightning35_qwen35.json}
        OVERRIDES+=(validation.eval_interval=${EVAL_INTERVAL:-500} validation.eval_iters=$EVAL_ITERS)
        CKPT=save_load; RESUBMIT=true; MAX_RESUBMITS=${MAX_RESUBMITS:-200}
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

################ Global batch size schedule ################
# Megatron-Core's step_batch_size_schedule (patches/megatron-bridge/0001-*): tools/batch_schedule.py
# turns the ramp into the schedule string, the iteration count for a token budget and LR
# warmup/decay lengths in final-size batches (Bridge counts the LR schedule in samples).
BATCH_SCHEDULE=0
GBS_RAMP=${GBS_RAMP:-none}
if [ -n "${GBS_SCHEDULE:-}" ] || [ "$GBS_RAMP" != none ] || [ -n "${TRAIN_TOKENS_ARG:-}" ]; then
    if [ -n "${GBS_SCHEDULE:-}" ]; then SCHED_ARG=(--schedule "$GBS_SCHEDULE")
    elif [ "$GBS_RAMP" != none ]; then SCHED_ARG=(--ramp "$GBS_RAMP")
    else SCHED_ARG=(--schedule "0:$GBS"); fi   # constant batch, token budget
    if [ -n "${TRAIN_TOKENS_ARG:-}" ]; then BUDGET_ARG=(--train-tokens "$TRAIN_TOKENS_ARG"); else BUDGET_ARG=(--train-iters "$STEPS"); fi
    SCHED_OUT=$(python3.11 "$WORKDIR/tools/batch_schedule.py" --seq-length "$SEQ_LEN" "${SCHED_ARG[@]}" \
        "${BUDGET_ARG[@]}" --warmup-tokens "$([ "$MODE" = train ] && echo "${WARMUP_TOKENS:-1048576000}" || echo 0)" \
        --divisor $((MBS * DP))) || exit 1
    eval "$SCHED_OUT"
    STEPS=$TRAIN_ITERS
    if [ -n "${GBS_SCHEDULE:-}" ] || [ "$GBS_RAMP" != none ]; then
        [ "$MODE" = throughput ] && { echo "batch-size schedules restart from checkpoints: smoke or train mode only"; exit 1; }
        [ -n "${WARMUP:-}" ] && [ "$MODE" = train ] && { echo "with a batch-size schedule set WARMUP_TOKENS, not WARMUP"; exit 1; }
        BATCH_SCHEDULE=1; GBS=$FINAL_GBS
        GBS_FIRST=${SCHEDULE#*:}; GBS_FIRST=${GBS_FIRST%% *}
        OVERRIDES+=("train.step_batch_size_schedule='$SCHEDULE'" scheduler.lr_decay_iters=$LR_DECAY_ITERS)
        [ "$MODE" = train ] && WARMUP=$LR_WARMUP_ITERS
        CKPT=save_load   # every batch-size step resumes from the checkpoint written at the step
    fi
fi
if [ -n "${MAX_STEPS:-}" ]; then
    (( MAX_STEPS > 0 && MAX_STEPS < STEPS )) || { echo "MAX_STEPS=$MAX_STEPS must be between 1 and the run's $STEPS steps"; exit 1; }
    # Keep the full run's LR decay length (with a batch schedule it is already set explicitly).
    [ "$BATCH_SCHEDULE" = 1 ] || OVERRIDES+=(scheduler.lr_decay_iters=${LR_DECAY_ITERS:-$STEPS})
    echo "MAX_STEPS: stopping at step $MAX_STEPS of $STEPS (schedules unchanged)"
    STEPS=$MAX_STEPS
fi
if (( GBS % (MBS * DP) != 0 )); then echo "GBS=$GBS must be a multiple of MBS*DP=$((MBS * DP))"; exit 1; fi
if [ "$OPTIMIZER" != adam ]; then
    [ "$OPTIMIZER" = muon ] && (( GLOBAL_DP > 1 )) && { echo "OPTIMIZER=muon with GLOBAL_DP>1: Megatron-Core's legacy layer-wise Muon has no optimizer instances; use OPTIMIZER=muon-zero1"; exit 1; }
    if [ "$OPTIMIZER" = dion-zero1 ]; then OPT_NAME=dist_dion; else OPT_NAME=dist_muon; fi
    OVERRIDES+=(optimizer.optimizer=$OPT_NAME optimizer.muon_momentum=0.95 optimizer.muon_nesterov=true
                optimizer.muon_scale_mode=spectral optimizer.muon_num_ns_steps=5 optimizer.muon_scalar_optimizer=adam
                ddp.use_distributed_optimizer=false optimizer.use_distributed_optimizer=false
                ddp.overlap_param_gather=false optimizer.overlap_param_gather=false
                comm_overlap.overlap_param_gather=false)
    [ "$OPTIMIZER" = muon ] || OVERRIDES+=(ddp.use_layer_wise_param_layout=true)
    [ "$SPEC_MODE" = ns ] || OVERRIDES+=(optimizer.muon_spectral_mode=$SPEC_MODE optimizer.muon_spectral_param=$SPEC_PARAM)
    if [ "$OPTIMIZER" = dion-zero1 ]; then
        (( TP == 1 && ETP == 1 )) || { echo "OPTIMIZER=dion-zero1 needs TP=1 and ETP=1 (whole matrices per rank)"; exit 1; }
        OVERRIDES+=(optimizer.muon_extra_scale_factor=$DION_SCALE)
        OVERRIDES+=(optimizer.dion_rank_fraction=$DION_RANK_FRACTION optimizer.dion_mu=0.95
                    optimizer.dion_comm_dtype=$DION_COMM_DTYPE optimizer.dion_sync_check_interval=$DION_SYNC_CHECK)
        (( GLOBAL_DP > 1 )) && OVERRIDES+=(ddp.layer_wise_skip_inter_instance_grad_sync=true)
    fi
fi
if (( GLOBAL_DP > 1 )); then
    EXPT_DP=$(( NODES * 4 / (EP * ETP * PP) ))
    (( DP % GLOBAL_DP == 0 && EXPT_DP % GLOBAL_DP == 0 )) \
        || { echo "GLOBAL_DP=$GLOBAL_DP must divide DP=$DP and expert DP=$EXPT_DP"; exit 1; }
    OVERRIDES+=(ddp.num_distributed_optimizer_instances=$GLOBAL_DP)
fi
if [ -n "$VPP" ]; then
    (( PP > 1 )) || { echo "VPP needs PP > 1"; exit 1; }
    (( $(model_layers) % (PP * VPP) == 0 )) || { echo "layers ($(model_layers)) not divisible by PP*VPP=$((PP * VPP))"; exit 1; }
fi
if [ "$EP_OVERLAP" = 1 ] && (( PP > 1 )) && [ -z "$VPP" ]; then echo "EP_OVERLAP=1 with PP > 1 needs VPP"; exit 1; fi
if [ -n "$PP_LAYOUT" ]; then
    LAYOUT_CHECK=$(python3.11 - "$PP_LAYOUT" "$PP" "${VPP:-1}" "$(model_layers)" <<'PYCHK'
import re, sys
layout, pp, vpp, layers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def expand(text):  # same grammar as Megatron's PipelineParallelLayerLayout: (x)*n, x*n
    while "(" in text:
        text = re.sub(r"\(([^()]*)\)\*(\d+)", lambda m: m.group(1) * int(m.group(2)), text)
    return re.sub(r"([EtmL])\*(\d+)", lambda m: m.group(1) * int(m.group(2)), text.replace(",", ""))
stages = expand(layout).split("|")
t = sum(st.count("t") for st in stages)
ok = len(stages) == pp * vpp and t == layers and stages[0].startswith("E") and stages[-1].endswith("L")
print("ok" if ok else f"layout has {len(stages)} stages / {t} layers; need PP*VPP={pp*vpp} stages, {layers} layers, E first, L last")
PYCHK
)
    [ "$LAYOUT_CHECK" = ok ] || { echo "PP_LAYOUT: $LAYOUT_CHECK"; exit 1; }
fi
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
[ -n "$VPP" ] && OVERRIDES+=(model.virtual_pipeline_model_parallel_size=$VPP)
[ -n "$PP_LAYOUT" ] && OVERRIDES+=("model.pipeline_model_parallel_layout='$PP_LAYOUT'")
[ "$EP_OVERLAP" = 1 ] && OVERRIDES+=(comm_overlap.overlap_moe_expert_parallel_comm=true)
case $DISPATCHER in
    alltoall|none) ;;
    hybridep)
        # HybridEP's NVLink path only: across nodes it needs InfiniBand GPU-initiated RDMA.
        # Expert groups are consecutive ranks (ETP innermost), so EP*ETP <= 4 stays in a node.
        (( EP * ETP <= 4 )) || { echo "DISPATCHER=hybridep needs EP*ETP <= 4 (expert group inside one node)"; exit 1; }
        OVERRIDES+=(model.moe_token_dispatcher_type=flex model.moe_flex_dispatcher_backend=hybridep) ;;
    *) echo "DISPATCHER must be alltoall or hybridep"; exit 1 ;;
esac
case $RECOMPUTE in
    none) ;;
    selective)
        if [ "$MOE" = 1 ]; then MODS="[moe_act,layernorm]"; else MODS="[mlp,layernorm]"; fi
        OVERRIDES+=(model.recompute_granularity=selective "model.recompute_modules=$MODS") ;;
    *) echo "RECOMPUTE must be none or selective"; exit 1 ;;
esac
[ -n "${LR:-}" ] && OVERRIDES+=(optimizer.lr=$LR)
[ -n "${MIN_LR:-}" ] && OVERRIDES+=(optimizer.min_lr=$MIN_LR)
[ -n "${WARMUP:-}" ] && OVERRIDES+=(scheduler.lr_warmup_iters=$WARMUP)
case $CKPT in
    none) OVERRIDES+=(checkpoint.save=null checkpoint.load=null) ;;
    save) OVERRIDES+=(checkpoint.save=RUN_DIR/checkpoints checkpoint.load=null checkpoint.save_interval=$SAVE_INTERVAL) ;;
    save_load)
        EXIT_MINS=$(( $(hms_to_mins "$WALLTIME") - 15 ))
        OVERRIDES+=(checkpoint.save=RUN_DIR/checkpoints checkpoint.load=RUN_DIR/checkpoints
                    checkpoint.save_interval=${SAVE_INTERVAL:-100} train.exit_duration_in_mins=$EXIT_MINS) ;;
esac
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ] && OVERRIDES+=($EXTRA)

################ Names and paths ################
TAG="${MTAG}-${NODES}n"
[ "$MOE" = 1 ] && TAG+="-ep${EP}"
(( TP > 1 )) && TAG+="-tp$TP"; (( PP > 1 )) && TAG+="-pp$PP"; [ -n "$VPP" ] && TAG+="-vpp$VPP"; (( CP > 1 )) && TAG+="-cp$CP"
[ -n "$PP_LAYOUT" ] && TAG+="-lay$(echo "$PP_LAYOUT" | tr -cd '0-9|' | tr '|' 'x')"
TAG+="-mbs${MBS}-gbs${GBS_FIRST:+${GBS_FIRST}to}${GBS}-sl${SEQ_LEN}"
[ "$GRAD_FP32" = 1 ] && TAG+="-gf32"; [ "$CUDA_GRAPHS" = 0 ] && TAG+="-nocg"; [ "$RECOMPUTE" != none ] && TAG+="-rc"
[ "$DISPATCHER" = hybridep ] && TAG+="-$DISPATCHER"; [ "$EP_OVERLAP" = 1 ] && TAG+="-epov"
GDP_TAG=""; (( GLOBAL_DP > 1 )) && GDP_TAG="-gdp$GLOBAL_DP"
OPT_TAG=""; [ "$OPTIMIZER" != adam ] && OPT_TAG="-$OPTIMIZER"
if [ "$OPTIMIZER" = dion-zero1 ]; then
    OPT_TAG+="-r$DION_RANK_FRACTION"
    if [ "$DION_COMM_DTYPE" = fp32 ]; then OPT_TAG+="-fp32comm"; fi
    if [ "$DION_SCALE" != 1 ]; then OPT_TAG+="-x$DION_SCALE"; fi
fi
if [ "$SPEC_MODE" != ns ]; then
    OPT_TAG+="-spec-$SPEC_MODE"
    if [ "$SPEC_MODE" != polar ]; then OPT_TAG+="$SPEC_PARAM"; fi
fi
PTAG=""   # optional patch sets in the run name, except those the OPTIMIZER implies
for set in ${PATCHES//,/ }; do
    [[ " $IMPLIED_SETS " == *" $set "* ]] && continue
    PTAG+="${PTAG:++}$set"
done
RUN_TAGS="${OPT_TAG}${GDP_TAG}${MAX_STEPS:+-max$MAX_STEPS}${PTAG:+-p$PTAG}"
if [ "$MODE" = train ]; then
    # Stable across resubmissions and node counts: torch_dist checkpoints reshard on load
    # across data-parallel sizes, so only data-semantic settings are in the name. The
    # optimizer state is saved DP-reshardable (Megatron default), so TP/PP/CP/EP/ETP/VPP must
    # stay fixed for the life of a run (checked below) unless it was saved with
    # checkpoint.dist_ckpt_optim_fully_reshardable=true.
    EXP_NAME="train-${MTAG}-gbs${GBS_FIRST:+${GBS_FIRST}to}${GBS}-sl${SEQ_LEN}${RUN_TAGS}${EXP_SUFFIX:+-$EXP_SUFFIX}"
else
    EXP_NAME="${MODE}-${TAG}${RUN_TAGS}${EXP_SUFFIX:+-$EXP_SUFFIX}"
fi
# A dry run gets its own run directory: it must not create (and leave files in) the real one.
[ "${DRYRUN:-0}" = 1 ] && EXP_NAME+="-dryrun"
JOB_NAME="mbcscs-${EXP_NAME}"
RUN_DIR=$RUN_ROOT/$EXP_NAME
SCRIPT="$WORKDIR/logs/${JOB_NAME}.sbatch"
mkdir -p "$WORKDIR/logs"
if [ "$MODE" = train ] && [ "${DRYRUN:-0}" != 1 ]; then
    LAYOUT="TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP${VPP:+ VPP=$VPP}${PP_LAYOUT:+ LAYOUT=$PP_LAYOUT}${GDP_TAG:+ GLOBAL_DP=$GLOBAL_DP}${OPT_TAG:+ OPTIMIZER=$OPTIMIZER}"
    OLD_LAYOUT=$(cat "$RUN_DIR/layout" 2>/dev/null || true)
    if [ -s "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ] && [ -n "$OLD_LAYOUT" ] \
            && [ "$OLD_LAYOUT" != "$LAYOUT" ] && [ "${FORCE_LAYOUT:-0}" != 1 ]; then
        echo "$RUN_DIR has checkpoints written with $OLD_LAYOUT; requested $LAYOUT."
        echo "The optimizer state is only DP-reshardable: keep the layout (the node count may change),"
        echo "or FORCE_LAYOUT=1 if the checkpoints were saved with checkpoint.dist_ckpt_optim_fully_reshardable=true."
        exit 1
    fi
    # With EP > 1 Bridge saves the RNG state per data-parallel rank (a PP x TP x DP sharded object), so
    # a run continued on another node count cannot load it ("Missing key in checkpoint state_dict:
    # rng_state/...", job 3704823). Nothing in this model draws from it (dropout 0, no router jitter,
    # the data order comes from the samplers' own seeds): start the RNG from the seed instead.
    OLD_NODES=$(cat "$RUN_DIR/nodes" 2>/dev/null || true)
    if [ -s "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ] && [ "$OLD_NODES" != "$NODES" ]; then
        echo "node count $NODES, checkpoints from ${OLD_NODES:-an unknown node count}: checkpoint.load_rng=false"
        OVERRIDES+=(checkpoint.load_rng=false)
    fi
    [ "${SUBMIT:-1}" = 1 ] && mkdir -p "$RUN_DIR" && echo "$LAYOUT" > "$RUN_DIR/layout" && echo "$NODES" > "$RUN_DIR/nodes"
    # A launch by hand restarts the crash-retry budget of job_body.sh.
    [ "${SUBMIT:-1}" = 1 ] && rm -f "$RUN_DIR/logs/consecutive-failures"
fi

################ Snapshot of this repository's runtime files ################
# Jobs run train/, tools/, conf/ and patches/ from a content-addressed copy made at submission,
# never from the working tree: queued and resubmitted jobs run what was submitted, and editing a
# script cannot hit a running job (bash reads a script while executing it; an in-place edit of
# train/rank_wrapper.sh once made a running job start its training a second time).
SRC_FILES=$(cd "$WORKDIR" && find train tools conf patches -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort)
SRC=$MBCSCS_IOPS/code/src-$(cd "$WORKDIR" && echo "$SRC_FILES" | xargs sha256sum | sha256sum | cut -c1-12)
if [ ! -d "$SRC" ]; then
    mkdir -p "$MBCSCS_IOPS/code"
    SRC_TMP=$(mktemp -d "$MBCSCS_IOPS/code/.src-tmp-XXXXXX")
    (cd "$WORKDIR" && echo "$SRC_FILES" | tar -c -T -) | tar -x -C "$SRC_TMP"
    find "$SRC_TMP" -type f -exec chmod a-w {} +   # accidental edits fail loudly
    chmod 755 "$SRC_TMP"
    mv -T "$SRC_TMP" "$SRC" 2>/dev/null || rm -rf "$SRC_TMP"   # another launch made it first
fi

TRAIN_ARGS=(--recipe "${RECIPE:-$DEFAULT_RECIPE}"
            --profile "$SRC/conf/qwen35_35b_a3b_gh200.yaml")
[ -n "${BLEND:-}" ] && TRAIN_ARGS+=(--blend "$BLEND")
[ "${DRYRUN:-0}" = 1 ] && TRAIN_ARGS+=(--dry-run)
for o in "${OVERRIDES[@]}"; do TRAIN_ARGS+=("${o//RUN_DIR/$RUN_DIR}"); done

if [ "${DRYRUN:-0}" = 1 ]; then
    NODES=1; PARTITION=debug; WALLTIME=00:15:00; NETLOG=0; NSYS=0; RESUBMIT=false; SUMMARIZE=0
    SCRIPT="$WORKDIR/logs/${JOB_NAME}.sbatch"
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
#SBATCH $([ "$MODE" = train ] && echo --requeue || echo --no-requeue)
#SBATCH --open-mode=append
#
# Generated by launch.sh on $(date -Iseconds) -- self-contained, resubmittable as is.

################ Settings ################
WORKDIR=$(printf %q "$WORKDIR")
SRC=$(printf %q "$SRC")
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
EXCLUDE_FLAKY=$EXCLUDE_FLAKY
PATCHES=$(printf %q "$PATCHES")
CRASH_RETRIES=${CRASH_RETRIES:-2}
TRAIN_ITERS=$STEPS
BATCH_SCHEDULE=$BATCH_SCHEDULE
FINAL_EVAL=${FINAL_EVAL:-$([ -n "${MAX_STEPS:-}" ] && echo 0 || echo 1)}
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
echo "  scripts: $SRC"
echo "  layout : nodes=$NODES world=$WORLD TP=$TP PP=$PP CP=$CP EP=$EP DP=$DP MBS=$MBS GBS=$GBS seq=$SEQ_LEN"
echo "  W&B project: ${WANDB_PROJECT:-none}"
echo "  logging: NETLOG=$NETLOG NSYS=$NSYS W&B=$([ -n "${WANDB_API_KEY:-}" ] && echo "on${WANDB_MODE:+ ($WANDB_MODE)}" || { [ "${WANDB_MODE:-}" = offline ] && echo "offline (no key)" || echo off; })"
if [ "${SUBMIT:-1}" = 1 ]; then
    FLAKY=$([ "$EXCLUDE_FLAKY" = 1 ] && "$WORKDIR/tools/flaky_nodes.sh" || true)
    [ -n "$FLAKY" ] && echo "  exclude: $(scontrol show hostnames "$FLAKY" | wc -l) nodes that recently dropped out of Slurm"
    sbatch ${FLAKY:+--exclude="$FLAKY"} "$SCRIPT"
fi
