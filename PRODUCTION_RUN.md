# Production run: Qwen3.5-35B-A3B from scratch on Clariden

Written 2026-10-20 around 06:30 for your morning check. Every measurement behind it is in `WORKLOG.md`;
usage is in `README.md`.

## Status

| | |
|---|---|
| Launched | 06:01 with `cd /iopsstor/scratch/cscs/yqin/Megatron-Bridge-CSCS && source config.sh && ./launch.sh train 8 1T` |
| First job | **3705060**: 8 nodes (32 GH200), 12 h, started immediately. Each job submits the next one until 1T tokens (99,958 steps) |
| Run directory | `/capstor/scratch/cscs/yqin/mbcscs/runs/train-q35a3b-gbs512to2560-sl4096` |
| W&B | https://wandb.ai/yuyqin-eth-zurich/qwen35-35b-a3b-cscs/runs/train-q35a3b-gbs512to2560-sl4096-1792468915 (the same run continues across all jobs) |
| Code | commit `aa2c2bd` (local, not pushed); jobs run from the read-only snapshot `/iopsstor/scratch/cscs/yqin/mbcscs/code/src-9acae3305f4d` |
| First steps (06:13–06:19) | **healthy.** Loss 12.83 → 10.10 by step 25 and grad norm 17 → 2.4, in LR warmup. Step time 11.5–12 s at GBS 512 (122–128 TFLOP/s/GPU), as in the tests. Host memory flat: workers 5.3 GB, ≥ 344 GB available per node. GPU memory: max allocated 65.1 / 65.9 GB (stage 0 / 1), and every GPU keeps **13.5–15.6 GB HBM free** |
| First checkpoint | step 100 at 06:35, saved in 76 s. Afterwards host memory is as expected (+63 GB shared memory for the save staging, ≥ 261 GB still available per node); loss 6.2 at step 108 |

## What is being trained

- **Model**: Qwen3.5-35B-A3B, trained from random init with NVIDIA's official Megatron-Bridge recipe
  `qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config` (Megatron-Bridge v0.6.2). It has 40 layers
  (Gated DeltaNet and gated attention, 3:1), 256 experts with top-8 routing plus a shared expert, one MTP
  layer and a 248k vocabulary. Only platform settings deviate from the recipe (next section).
- **Data**: the 23-source Nemotron 3.5 Lightning open-data blend, tokenized with the Qwen3.5 tokenizer
  (846B unique tokens, 964 shards on iopsstor). Sources are mixed uniformly, as in NVIDIA's recipe (I
  checked their code). At 1T tokens most sources get ~0.95 epochs; small ones repeat: Wiki-Rewrite 5.5×,
  Scientific-Coding 34.7×.
- **Schedule**:
  - Global batch ramps DeepSeek-style: 512 → 1024 → 1536 → 2048 → 2560 sequences (2.1M → 10.5M tokens)
    at 0, 7.5, 15, 22.5 and 30B tokens. Sequence length 4096.
  - LR 3e-4, warmup over 1.05B tokens, cosine to 3e-5 at 1T. These are recipe values: AdamW β2 0.95,
    weight decay 0.033 via the scheduler (Bridge's pretraining default), bf16, global-batch load-balancing
    loss 1e-3.

## Configuration and why

| Setting | Value | Evidence (8 nodes unless noted) |
|---|---|---|
| Parallelism | EP 4 (inside a node), PP 2, TP 1, DP 16, MBS 1 | PP 1 does not fit (OOM). TP 2 shards only the 7% dense part and was 25–40% slower. MBS 2 does not fit at PP 2, and PP 4 + MBS 2 ran out of memory at CUDA-graph capture |
| Pipeline split | `Et*21\|t*19mL` (21 layers + embedding / 19 layers + MTP + loss) | Mock data favoured 23/17 (+12% over 20/20). On real data 21/19 is just as fast (18.67 vs 18.62 s/step at GBS 1024). Its fullest GPU keeps **11 GB HBM free vs 6.6 GB** with 23/17, which protects the busiest expert ranks from OOM (jobs 3704884–87). The split cannot change during a run |
| MoE dispatcher | HybridEP (NVLink, inside the node) | +10% over NCCL all-to-all |
| CUDA graphs | TE-scoped (attention, router, preprocess) | +33%; the Grace CPU cannot launch ~230k small kernels per step fast enough |
| Gradient reduction | bf16 | As NVIDIA's Hopper MoE recipe; fp32 would cost ~20 GB/GPU |
| Data loader | 2 workers per rank, `create_attention_mask: false` | Each of the 964 per-shard datasets cached a 16.7 MB mask in every worker, so host memory grew 5 GB/min until a rank was OOM-killed. Fixed, and verified flat for 47 min at full scale (job 3704806) |
| Checkpoints | every 100 steps (~77 min at the final batch size), keep the last 3 (469 GB each), parallel load | Saves block 23–106 s; loads take 3–6 min |
| Evaluation | every 500 steps, 16 iterations | ~1% of the time |
| Logging | W&B (loss, LR, throughput, `net/*` NIC traffic, `node/*` host and GPU memory); per-node NIC/NVLink counters every second; **NETLOG=2**: NCCL inspector records every collective and P2P operation | Inspector overhead 2.4%; ~4 GB compressed logs per 12-h job |
| Job chain | 12-h jobs that resubmit themselves (max 200). Requeue on node failure. Flaky nodes excluded at every submission. Crashed jobs retried from the last checkpoint, at most 2 times without progress | Each mechanism was tested in Slurm tonight (see below) |

## Expected duration and cost

- **Measured step times**: 10.4 s per step at GBS 512 and 46.1 s at GBS 2560 with mock data (balanced
  routing); 160 TFLOP/s/GPU at the final batch size.
- **Real data is slower**, and the slowdown depends on the routing state of the model. Step times so far
  run 1.1–1.3× the mock numbers: 21–25 s instead of ~19 s at GBS 1024.
- **Totals**, including eval, saves, ~20 min of restart per 12-h job and the inspector:

| Nodes | Wall time if jobs run back-to-back | Node-hours |
|---|---|---|
| 8 (current) | **55–71 days** | 10,500–13,700 (42–55k GPU-hours) |
| 16 (91% scaling efficiency measured, job 3702548) | **30–39 days** | 11,500–15,000 |
| 32 (~87% measured in steady iterations, job 3702549) | **16–21 days** | 12,100–15,700 |

- **Queue time comes on top.** The `normal` partition caps all users together at 400 nodes, and it is full
  (≈2,200 nodes pending). Our fairshare is 0.17. The first job started at once; later ones may wait.
- **Ramp phase** (first 30B tokens, until step 7,453): about 2 days at 8 nodes.

## How to monitor

```bash
squeue -u yqin                                    # the running job and its successor
cd /iopsstor/scratch/cscs/yqin/Megatron-Bridge-CSCS
L=$(ls -t logs/mbcscs-train-q35a3b-gbs512to2560-sl4096-*.log | head -1)
grep -a " iteration " $L | tail -3                # step, loss, step time, TFLOP/s
grep -aE "\[resubmit\]|\[batch-size step\]|Traceback|Out Of Memory|Node failure" $L
R=/capstor/scratch/cscs/yqin/mbcscs/runs/train-q35a3b-gbs512to2560-sl4096
cat $R/checkpoints/latest_checkpointed_iteration.txt
tail -1 $R/netlogs/job_*/steps/nid*.jsonl | grep -o '"host_mem_gb".*'   # per node: host memory, workers, GPU HBM
```

What healthy looks like:

- lm loss falls from ~12.8: about 10 after 20 steps, 7.5 after 50 steps (in tests).
- **Step time**: ~12 s at GBS 512, rising roughly in proportion to the batch.
- **Host memory**: `MemAvailable` stays above ~140 GB per node and `workers_private` stays flat (~16 GB).
- **GPU memory**: `gpu_hbm_min_free` (the node's fullest GPU) stays above ~5 GB.
- **W&B**: the same metrics, plus `node/*` memory for the last node.

## How to stop, resume or change

- **Stop for good**: `scancel <running job> <pending successor>`. Up to 100 steps are lost.
- **Resume**: rerun the launch command above. It continues from the last checkpoint with the same W&B run.
  It refuses a different TP/PP/EP/split once checkpoints exist.
- **Move to 16 nodes**: wait until the ramp ends (step 7,453, ~2 days) or do it any time. Cancel the
  pending successor (or the running job), then run `./launch.sh train 16 1T`. A different node count is
  supported: tested 2 → 4 nodes, job 3704855. The launcher then skips restoring the RNG state, which
  nothing in this model uses.
- **Automatic handling**:
  - A node failure requeues the job.
  - A crash resubmits from the last checkpoint, at most 2 times without a new checkpoint.
  - After that the chain stops. Fix the cause, then rerun the launch command (it resets the counter).

## Verified before launch (Slurm jobs, details in WORKLOG.md)

- **Memory at full scale with real data** (3704806, 47 min): batch-size step and in-job restart,
  checkpoint load, 2 evals, saves and a clean exit; host memory flat.
- **Job chain**:
  - Walltime exit followed by resubmission and resume, twice (3704360/3704705, 3704797/3704822). The
    second pair also covered resubmission with a freshly computed flaky-node list.
  - Resume on a different node count (3704855).
  - Crash retry, by killing the training step (3704910 → 3704933).
  - Node failures requeued and resumed (3703471, 3704232).
- **Loss continuity** across all resumes; no grad-norm spikes after resharding.
- **Throughput**:
  - Mock data: layout sweep (20/20 … 25/15).
  - Real data: 4-split A/B and a low-LR control showing the slower real-data steps come from the
    model's routing state, not from restarts.
  - Scaling 8 → 16 nodes.

## Risks and open decisions

1. **Node count — decided: 8 nodes** (2026-10-20, to save node-hours). Scaling measured with mock
   data at GBS 2560: 16 nodes 91–94% efficiency, 32 nodes ~87% (91% of 16), 64 nodes ~71%; each doubling
   costs node-hours. MBS 2 does not fit with this stack: ZeRO-1 + PP 2 runs out of activation memory, and
   ZeRO-2/3 (Megatron-FSDP) breaks with PP (shared MTP embedding) and with the TE CUDA graphs, while
   PP 1 + ZeRO-3 runs out of memory at MBS 2 (all 40 layers + both logit tensors on one GPU). Details:
   WORKLOG.md. Switching later still works (`./launch.sh train 16 1T`, tested).
2. **Clariden instability**: ~80 nodes keep dropping out of Slurm and kill jobs. They are excluded
   automatically (91% of such events avoided in a backtest), but expect occasional requeues. Each one
   loses ≤ 100 steps.
3. **Scratch cleanup**: iopsstor deletes files not read for 14 days, capstor for 30. The running chain
   keeps the data alive, but a pause of more than 14 days deletes the tokenized data (re-tokenize:
   `sbatch data/tokenize.sbatch`). Old network logs, W&B and TensorBoard files are deleted after 30
   days; copy what you want to keep. `prep23` has no permanent Store allocation.
4. **Data repetition** from the uniform Lightning weights (Scientific-Coding 34.7 epochs at 1T). Change
   per-source weights with `data/build_blend.py --source-weights` before the run gets far; this would
   need a new run, or accepting a mid-run blend change.
5. **Speed on real data** is 10–30% below the mock-data numbers while expert routing is uneven. The
   recipe's global-batch balancing loss (1e-3) is kept as published. Aux-loss-free expert-bias
   balancing (DeepSeek-V3 style) would likely be faster, but it changes the training recipe.
6. **NETLOG=2** costs 2.4% (~1.5 days over the run) for full per-operation network logs. `NETLOG=1`
   keeps the NIC/NVLink counters and per-step W&B traffic at no measurable cost. It can be changed at
   any relaunch.
7. **Nothing is pushed to GitHub**: all commits are local on `main`, and the remote has no `main` branch yet.
   The repository lives on iopsstor scratch, which is subject to the 14-day cleanup: push it (your call),
   or keep a copy somewhere permanent.
