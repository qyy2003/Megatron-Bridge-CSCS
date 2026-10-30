# Megatron-Bridge-CSCS

> **Start with [HANDOVER.md](HANDOVER.md)**: current state, where everything lives, how to restart on a new account, and the experiment in progress.

Pretrain **Qwen3.5/3.6-35B-A3B from scratch** on CSCS Alps (Clariden, 4× GH200 per node,
Slingshot-11) with **Megatron-Bridge**, on the **Nemotron 3.5 Lightning open-data blend**,
with W&B loss logging and a record of the network traffic.

- Model: the official recipe `qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config`
  (text-only Qwen3.5-35B-A3B: 40 layers = 10 × [3 Gated DeltaNet + 1 gated attention],
  256 experts top-8 + shared expert, 1 MTP layer, vocab 248k). Qwen3.6-35B-A3B has the same
  architecture (identical text config), so from scratch the two are the same model.
- Data: the 23-source blend of NVIDIA's Nemotron 3.5 Lightning stage-0 recipe
  (Nemotron-CC-v2/v2.1, CC-Math-v1, Pretraining-Specialized-v1), read from the Swiss AI mirror
  on capstor, tokenized with the Qwen3.5 tokenizer.
- Code management as in the earlier Megatron-LM project: upstream as pinned submodules,
  every local change a self-documenting patch, `launch.sh` writes a self-contained sbatch.
- **Production run** (started 2026-10-20, 8 nodes, 1T tokens): status, configuration and how to
  monitor / stop / resume / rescale in [`PRODUCTION_RUN.md`](PRODUCTION_RUN.md); the measurements
  behind every choice in [`WORKLOG.md`](WORKLOG.md).

## Pinned versions (stable releases only) — `env/versions.env`

| Component | Version |
|---|---|
| Megatron-Bridge | `v0.6.2` (submodule) → Megatron-LM `a0f793d` (nested submodule) |
| Container | `nvcr.io/nvidia/nemo:26.08.01` (the release that ships Bridge v0.6.2: CUDA 13.3, PyTorch 2.13, TE 2.18) |
| Network stack | CSCS Alps extended-image recipe (`eth-cscs/alps-extended-images@6120245`): libfabric 2.6.0 (CXI), NCCL 2.30.7-1 + NCCL inspector plugin, aws-ofi-nccl v1.20.0, NVSHMEM 3.6.5, OpenMPI 5.0.10 |
| Data recipe | `NVIDIA-NeMo/Nemotron` `v0.2.0` (`lightning35/stage0_pretrain`) |
| Tokenizer/config | `Qwen/Qwen3.5-35B-A3B-Base` @ `0f08130` |

The image is built on Clariden with CSCS's own Alps recipe (CSCS publishes Alps images only up
to `nemo:26.02`, which is too old for the Qwen3.5 recipe). Clariden's driver is 595.91
(CUDA 13.2); the CUDA 13.3 container runs through CUDA minor-version compatibility (verified:
training runs). The image's Bridge (`r0.6.0@2a8fcdd`) differs from `v0.6.2` only by 4
docs/version commits; jobs mount our pinned snapshot over it anyway.

`env/build_image.sbatch` does three things on top of CSCS's recipe, each for a reason found
while building (details in `WORKLOG.md`): (0) runs CSCS's purge step as a layer of its own —
rootless podman on Alps cannot re-create, in the same step, a directory deleted from a lower
layer, which broke the NCCL `.deb` install; (1) the Alps stack; (2) `chmod -R a+rX` on
`/opt/Megatron-Bridge` and `/workspace` — NeMo 26.08.01 ships directories with mode 644,
which non-root users (the Container Engine runs as you) cannot traverse.

Measured on Clariden with this image (`tools/test-infra.sbatch`, 2 nodes, large messages):
all-reduce 335 GB/s busbw in a node (NVLink), 158 GB/s across nodes (Slingshot); all-to-all
334 GB/s in a node vs 40 GB/s across nodes — the reason the experts stay inside a node.

## Layout

```
config.sh.example        -> config.sh (account, W&B key, storage paths; git-ignored)
env/versions.env         pinned versions
env/build_image.sbatch   builds + imports the container (squashfs)
Megatron-Bridge/         submodule, v0.6.2 (+ 3rdparty/Megatron-LM)
patches/{megatron-bridge,megatron-lm}/   local changes as patches (see How_To_Edit.md)
code_store.sh / code_restore.sh          save WIP as patch / apply patches to the trees
conf/qwen35_35b_a3b_gh200.yaml           GH200/Slingshot deviations from the GB200 recipe
train/pretrain.py        entry point: official recipe + conf + blend + overrides (+ callbacks)
train/rank_wrapper.sh    per-rank NUMA binding, netmon, nsys
train/job_body.sh        body of every generated sbatch
launch.sh                smoke | throughput | train -> logs/<job>.sbatch -> sbatch
data/                    Lightning blend: sources, tokenizer (parquet -> bin/idx), blend builder
tools/                   prefetch_hf.sh, netmon.py, summarize_netlogs.py, test-infra.sbatch
```

## Setup (once)

```bash
cp config.sh.example config.sh   # set SBATCH_ACCOUNT, WANDB_API_KEY (+ WANDB_ENTITY)
source config.sh
git submodule update --init --recursive
sbatch env/build_image.sbatch     # ~1.5 h, one node: NeMo 26.08.01 + Alps network stack (31 GB squashfs)
./tools/prefetch_hf.sh            # Qwen3.5 config + tokenizer for offline jobs (login node, MBs)
sbatch tools/test-infra.sbatch    # NCCL over Slingshot/NVLink + NCCL inspector check (2 nodes)
```

## Data: Nemotron 3.5 Lightning open-data blend

`data/lightning35_sources.json` is NVIDIA's `data_blend_raw.json` (Nemotron v0.2.0) verbatim,
plus where each subset lives in the mirror. Every subset was checked file-by-file against
Hugging Face (counts and byte sizes identical; 15.5 TB of parquet), so nothing needs to be
downloaded. Note Nemotron-CC-v2/v2.1 are gated datasets on Hugging Face: accept their license
there before using them.

```bash
source config.sh
sbatch data/tokenize.sbatch                          # 23 array tasks, one node each
TARGET_TOKENS=50e9 sbatch data/tokenize.sbatch       # more tokens per source (resumes)
python3.11 data/build_blend.py --train-tokens 500e9  # -> $DATA_ROOT/blends/lightning35_qwen35.json
```

- One `.bin/.idx` per parquet file (`$DATA_ROOT/qwen35/<source>/`), int32, each document
  `text + <|endoftext|>` (no BOS), as the Lightning recipe (`add_bos=false, add_eos=true`).
- Files are taken in a seeded random order per source until `TARGET_TOKENS` (default 25B per
  source); rerunning continues, receipts make it restartable. Measured: **45M tokens/s per
  node** (one 0.74B-token parquet file in 16 s), so the default (~0.5T tokens, ~2 TB) is about
  3 node-hours. `MAX_ROWS` smoke runs write to `$DATA_ROOT/qwen35-smoke` and never mark real
  files as done.
- Weights: like the Lightning recipe, every source has weight 1.0, i.e. a **uniform mixture
  over the 23 sources** regardless of size, so small sources are repeated: from the parquet
  metadata the mirror holds roughly 10T tokens, but e.g. Scientific-Coding has only ~1.2B
  tokens (~18 epochs in a 500B-token run at 1/23 of the samples), Wiki-Rewrite ~6B (~3.5),
  InfiniByte-Reasoning ~16B; `build_blend.py` prints the epochs per source for your budget. `--weights natural` mixes by size instead; `--source-weights file.json` sets
  per-source weights (e.g. a quality-heavier phase 2: the Lightning recipe upweights
  High-Quality, High-Quality-Synthetic and Specialized-v1 there; NVIDIA does not publish the
  numbers). Validation/test are carved out of every shard (`split 9990,8,2`).

## Training

```bash
./launch.sh smoke                    # 2 nodes, debug partition: whole pipeline in ~7 min (verified)
DRYRUN=1 ./launch.sh train 16 1T     # resolve + validate the full config (1-node job)
./launch.sh throughput 8             # full model, 30 steps, mock data, network logs
./launch.sh train 16 1T              # real data, 1T tokens, batch ramp, checkpoints, auto-resume/resubmit
```

`launch.sh` validates cheap things before anything is queued (divisibility, patches apply to
the pinned commits, image/data/tokenizer present, generated script parses) and
`train/pretrain.py` rejects config overrides that do not exist (Bridge itself would only warn).
Knobs are environment variables, see the header of `launch.sh`; any config field can be set
with `EXTRA="key=value ..."`. A broken node can be avoided with `SBATCH_EXCLUDE=nid00XXXX`.

### Short experiments: optimizers, inter-datacenter split, per-job patches

```bash
OPTIMIZER=muon MAX_STEPS=500 WALLTIME=04:00:00 MAX_RESUBMITS=0 ./launch.sh train 8 1T
OPTIMIZER=muon-zero1 GLOBAL_DP=2 MAX_STEPS=500 WALLTIME=04:00:00 MAX_RESUBMITS=0 ./launch.sh train 8 1T
```

- `MODEL=35b-a3b|0.8b|2b|4b|9b`: Qwen3.5-35B-A3B (default, MoE) or a Qwen3.5 dense model. The dense
  sizes use NVIDIA's GB200 Qwen3.5-9B text recipe built from each size's HF config
  (`Qwen/Qwen3.5-<size>-Base`; same tokenizer, so the tokenized data is shared), default to pure data
  parallelism (EP 1, PP 1) and work with every `OPTIMIZER` and `GLOBAL_DP`. Layouts that fit and are
  fastest: `analysis/2026-10-29-dense-parallelism/REPORT.md` (0.8B/2B: MBS 2; 4B: MBS 1; 9B: TP 2 MBS 2, or
  PP 2 for Dion). Each model logs to its own W&B project `qwen35-<model>-cscs`.
- `MAX_STEPS=N` stops at step N but keeps the batch and LR schedules of the full token budget, so
  the short run is the start of the long one (run name `-max<N>`, own checkpoints).
- `OPTIMIZER=adam` (recipe, ZeRO-1) | `muon` (Megatron-Core's layer-wise Muon, Kimi-K2 settings,
  no ZeRO) | `muon-zero1` (Muon with DeepSeek-V4-style shard-aligned ZeRO-1, optional patch set
  `muon-zero1`).
- `GLOBAL_DP=N` splits the nodes into N "datacenters" that each hold a full model replica
  (`ddp.num_distributed_optimizer_instances`): pipeline, expert all-to-all and the gradient
  reduce-scatter / parameter all-gather stay inside a datacenter, and the gradient shards are
  all-reduced between datacenters once per step. Works with `adam`, `muon-zero1` and `dion-zero1`.
- `OPTIMIZER=dion-zero1` (optional patch sets `muon-zero1` + `dion`; own implementation of Dion,
  Ahn et al. 2025, in `megatron/core/optimizer/dion.py`): Dion for the weight matrices on the same
  ZeRO-1 layout, Adam for the rest. With `GLOBAL_DP>1` it replaces the cross-datacenter gradient
  all-reduce of the matrices: each datacenter keeps its own Dion momentum and only the rank-r
  factors cross ((m+n)·r instead of m·n values per matrix; exactly equivalent to Dion on the global
  gradient). `DION_RANK_FRACTION` [0.25], `DION_COMM_DTYPE` [bf16], `DION_SYNC_CHECK=N` (check every
  N steps that the datacenters hold identical weights). Measured at 8 nodes: 3.0×/4.8× fewer
  cross-datacenter bytes per GPU (stage 0/1) at r=1/4 and faster steps than Muon (WORKLOG.md);
  unit tests: `SNAP=<snapshot> sbatch tests/run_dion_tests.sbatch`.
- `MUON_SPECTRAL=polar|power:<p>|top:<f>` (with `OPTIMIZER=muon|muon-zero1`, patch set `muon-spectral`): Muon's update
  from an exact SVD with reweighted singular values, rescaled to Muon's update RMS (all equal, s**p, or only the top
  fraction), to test which singular directions matter (WORKLOG §8). `DION_SCALE` multiplies Dion's update.
- `PATCHES=set1,set2` applies optional patch sets `patches/optional/<set>/` to this job only (its own
  code snapshot, so jobs with different sets can run at the same time); see
  `patches/optional/README.md`.

Results of the 8-node, 500-step comparison (Adam vs Muon vs Muon ZeRO-1 + GLOBAL_DP=2: loss, step
time, network per step, who talks to whom on the dragonfly): `analysis/2026-10-21-adam-muon-gdp2/REPORT.md`.

### Global batch size ramp (DeepSeek-style)

In train mode the budget is given in tokens and the global batch grows like DeepSeek-V2/V3
(V3: 3072 → 15360 sequences over its first 469B of 14.8T tokens): by default from 512 to 2560
sequences (**2.1M → 10.5M tokens per batch**) in 4 equal steps over the first 3% of the run.
`launch.sh` prints the schedule it submits, e.g. for `./launch.sh train 8 1T`:

```
  from iteration        1  (        0 tokens):   512 sequences =  2.10M tokens/batch
  from iteration     3578  (   7.502B tokens):  1024 sequences =  4.19M tokens/batch
  from iteration     5366  (      15B tokens):  1536 sequences =  6.29M tokens/batch
  from iteration     6558  (    22.5B tokens):  2048 sequences =  8.39M tokens/batch
  from iteration     7453  (   30.01B tokens):  2560 sequences = 10.49M tokens/batch
  total: 99958 iterations, 1T tokens; LR warmup 1.049B tokens, decay over all of them
```

- `GBS_RAMP=START:FINAL:STEPS:LENGTH` (LENGTH in tokens or `%` of the run), `GBS_SCHEDULE="0:512
  20B:1024 ..."` for an explicit schedule, `GBS_RAMP=none GBS=1024` for a constant batch. Every
  batch size must be a multiple of MBS × data-parallel size (checked before submitting).
- The LR schedule is counted in samples, so it is unchanged by the ramp: warmup `WARMUP_TOKENS`
  (default 1.05B = the recipe's 500 steps × 512 × 4096), cosine to `MIN_LR` over the whole budget.
- Megatron-Bridge v0.6.2 cannot do this by itself: Megatron-Core ignores the deprecated
  `rampup_batch_size` and Bridge does not pass `step_batch_size_schedule` —
  `patches/megatron-bridge/0001-step-batch-size-schedule.patch` fixes that.
- With PP the CUDA graphs are captured per microbatch count, so the batch does not grow inside a
  running process: at the last iteration of each batch size the trainer saves a checkpoint and
  stops, and the job restarts training in the same allocation (a few minutes per step, 4 times per
  run); the new process resumes with the larger batch.

### Job lifecycle in train mode (verified at 8 nodes on the real blend)

- Jobs of `WALLTIME` (12 h); training stops 15 min before the limit with a checkpoint and the job
  resubmits itself right away (Slurm's own copy of the running script, so later `launch.sh` calls
  cannot change a running experiment; the network logs are compressed only afterwards) until the
  token budget is reached. `MAX_RESUBMITS` [200] caps the chain. Every job runs `train/`, `tools/`,
  `conf/` and `patches/` from a read-only snapshot taken at submission.
- **Node failures**: train jobs are `--requeue`; Slurm requeues a job that loses a node and it
  resumes from the last checkpoint (happened twice in tests). A requeued job that had already
  submitted its successor exits instead of training twice. Both failures were Clariden-side: a set
  of ~70–100 nodes keeps dropping out of Slurm for an instant (`DOWN* Not responding`, hundreds of
  events a day), which kills every job on them. launch.sh and each resubmission therefore pass
  `--exclude=` the nodes with recent events (`tools/flaky_nodes.sh`: ≥ 1 in 12 h or ≥ 3 in 48 h;
  would have avoided 91% of the events of 18–20 Oct; `EXCLUDE_FLAKY=0` disables it), and train mode
  saves every 100 steps (`SAVE_INTERVAL`; ~77 min at GBS 2560 on 8 nodes) so a failure loses at most
  that.
- **Crashes** (NCCL timeout, GPU error, ...): the job resubmits from the last checkpoint too, but at
  most `CRASH_RETRIES` [2] times without a new checkpoint in between; then the chain stops until you
  relaunch (tested by killing a training step: job 3704910).
- **Batch-size steps**: the trainer saves a checkpoint and stops; the job restarts training in
  the same allocation if ≥ 25 min are left, otherwise the next job continues (both seen in tests).
- **Checkpoints** (torch_dist, 469 GB each with the fp32 optimizer state, last 3 kept): saving
  takes 23–106 s at 8 nodes (median ~50 s); loading took 6–17 min with the default per-replica reads,
  3–6 min split across data-parallel ranks (`checkpoint.fully_parallel_load: true`). The optimizer state is
  DP-reshardable only: the node count may change between jobs, TP/PP/CP/EP/ETP/VPP may not
  (launch.sh checks `$RUN_DIR/layout`). On a new node count launch.sh also sets
  `checkpoint.load_rng=false`: with EP > 1 the RNG state is saved per data-parallel rank and cannot
  be resharded (nothing in this model draws from it: no dropout, no router jitter).
- **Host memory**: the node's anonymous memory grew ~5 GB/min until the kernel OOM-killed a rank,
  20–40 min into a job. Cause: Megatron's `GPTDataset` builds a 16.7 MB `[1, 4096, 4096]` attention
  mask and caches it per dataset object, and the blend has one per prefix (964) — in every dataloader
  worker, for the train, validation and test loaders: up to 16 GB per worker. Bridge never uses that
  mask (attention applies its causal mask itself; the batch gets `attention_mask=None`), so the
  profile sets `dataset.create_attention_mask: false`. Also 2 workers per rank instead of 8 (plenty
  for ≤ 3.5 samples/s per rank), `gc.freeze()` before they fork, `malloc_trim` after the checkpoint
  load and every save. Host and GPU memory of every node are in the step logs and W&B (`node/*`).

### Parallel layout on GH200

**How the parallelism works.** Megatron's "MoE parallel folding" gives the two halves of the
model separate grids on the same GPUs:

```
dense part  (gated attention, Gated DeltaNet, norms, router, shared expert, embeddings: 2.5B, 7%)
            -> TP x CP x DP x PP
routed experts (256 x top-8:  33.0B, 93%)
            -> ETP x EP x expert-DP x PP
```

So "EP for the feed-forward experts, TP/SP for attention" is directly supported (e.g. `TP=2 EP=4`:
attention and GDN tensor-parallel with sequence parallelism, experts expert-parallel). We measured it
and do not use it here: TP shards only the 7% dense part (the experts stay whole per EP rank,
ETP=1), so TP=2 saved just ~3 GB of static memory per GPU, and it was 25–40% slower per GPU (hidden
size 2048 is thin: TP halves already-small GEMMs and adds 4 collectives per layer). What has to be
sharded is the experts: by EP (EP > 4 leaves the node: all-to-all 40 vs 334 GB/s), by more data
parallelism (optimizer shards) or by **pipeline parallelism**, which halves everything per GPU.

**Memory** (measured, full model on 8 nodes, details in `WORKLOG.md`): a stage of 20 layers holds
~39 GB of weights, gradients and optimizer shards (the last stage ~44 GB: output layer, MTP layer);
activations are ~0.57 GB per layer per 4096-token microbatch plus ~12 GB for the two 248k-vocab
logit tensors (main and MTP head). PP=1 does not fit (~85 GB peak plus CUDA/NCCL, the 8-node run
died with `Triton Error [CUDA]: out of memory`); keep the peak below ~80 GB on a 95.6 GB GH200.

**Micro-batch size.** The global batch is fixed, so MBS does not change the training math, only the
efficiency (bigger GEMMs, fewer launches per token) against activation memory. With PP=2, MBS 2
needs ~86 / ~92 GB and fails (measured); MBS 2 requires PP=4. (The GB200 recipe notes that even
186 GB GPUs OOM at MBS 4 with learned routing.)

Measured on 8 nodes (32 GH200), full model, mock data, GBS 512 × 4096, iterations 11–30:

| Layout | Peak GB (first / last stage) | s / iteration | tokens/s/GPU | TFLOP/s/GPU |
|---|---|---|---|---|
| EP4 PP1 MBS1 | OOM | – | – | – |
| EP4 PP2 MBS1, alltoall | 62.5 / 68.1 | 12.86 | 5,100 | 115 |
| EP4 PP2 MBS1, **HybridEP** | 62.1 / 68.2 | **11.68** | **5,610** | **127** |
| EP4 PP2 MBS2 | OOM | – | – | – |
| EP4 PP4 MBS2, alltoall | 82 (stage 0) | OOM at CUDA-graph capture (4 microbatches in flight) | – | – |
| EP4 PP2 MBS1, HybridEP, CUDA graphs off | 62.1 / 68.2 | 15.53 | 4,220 | 96 |
| EP4 PP2 MBS1, HybridEP, NCCL inspector on (NETLOG=2) | 62.1 / 68.2 | 11.78 | 5,480 | 125 |
| EP4 PP2 MBS1, HybridEP, **GBS 2560** (final ramp size, 160 microbatches/rank) | 68.2 alloc / 73.1 reserved (last stage, after graph capture) | 50.2 | 6,530 | 147 |
| EP4 PP2 layout 22/18, HybridEP | 68.2 / 63.5 | 10.96 | 5,980 | 131 |
| EP4 PP2 layout **23/17**, HybridEP | 71.2 / 61.2 | **10.44** | **6,280** | 138 |
| EP4 PP2 layout 24/16, HybridEP | 73.9 / 59.1 | 10.59 | 6,190 | 139 |
| EP4 PP2 layout 25/15, HybridEP | 76.9 / 56.7 | 11.01 | 5,950 | 134 |
| EP4 PP2 VPP2 (± EP overlap), HybridEP | 69–71 / 75 | 11.86 with EP overlap | – | failed: kernel fault (VPP2), hang (VPP2 + EP overlap) |
| EP4 PP2 layout 23/17, HybridEP, GBS 2560 | 71.3 alloc / 76.3 reserved (stage 0, after graph capture) / 61.2 / 66 | 46.1 | 7,110 | 160 |

TE-scoped CUDA graphs are worth +33% on GH200 (the Grace CPU cannot launch the many small MoE /
GDN kernels fast enough: ~230k kernels per step, 11–16 µs on average), HybridEP +10%, the final
batch size +16% (pipeline bubble 1/160 instead of 1/32), and an **uneven pipeline split +12%**: an
nsys profile showed the first stage waiting ~22% of the time for the last one, which also computes
both 248k-vocab output heads (main and MTP) and the MTP layer; `Et*23|t*17mL` (23 layers + embedding
| 17 layers + MTP + loss) is the optimum on mock data. VPP (interleaving, required for MoE
EP-communication overlap with PP) crashed or hung on this stack and was not faster. The NCCL
inspector costs 2.4%.

**Real data changes the picture** (8 nodes, GBS 1024, 45 iterations from scratch, same seed and data
for every split, so the same routing at each step): once the router learns, the experts' load becomes
uneven (`global_load_balancing_loss` ~3 instead of ~1 on mock data), every MoE layer waits for its
busiest expert-parallel rank, and steps get slower as training proceeds (16.8 → 21.6 s for 23/17
between iterations 20 and 45). The per-layer cost then dominates the last stage's extra work, and
the optimum moves toward an even split:

| Split | s/it, its 16–25 / 26–35 / 36–45 | Lowest free HBM on any GPU |
|---|---|---|
| 20/20 | 18.99 / 18.08 / 21.27 | 8.2 GB (last stage) |
| **21/19** | 18.01 / 17.50 / **20.51** | **11.0 GB** |
| 22/18 | 17.99 / 18.04 / 21.88 | 8.2 GB (first stage) |
| 23/17 | **16.99** / 17.62 / 21.24 | 6.6 GB (first stage) |

21/19 is as fast as 23/17 over the run (18.67 vs 18.62 s) and faster in the latest window, and it
leaves 11 GB on the fullest GPU instead of 6.6 GB: the busiest expert ranks need the most activation
memory, so it is the default for `PP=2` (`Et*21|t*19mL`). Expect real steps to be slower than the
mock-data numbers above while routing is unbalanced.

Default: **EP=4, PP=2 with the 21/19 layout**, TP=CP=1, MBS=1, HybridEP, TE-scoped CUDA graphs,
seq 4096, bf16 gradient reduction (`GRAD_FP32=0`, what NVIDIA's own Hopper MoE recipe for
Qwen3-30B-A3B uses), ≥ 8 nodes; global batch 512 → 2560 (ramp).

### Deviations from the GB200 recipe (`conf/qwen35_35b_a3b_gh200.yaml` + launch.sh)

| Setting | GB200 recipe | Here | Why |
|---|---|---|---|
| MoE dispatcher | flex / HybridEP | HybridEP while the expert group is inside a node (`DISPATCHER=auto`), else `alltoall` | HybridEP/DeepEP cross-node paths need InfiniBand GPU-initiated RDMA, which Slingshot lacks; HybridEP's NVLink path works and is 10% faster at EP=4 |
| TP comm overlap | on | off | TP=1 (Bridge would disable it anyway) |
| DP overlap (grad RS / param AG) | off | on | DP traffic crosses Slingshot here |
| EP / PP | EP 8, PP 1 | EP 4, PP 2 (`EP`, `PP`) | memory (95.6 vs 186 GB) + keep all-to-all on NVLink |
| grad reduce | fp32 | bf16 (`GRAD_FP32=0`) | memory; NVIDIA Hopper MoE practice |
| logging | every 10 steps | every step, throughput + memory + timers | |
| NCCL timeout | 10 min | 30 min | first-step JIT / CUDA-graph capture |

Everything else (architecture, init, optimizer, LR schedule 3e-4→3e-5 cosine, warmup 500,
GBS 512, seq 4096, TE-scoped CUDA graphs, fusions) is the official recipe. The Lightning
recipe's own hyper-parameters differ (seq 8192, LR 1.6e-3→1.6e-5, warmup 333, init std
0.0173, aux-loss-free sigmoid router); set them with `SEQ_LEN`, `LR`, `MIN_LR`, `WARMUP`,
`EXTRA` if you want to follow Lightning more closely.

## Logging

- **W&B** (rank N-1): loss, LR, grad norm, `throughput/device/tokens_per_sec`, TFLOP/s/GPU,
  iteration time, memory, and `net/*` (Slingshot bytes and GB/s of that rank's node per step).
  One W&B run per experiment, continued across resubmitted jobs (`WANDB_RUN_ID` is stored in
  the run directory). Disabled when `WANDB_API_KEY` is empty.
- **TensorBoard**: `$RUN_DIR/tensorboard`.
- **Training time**: `$RUN_DIR/logs/train_time.jsonl` (per step: end time, cumulative time in the
  training loop over all jobs of the run, LM loss) and W&B `time/train_hours` with `vs_time/<loss>`
  charts (loss against training time).
- **Perplexity**: training PPL = exp(loss of the step) as W&B `ppl/lm loss` (and `vs_time/ppl lm loss`,
  `lm_ppl` in `train_time.jsonl`); validation PPL as `lm loss validation ppl` (Bridge, enabled in the
  profile). `FINAL_EVAL=0` (default with `MAX_STEPS`) skips Bridge's validation + test passes after the
  last step, which repeat the last step's evaluation.
- **Per-phase timers** (`logger.timing_log_level: 1`): W&B / TensorBoard `forward-backward-time`,
  `all-grads-sync-time` (gradient reduction left after the backward, i.e. exposed DP
  communication), `optimizer-time` and its parts. Console printing is suppressed (the W&B write
  resets them). Cost in the smoke run: +2.7% on 0.9-s steps (fixed per step, ~0.2% at 13 s).
- **Data wait** (`train/pretrain.py` `DataWaitTimer`): host time each rank spends in
  `next(data_iterator)` per step, no GPU sync; W&B `time/data_wait_{max,mean}_s` and
  `time/data_wait_max_rank`, the same in `train_time.jsonl`, and a warning with the host when a rank
  waits more than 30 s (a rank stuck on data stalls every rank at the next collective, job 3708048).
- **Network traffic** (`NETLOG`, default 2), in `$RUN_DIR/netlogs/job_<id>/`:

  | Level | What | Where |
  |---|---|---|
  | 1 | per step and node: wall-clock window + TX/RX bytes of each Slingshot NIC (hardware counters) | `steps/<host>.jsonl` |
  | 1 | per node, every second: NIC counters (HPC vs IP traffic, packets per traffic class), every 5 s NVLink TX/RX per link | `netmon/<host>.jsonl` |
  | 2 | **every NCCL collective and P2P op on every rank** (MoE all-to-all, DP reduce-scatter / all-gather, ...): op, bytes, GPU-timed duration, bus bandwidth, peer, communicator name (NCCL inspector) | `nccl-inspector/*.log.gz` |
  | 3 | + exact per-op and per-channel timestamps (verbose inspector, much larger) | same |

  `tools/summarize_netlogs.py <dir>` (run automatically after smoke/throughput jobs) writes
  per-step and per-communicator CSVs and a `summary.json`.

  PyTorch does not name its NCCL communicators (`comm_name: unknown`), but they are easy to
  tell apart by size and span (`n_ranks`, `nnodes`, plus the communicator `id`): with EP=4,
  P2P Send/Recv on a 4-rank/1-node communicator is the MoE all-to-all (NVLink), AllGather +
  ReduceScatter on the world-sized one is the dense data-parallel traffic, the
  (world/4)-rank one spanning all nodes is the expert data-parallel traffic, and the small
  AllGather/AllReduce ops are token counts, loss and grad norm. A few P2P records carry
  negative sizes wrapped to uint64 (inspector quirk); the summarizer counts them as
  `invalid_size_records`. Volume (smoke run, 8 ranks, 20 steps of a 4-layer model):
  149k records, 70 MB raw, 4.6 MB gzipped — it grows with layers × microbatches × ranks, so
  for long production runs check the size of the first job's `nccl-inspector/` (or use
  `NETLOG=1`). At full scale (8 nodes, HybridEP): 1.17M records (546 MB raw, 23 MB gzipped) per
  25 steps at GBS 512; projected ~10 TB raw / ~430 GB gzipped for a 1T-token run (the record count
  follows the number of microbatches, not the node count); inspector overhead 2.4%. With HybridEP
  the MoE dispatch/combine bypasses NCCL, so it appears as NVLink byte counters (netmon), not as
  per-op records; `DISPATCHER=alltoall` brings per-op MoE records back at ~10% speed.
- **Host memory** (since the lifecycle test ran a node out of host memory): netmon samples
  `/proc/meminfo` every second, every step logs the node's MemAvailable/Shmem and the rank's
  VmRSS/VmHWM (`steps/<host>.jsonl`, W&B `node/*_GB`), and the summary reports the minimum
  MemAvailable per node.
- **nsys** (`NSYS=1`, optional): raw timeline of steps `NSYS_STEPS` (default 10:12) on ranks
  `NSYS_RANKS` (default 0) with CUDA + NVTX (NCCL's own NVTX ranges included), CUDA graphs
  traced per node. Collection only runs inside the window (cudaProfilerApi capture range);
  in the smoke run the profiled step was ~4% slower and opening/closing the window cost
  ~1.4 s each; at full scale (8 nodes, ranks 0 and 16) the captured steps were 60–75% slower,
  closing the window cost ~28 s, and each report is ~150 MB for 3 steps — fine for an occasional
  window, but the timings inside it are not representative.

## Storage (CSCS cleanup: 14 days on iopsstor, 30 days on capstor scratch)

| What | Where (config.sh) |
|---|---|
| tokenized data | `$DATA_ROOT` on iopsstor (random reads); regenerable from the mirror |
| runs: checkpoints, logs, network logs, nsys | `$RUN_ROOT` on capstor scratch |
| container image | `$IMAGE_DIR` on capstor scratch (striped) |
| code snapshots, dataset index cache, HF files | iopsstor |

The project `prep23` has no Store / datacache allocation: copy anything worth keeping
(final checkpoints) elsewhere before the cleanup policy deletes it. A running training reads the
tokenized data, snapshots and caches at every job start, which keeps them; a pause of more than 14
days loses them (re-tokenize: `sbatch data/tokenize.sbatch`). Per-job network logs, nsys reports and
old W&B/TensorBoard files are never read again and disappear 30 days after they were written.

## Patches

See [How_To_Edit.md](How_To_Edit.md). None of the GH200 patches of the earlier Megatron-LM
project are needed with these versions (all four were fixed upstream or do not apply).
