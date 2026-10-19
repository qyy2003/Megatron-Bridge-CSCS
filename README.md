# Megatron-Bridge-CSCS

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
DRYRUN=1 ./launch.sh train 16 100000 # resolve + validate the full config (1-node job)
./launch.sh throughput 8             # full model, 30 steps, mock data, network logs
./launch.sh train 16 300000          # real data, checkpoints, auto-resume/resubmit
```

`launch.sh` validates cheap things before anything is queued (divisibility, patches apply to
the pinned commits, image/data/tokenizer present, generated script parses) and
`train/pretrain.py` rejects config overrides that do not exist (Bridge itself would only warn).
Knobs are environment variables, see the header of `launch.sh`; any config field can be set
with `EXTRA="key=value ..."`.

### Parallel layout on GH200

GB200 has 186 GB per GPU and an NVL72 domain; the recipe puts EP=8 on 8 GPUs. On GH200
(95.6 GiB, 4 GPUs per NVLink domain, Slingshot between nodes) the static state alone of that
layout is ~93 GB/GPU. Per-GPU static memory (bf16 params, fp32 Adam via the distributed
optimizer, TP=PP=1; the model has 35.5B parameters incl. the MTP layer: 33.0B routed-expert
+ 2.5B other):

| Layout | params + grads | optimizer | static (+ activations, ~30 GB at MBS 1 × 4096) |
|---|---|---|---|
| 8 GPUs, EP=8, fp32 grads (GB200 recipe) | 40 GB | 53 GB | 93 GB |
| 32 GPUs, EP=4, bf16 grads | 43 GB | 13 GB | 56 GB |
| 64 GPUs, EP=4, bf16 grads | 43 GB | 7 GB | 50 GB |
| 32 GPUs, EP=8, fp32 grads | 40 GB | 13 GB | 53 GB |

How the parallelism works (Megatron "MoE parallel folding"): EP shards only the routed
experts. The dense part — gated attention, Gated DeltaNet, norms, router, shared expert,
embeddings/output — has its own TP×CP×DP grid on the same GPUs. With TP=1 every GPU holds the
dense part (~7% of the parameters) and runs attention/GDN for its own sequences; the MoE
all-to-all then moves tokens between the 4 GPUs of a node that own the experts. TP=1 is also
what NVIDIA uses for this model class (thin model, hidden 2048: TP adds 4 collectives per
layer and narrows the GEMMs). TP=2 (or 4 with KV-head replication; Qwen3.5 has 2 KV heads)
with sequence parallel is supported (`TP=2`) and halves the dense activations, e.g. the
248k-vocab logits, which is the main way to afford MBS 2. MBS does not change training (GBS
is fixed); it trades gradient-accumulation steps for per-microbatch efficiency.

Default: **EP=4, PP=2** (the expert all-to-all stays on NVLink inside a node; only
data-/pipeline-parallel traffic crosses Slingshot), TP=CP=1, MBS=1, GBS=512, seq 4096, bf16
gradient reduction (`GRAD_FP32=0`, what NVIDIA's own Hopper MoE recipe for Qwen3-30B-A3B uses),
at ≥ 8 nodes. The PP=1 rows above do not fit in practice: measured activations add ~24 GB at
MBS 1 (mostly the two 248k-vocab logit tensors of the main and MTP heads), and the 8-node PP=1
run failed with `Triton Error [CUDA]: out of memory`; PP=2 halves the static memory (memory
model and probes in `WORKLOG.md`). MBS and dispatcher defaults are being benchmarked at 8 nodes.

### Deviations from the GB200 recipe (`conf/qwen35_35b_a3b_gh200.yaml` + launch.sh)

| Setting | GB200 recipe | Here | Why |
|---|---|---|---|
| MoE dispatcher | flex / HybridEP | `alltoall` | HybridEP/DeepEP cross-node paths need InfiniBand GPU-initiated RDMA; Slingshot has none |
| TP comm overlap | on | off | TP=1 (Bridge would disable it anyway) |
| DP overlap (grad RS / param AG) | off | on | DP traffic crosses Slingshot here |
| EP | 8 | 4 (`EP`) | memory + keep all-to-all on NVLink |
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
  `NETLOG=1`).
- **nsys** (`NSYS=1`, optional): raw timeline of steps `NSYS_STEPS` (default 10:12) on ranks
  `NSYS_RANKS` (default 0) with CUDA + NVTX (NCCL's own NVTX ranges included), CUDA graphs
  traced per node. Collection only runs inside the window (cudaProfilerApi capture range);
  in the smoke run the profiled step was ~4% slower and opening/closing the window cost
  ~1.4 s each.

## Storage (CSCS cleanup: 14 days on iopsstor, 30 days on capstor scratch)

| What | Where (config.sh) |
|---|---|
| tokenized data | `$DATA_ROOT` on iopsstor (random reads); regenerable from the mirror |
| runs: checkpoints, logs, network logs, nsys | `$RUN_ROOT` on capstor scratch |
| container image | `$IMAGE_DIR` on capstor scratch (striped) |
| code snapshots, dataset index cache, HF files | iopsstor |

The project `prep23` has no Store / datacache allocation: copy anything worth keeping
(final checkpoints) elsewhere before the cleanup policy deletes it.

## Patches

See [How_To_Edit.md](How_To_Edit.md). None of the GH200 patches of the earlier Megatron-LM
project are needed with these versions (all four were fixed upstream or do not apply).
