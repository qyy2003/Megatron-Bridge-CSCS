# Handover: state on 2026-10-30 and how to restart

This file gives a person or a coding agent everything needed to continue the project, including
after the CSCS account `yqin` / project `prep23` is closed. Read it first. Then use the other
documents for detail:

| Document | What it holds |
|---|---|
| `README.md` | How to use everything: setup, data, every `launch.sh` mode and knob, logging, storage |
| `WORKLOG.md` | Every change, test and result, with Slurm job IDs, in order |
| `analysis/*/REPORT.md` | Write-ups of the experiment campaigns, with figures, data and scripts |
| `PRODUCTION_RUN.md` | The 35B-A3B production run (stopped; kept as the Adam baseline) |
| `CLAUDE.md` | Working rules for a coding agent (the user's conventions) |

## 1. What this repository is

The repository trains **Qwen3.5 from scratch** with **Megatron-Bridge v0.6.2** on CSCS Alps
(Clariden, GH200, Slingshot) on NVIDIA's **Nemotron 3.5 Lightning open-data blend**.

**Models**
- Qwen3.5-35B-A3B (MoE; the official `qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config` recipe).
- Qwen3.5 dense 0.8B / 2B / 4B / 9B (NVIDIA's 9B text recipe built from each size's config).

**Code changes** live as patches over the pinned submodules (`patches/`), with optional per-job
patch sets (`patches/optional/<set>`).

**What was built on top of the recipes**
- Batch-size ramp and job chaining with automatic resubmission.
- Network logging (NIC counters, NCCL inspector) and W&B logging: loss, perplexity, training time,
  data-wait, per-phase timers.
- Optimizers: Muon, ZeRO-1 Muon, and Dion (own implementation, Ahn et al. 2025).
- Simulated inter-datacenter training (`GLOBAL_DP`).
- Experimental Muon spectral modes (`MUON_SPECTRAL`).

## 2. Where things are (and what disappears with the account)

| What | Where | Survives account closure? | How to recreate |
|---|---|---|---|
| Code, docs, worklog, analysis data and figures | GitHub `qyy2003/Megatron-Bridge-CSCS`, branch `main` | yes | clone with `--recursive` |
| W&B runs (all losses, perplexity, timers, network metrics) | wandb.ai entity `yuyqin-eth-zurich`, projects `qwen35-35b-a3b-cscs`, `qwen35-9b-cscs`, `qwen35-0.8b-cscs` (one project per model) | yes (external) | n/a |
| Container image (31 GB squashfs) | `/capstor/scratch/cscs/yqin/mbcscs/images/nemo-26.08.01-alps-mbcscs.sqsh` | **no** | `sbatch env/build_image.sbatch` (~1.5 h, 1 node) |
| Tokenized data (846B tokens, int32 ≈ 3.4 TB) + blends | `/iopsstor/scratch/cscs/yqin/mbcscs/data/{qwen35,blends}` | **no** | `data/tokenize.sbatch` + `data/build_blend.py` (README "Data"; ~3 node-hours per 0.5T tokens) |
| HF configs + tokenizers (no weights) | `/iopsstor/scratch/cscs/yqin/mbcscs/hf_home` | **no** | `./tools/prefetch_hf.sh` (login node; pins in `env/versions.env`) |
| Dataset index cache, code snapshots | `/iopsstor/scratch/cscs/yqin/mbcscs/{cache,code}` | **no** | rebuilt automatically by the first job |
| Run directories: checkpoints, `logs/train_time.jsonl`, network logs, TensorBoard | `/capstor/scratch/cscs/yqin/mbcscs/runs/<run name>` | **no** | rerun (commands in WORKLOG / below) |
| Slurm job logs (`logs/*.log`, git-ignored) | repo `logs/` | **no** | n/a. The key numbers are in WORKLOG and in `analysis/*/data` |

**Checkpoints that exist now** (lost with the account; none are irreplaceable):
- 35B-A3B Adam production run, iteration 1000 (`train-q35a3b-gbs512to2560-sl4096`, ~1.4 TB for 3 kept).
- 35B Muon, 500 (`…-muon-max500`).
- 35B Muon ZeRO-1 + GDP2, 461 (`…-muon-zero1-gdp2-max500`).
- 9B Muon ZeRO-1 + GDP2, 500, and 9B Dion r=1/4 + GDP2, 500 (`train-q35-9b-…-max500`).

The source parquet data is read from the Swiss AI mirror
`/capstor/store/cscs/swissai/infra01/datasets/nvidia` (`SOURCE_MIRROR` in `config.sh`). On another
account this path may not be readable. In that case download the Nemotron subsets listed in
`data/lightning35_sources.json` from Hugging Face. Nemotron-CC v2 and v2.1 are gated: accept their
licenses first.

## 3. Restart from zero (new account or cluster)

```bash
git clone --recursive git@github.com:qyy2003/Megatron-Bridge-CSCS.git && cd Megatron-Bridge-CSCS
cp config.sh.example config.sh   # set SBATCH_ACCOUNT, WANDB_API_KEY (+ WANDB_ENTITY), the storage paths
source config.sh
sbatch env/build_image.sbatch     # container (NeMo 26.08.01 + Alps network stack)
./tools/prefetch_hf.sh            # Qwen3.5 35B-A3B + dense configs/tokenizers (login node)
sbatch tools/test-infra.sbatch    # NCCL / inspector check, 2 nodes
sbatch data/tokenize.sbatch       # TARGET_TOKENS=45e9 gave the ~1T-token data used here
python3.11 data/build_blend.py --train-tokens 1e12
./launch.sh smoke                 # 2 nodes, debug partition, ~7 min: the whole pipeline
```

- **Never commit `config.sh`.** It is git-ignored and holds the W&B key, which generated job
  scripts never contain.
- **On another cluster:** account, partitions and the flaky-node list are Clariden-specific (README
  "Job lifecycle", `tools/flaky_nodes.sh`). Adapt `config.sh` and `env/` first.
- **Node count:** each model has an efficient size (§5).

## 4. Results so far (one line each; details in the reports)

**35B-A3B, 8 nodes, first 500 steps** (`analysis/2026-10-21-adam-muon-gdp2/REPORT.md`)
- Validation at step 500: Muon **2.536** vs Adam 2.820. Adam leads only in the first ~60 steps.
- Muon ZeRO-1 + GLOBAL_DP=2 trains like Muon.
- The cross-datacenter all-reduce costs ≤ 3.3% of a step.
- 93 min of stalls in one run came from the shared filesystem; a data-wait timer was added since.

**Dion for the cross-datacenter sync, 35B** (`analysis/2026-10-28-dion-xdc/REPORT.md`)
- Exact: weights identical across datacenters.
- Cross-datacenter bytes 3.0–4.8× lower at r=1/4 and 10–11× lower at r=1/16.
- Steps faster than Muon ZeRO-1.

**Dense models** (`analysis/2026-10-29-dense-parallelism/REPORT.md`)
- Best layouts: 0.8B and 2B at TP1 MBS 2; 4B at TP1 MBS 1; 9B at TP2 MBS 2 (or PP2 for Dion).
- The 248k-vocabulary logits limit the micro-batch size.
- 9B Dion r=1/4 vs Muon (GDP2, 4 nodes, 500 steps):

  | | Muon | Dion |
  |---|---|---|
  | Validation at step 500 | **2.617** | 2.641 |
  | MTP acceptance | 46.7% | 45.9% |
  | Cross-datacenter traffic | 1× | 2.3–2.4× lower |
  | Step time | 1× | 1.4% faster |

**Open question that motivated the current experiment:** is Dion behind because of the low rank,
or because of other differences from Muon? The user's hypothesis is that directions with small
singular values matter little. The candidate explanations are discussed in WORKLOG §8.

## 5. In progress: 0.8B spectral experiment (WORKLOG §8)

These seven jobs were **queued and had not started** on 2026-10-30. They were blocked by the
normal partition's 400-node group limit (`QOSGrpNodeLimit`). If the account closes they are gone;
resubmit with the commands below.

Common setup for all seven: 2 nodes each, about 1.3 h and 2.5 node-hours per run once started.
Runs use the `train` schedule: 500 steps, ramp from 512, validation at steps 250 and 500. W&B
project `qwen35-0.8b-cscs`.

```bash
source config.sh
COMMON="MODEL=0.8b MBS=2 GLOBAL_DP=2 MAX_STEPS=500 EVAL_INTERVAL=250 WALLTIME=03:00:00 MAX_RESUBMITS=0"
env $COMMON OPTIMIZER=muon-zero1 ./launch.sh train 2 1T                                       # A Muon (Newton-Schulz)    job 3797996
env $COMMON OPTIMIZER=dion-zero1 DION_RANK_FRACTION=0.25 DION_SYNC_CHECK=100 ./launch.sh train 2 1T   # G Dion r=1/4 (9B setting) 3797997
env $COMMON OPTIMIZER=dion-zero1 DION_RANK_FRACTION=1.0 DION_SYNC_CHECK=100 ./launch.sh train 2 1T    # B Dion full rank          3797998
env $COMMON OPTIMIZER=dion-zero1 DION_RANK_FRACTION=0.25 DION_SCALE=2 DION_SYNC_CHECK=100 ./launch.sh train 2 1T  # C Dion r=1/4, Muon's RMS 3797999
env $COMMON OPTIMIZER=muon-zero1 MUON_SPECTRAL=top:0.25 ./launch.sh train 2 1T                # D top-1/4 directions only  3798000
env $COMMON OPTIMIZER=muon-zero1 MUON_SPECTRAL=power:0.5 ./launch.sh train 2 1T               # E singular values ** 0.5   3798001
env $COMMON OPTIMIZER=muon-zero1 MUON_SPECTRAL=polar ./launch.sh train 2 1T                   # F exact polar (reference)  3798002
```

What each comparison isolates:

| Comparison | Question it answers |
|---|---|
| A vs G | Does the 9B gap reproduce at 0.8B? |
| B vs A | Dion's momentum, power iteration and fused-QKV handling, without the low rank |
| C vs G | Update size (r=1/4 carries half of Muon's RMS) |
| D vs F | The user's hypothesis: drop the weak directions at equal step size |
| E vs F | Partial whitening |
| F vs A | Exact SVD vs Newton–Schulz (they differ, cosine 0.90–0.97, on gradient-like matrices: unit test) |

Compare the validation loss and perplexity at steps 250 and 500, `mtp_1 loss`,
`mtp_1_acceptance_rate`, and the 10-step-mean curves. Use the scripts in
`analysis/2026-10-29-dense-parallelism/scripts/` as a template. Single seeds: treat differences
below ~0.01 in loss as noise.

## 6. Open issues and known quirks

- **Muon ZeRO-1 + GLOBAL_DP: stage 1 sends ~2× the cross-datacenter bytes of stage 0.** Seen with
  35B and 9B. Pipeline stages after the first get a single gradient bucket; the cause of the 2× is
  not found. Dion does not show it.
- **Dion supports TP=1 only** (whole matrices per GPU). For 9B use PP 2.
- **The MTP loss's perplexity is not logged per step** (it is exp of W&B `mtp_1 loss`). The level-1
  timers reach W&B and TensorBoard only; console printing is suppressed by Bridge.
- **Stalls on the shared filesystem** (iopsstor) can cost tens of minutes.
  `time/data_wait_max_s` shows whether a rank waited for data; no stall has been caught with it yet.
- **9B at TP1 PP1 runs out of memory even on 8 nodes.** With GLOBAL_DP=2 the optimizer state is
  sharded over half as many GPUs: 9B needs ≥ 4 nodes (PP 2).
- **Clariden:** ~80 nodes keep dropping out and are excluded automatically
  (`tools/flaky_nodes.sh`). The normal partition caps all users together at 400 nodes, so queue
  times can be long.
- **Scratch cleanup:** files unused for 14 days on iopsstor and 30 days on capstor are deleted.

## 7. Code map

| File | Role |
|---|---|
| `launch.sh` | Every knob is documented in its header (MODEL, OPTIMIZER, GLOBAL_DP, DION_*, MUON_SPECTRAL, MAX_STEPS, FINAL_EVAL, PATCHES, NETLOG, …). Generates `logs/<job>.sbatch` from `train/job_body.sh`. `SUBMIT=0` writes the script without submitting |
| `train/job_body.sh` | Job lifecycle: code snapshot, container, restarts, resubmission, network monitors |
| `train/pretrain.py` | Recipe + profile + overrides. Callbacks: network/memory telemetry, batch-size steps, training time + perplexity, final-eval switch, data-wait timer. Dense recipe wrapper `qwen35_text_dense:<size>` |
| `conf/qwen35_35b_a3b_gh200.yaml` | GH200 profile (also used for dense models) |
| `patches/{megatron-bridge,megatron-lm}` | Always-on patches |
| `patches/optional/` | Optional sets: `muon-zero1`, `dion`, `muon-spectral`; see `patches/optional/README.md` |
| `tools/` | Snapshot build, flaky-node list, HF prefetch, network summaries, `throughput_table.py` |
| `tests/` | Dion and Muon-spectral unit tests (`tests/run_dion_tests.sbatch`, 1 node) |
| `analysis/` | One folder per campaign: `REPORT.md`, `data/`, `figures/`, `scripts/` |
