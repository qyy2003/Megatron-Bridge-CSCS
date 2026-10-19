# Work log: Qwen3.5/3.6-35B-A3B on Clariden with Megatron-Bridge

Brief record of what was set up, what was checked, and the results (2026-10-19).
Usage docs are in `README.md` / `How_To_Edit.md`; this file is the audit trail.

## 1. Decisions (and why)

| Topic | Choice | Reason |
|---|---|---|
| Framework | Megatron-Bridge **v0.6.2** (latest stable; 26.09 tags are RCs), submodule; nested Megatron-LM `a0f793d` | requested recipe `qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config` exists there |
| Container | `nvcr.io/nvidia/nemo:26.08.01` + CSCS Alps network stack (CSCS recipe, `alps-extended-images@6120245`) | the image's Bridge is r0.6.0@`2a8fcdd` = v0.6.2 minus 4 docs/version commits; CSCS ships Alps NeMo images only up to 26.02 |
| Code layout | as in the earlier Megatron-LM project: pinned submodules + `patches/` + `code_store.sh`/`code_restore.sh` + `launch.sh` generating a self-contained sbatch | user preference |
| Jobs use | content-addressed code snapshot (pinned commits + patches) mounted at `/opt/Megatron-Bridge` | reproducible; working trees never touched by jobs |
| Data | Nemotron 3.5 Lightning open-data blend (Nemotron `v0.2.0`, 23 sources, uniform weights) from the Swiss AI mirror, **Qwen3.5 tokenizer** (EOS only) | recipe vocab = Qwen3.5; mirror is complete |
| Layout on GH200 | EP=4 (intra-node), **PP=2**, TP=CP=1, MBS 1, seq 4096, bf16 grad reduce, ≥ 8 nodes | GB200 layout (EP=8 on 8 GPUs) needs ~93 GB static/GPU; EP=4 keeps all-to-all on NVLink; PP=1 does not fit (job 3701785), PP=2 measured 62/68 GB peak (job 3702069); TP2 measured slower and saves little memory (experts are not TP-sharded) |
| MoE dispatcher | **HybridEP** while the expert group is inside a node (`DISPATCHER=auto`: EP×ETP ≤ 4), else `alltoall` | HybridEP/DeepEP cross-node need InfiniBand GPU-initiated RDMA; inside a node HybridEP works (job 3701972) and is 10% faster at full scale with the same memory (job 3702071 vs 3702069) |
| Global batch size | DeepSeek-style ramp in train mode: 512 → 2560 sequences (2.1M → 10.5M tokens) in 4 steps over the first 3% of the token budget (user request); LR warmup/decay counted in samples | Bridge v0.6.2 cannot ramp (patch 0001); each batch step restarts the training process from a checkpoint inside the job because PP + TE CUDA graphs are captured per microbatch count |
| Launch | srun-native, 1 task/GPU, `numactl --cpunodebind=$LOCALID --membind=0-3` | Bridge derives ranks from Slurm; GPU HBM is also exposed as NUMA nodes 4/12/20/28 |
| Network logging | NIC/NVLink counters (netmon), per-step NIC bytes (callback → W&B), NCCL inspector (every coll/P2P op), optional nsys window | "log every network traffic"; nsys optional |
| Patches from old project | none needed (all 4 fixed upstream or not applicable) | checked in source, see How_To_Edit.md |

## 2. Facts verified on the cluster (Slurm debug jobs 3700028, 3700038)

- Driver 595.91.07 (CUDA 13.2); NeMo 26.08.01 is CUDA 13.3 → relies on minor-version compatibility.
- GH200 95.6 GiB/GPU; 4 GPUs fully connected (NV6); GPU i ↔ Grace NUMA i (cores 72i–72i+71).
- `--ntasks-per-node=4 --cpus-per-task=72` binds each task to its own socket.
- 4× Slingshot-11 200G NICs/node; CXI telemetry in sysfs (`hni_sts_{tx,rx}_ok_octets`, per-TC packets),
  ~1 ms per read, readable inside containers; `nvidia-smi nvlink -gt d` gives per-link counters (0.25 s).
- podman 5.8.6 + enroot + 334 GB tmpfs on compute nodes; debug QOS = 1 running / 2 queued jobs, ≤ 4 nodes.
- Mirror `/capstor/store/cscs/swissai/infra01/datasets/nvidia`: all 23 subsets identical to HF
  (file count + byte size, 15.5 TB); every subset has a `text` column; ~10T tokens (estimate).
- NeMo 26.08.01 image (job 3700381): NCCL 2.30.7 as .debs, torch maps `/usr/lib/aarch64-linux-gnu/libnccl.so.2.30.7`,
  Bridge `2a8fcdd` (0.6.1), transformers 5.17, wandb 0.29, nsys 2026.5.1, numactl, pybind11.
- Qwen3.6-35B-A3B text config = Qwen3.5-35B-A3B-Base text config (architecture identical).

## 3. Local checks (login node, before submitting anything)

| Check | Result |
|---|---|
| `bash -n` on every script and on generated/embedded scripts | OK |
| `train/pretrain.py`: unknown override keys / `None` sub-configs rejected, arg parsing | OK |
| `tools/netmon.py` sampling CXI + NVLink (Python 3.6 and 3.11) | OK |
| `tools/summarize_netlogs.py` on synthetic steps/netmon/inspector logs (plain + gzip) | OK, ops attributed to the right steps |
| `data/build_blend.py` on synthetic receipts | OK, uniform per source, token-proportional per shard, sums to 1 |
| `data/tokenize_parquet.py` core on a synthetic parquet with the real Qwen3.5 tokenizer | OK: order, EOS, empty-doc skip, exact round-trip, index layout |
| patch workflow: store → promote → force-restore (incl. new files) → refuse-if-dirty | OK |
| `launch.sh` patch check against pinned commits (temp index), broken patch rejected | OK |
| code snapshot (git archive + patches) | OK, 1.8 s, 5.6k files, reused when unchanged |
| HF prefetch of Qwen3.5 config/tokenizer (`tools/prefetch_hf.sh`) | OK, revision `0f08130` |
| `tools/batch_schedule.py` (mirrors Megatron-Core's step schedule): 1T + default ramp, explicit schedule + iteration budget, divisibility error (MBS×DP=48), Python 3.6 and 3.11 | OK: 1T → 99,958 iterations, steps at 7.5/15/22.5/30B tokens, LR warmup 100 / decay 95,368 final-size batches |
| `launch.sh` schedule paths (`SUBMIT=0`): train with tokens + ramp, constant GBS + token budget (500B → 119,210 its), iteration budget + `%` ramp (rejected), throughput + schedule (rejected), smoke + `GBS_SCHEDULE` (switches to save+load checkpoints) | OK |
| `time_left_mins` (job restart budget) with a stubbed `squeue`: `11:42:07`, `1-02:03:04`, `37:12`, `UNLIMITED`, empty | OK (702, 1563, 37, 0, 0 min; unparseable → no in-job restart, resubmission continues) |
| Bridge patches 0001 + 0002 applied with `code_restore.sh -f`; launch.sh checks them against the pinned commit | OK |
| W&B key handling: value never appears in generated sbatch files (`grep -cF` of the key → 0), `config.sh` git-ignored | OK |

## 4. Problems found and fixed

| Problem | Fix |
|---|---|
| Recipe keeps `mixed_precision="bf16_mixed"` (string) which overwrites `ddp.grad_reduce_in_fp32` at setup | entry point materializes it to a dataclass before overrides |
| Recipe `CommOverlapConfig(overlap_*=False)` overrides `ddp.overlap_*` | profile sets `comm_overlap.overlap_*` |
| Bridge only warns on unknown override keys | strict key check in `train/pretrain.py` (CLI + profile YAML) |
| `cfg.profiling` is `None` → dict override would be a raw dict | `ProfilingConfig` built in Python (`--nsys-steps`) |
| Bridge compiles dataset helpers on every node into a shared dir (race) | job pre-step compiles once per snapshot |
| Smoke reruns: dist-checkpointing refuses existing `iter_*` dirs | smoke job clears its own checkpoints |
| MAX_ROWS smoke receipts could mark real files as done | smoke output in `qwen35-smoke/`; partial receipts never count for full runs |
| `Path.with_suffix` would mangle dotted parquet stems | explicit path building |
| code_restore.sh: `grep` no-match + `pipefail` aborted restore | `|| true` |
| Build 4: `enroot import` exited non-zero after writing a complete squashfs | validated with `unsquashfs` and kept; build script now judges by the squashfs |
| NeMo 26.08.01 ships `/opt/Megatron-Bridge` with directories of mode 644 (no x): not traversable for non-root users (Container Engine runs as the user) | training mounts the snapshot over it; tokenizer job mounts the submodule tree; future builds add a `chmod -R a+rX` layer |
| A node with a hung Slurm prolog had no usable GPUs; the job died only minutes later inside Megatron | pre-step checks `torch.cuda.device_count()==4` on every node; train mode resubmits with `--exclude` |
| launch.sh patch check broke on relative paths under `git -C` | `realpath` |
| Jobs starting together waited for each other: the pre-step held the shared code-snapshot lock for its whole `srun` (health check + versions), although only the first dataset-helper compile needs it | the lock is taken only while `.helpers_ready` is missing |
| The 8-node PP1 run died with `Triton Error [CUDA]: out of memory` at ~85 GB predicted peak | layout moved to PP2 (see Memory model) |
| launch.sh claimed a train run could continue "on a different node count / EP": Megatron saves the distributed-optimizer state DP-reshardable by default (`dist_ckpt_optim_fully_reshardable=False`), so only the data-parallel size may change | comment fixed; launch.sh records `TP/PP/CP/EP/ETP` in `$RUN_DIR/layout` and refuses a different layout once checkpoints exist (`FORCE_LAYOUT=1` to override); tested with a fake run dir |
| W&B could not be tested without an API key | `WANDB_MODE=offline` runs the full W&B path without a key (upload later with `wandb sync`) |
| **Batch-size ramp-up is silently a no-op in Bridge v0.6.2** (also upstream main 2026-10-19): Megatron-Core `a0f793d` ignores the deprecated `train.rampup_batch_size` (only a warning) and Bridge never passes `train.step_batch_size_schedule` to the microbatch calculator | `patches/megatron-bridge/0001-step-batch-size-schedule.patch`: pass the schedule (thresholds in tokens), reject `rampup_batch_size`, require `global_batch_size` = final batch size |
| Bridge's `throughput/*` W&B metrics use the nominal `train.global_batch_size` → tokens/s would be overstated up to 5× during the ramp | same patch: current batch size |
| With PP > 1 the TE-scoped CUDA graphs are captured once per microbatch slot, in the 1F1B order of the microbatch count at capture, with buffers shared by that order (`_order`, `_reuse_graph_input_output_buffers`); more microbatches later would replay them out of order (`current_microbatch % len(cuda_graphs)`) | no in-process batch growth: `BatchSizeStepCallback` (train/pretrain.py) sets `exit_interval` at the last iteration of each batch size → Bridge saves a checkpoint and stops; `job_body.sh` restarts training in the same allocation (time budget from `squeue %L`), the new process loads the checkpoint (Bridge re-derives the microbatch count from `consumed_train_samples`) and captures graphs for the new count |
| A planned stop (`exit_duration_in_mins` at every 12 h job end, `exit_interval`, signal) is logged by every rank as `Pretrain failed ... SystemExit: 0` with a traceback (exit code 0 and cleanup are correct) | `patches/megatron-bridge/0002-planned-exit-not-failure.patch`: one info line instead |
| nsys report names were per job: a restarted training process could overwrite an earlier capture | report name includes the Slurm step id |
| PP2 + MBS2 at 8 nodes: `Cuda failure 2 'out of memory'` in NCCL (`ncclCuMemAlloc`) during the first iteration (job 3702070) | MBS2 needs PP4 (job 3702285); default stays MBS1 |
| **A running job re-ran its training**: I edited `train/rank_wrapper.sh` in place while job 3702069 executed it; bash reads a script incrementally, resumed at its old byte offset in the changed file (21 bytes earlier, inside `numactl ... "${CMD[@]}"`) and started `python train/pretrain.py` again on all 32 ranks (rank 12: PID 71729, then 81540; `merged profile` logged twice per rank) | `launch.sh` snapshots `train/ tools/ conf/ patches/` at submission into `$MBCSCS_IOPS/code/src-<hash>` (read-only, content-addressed) and jobs run only from there; train-mode resubmissions use Slurm's copy of the running script (`$0` → `$RUN_DIR/logs/sbatch-<job>.sh`), not `logs/<job>.sbatch`, which a later `launch.sh` call may regenerate. Files used by already-queued jobs are replaced atomically (new inode) |
| Every start/restart of training spends ~4 min in its first iteration (restart in the same job: 239 s vs 242 s cold; the Triton compile cache on node-local tmpfs does not help — the GDN kernels' autotuning is redone) | `TRITON_CACHE_AUTOTUNING=1` (Triton 3.7 in the image) keeps autotune results in the same tmpfs cache; effect to be measured on the next restart |
| Throughput runs spent ~6 min (8 nodes) after the last step in Bridge's final validation + test evaluation (32 iterations each at the global batch size; ~30 min at GBS 2560) | throughput mode sets `validation.eval_iters=0` (no eval loaders, no final eval); pending benchmarks resubmitted |
| Image builds 1+2 (jobs 3700126, 3700426) failed: NCCL .deb `unable to install new version of './usr/include/nccl_device': File exists`. Removing the leftover paths (build 2) did not help | root cause found with a podman experiment on the NeMo base (job 3700693): **rootless podman on Alps cannot re-create, in the same build step, a directory deleted from a lower layer** (`mv: cannot overwrite non-directory ... with directory`); delete in one layer + create in the next works. CSCS's Alps step purges NeMo's NCCL .debs and re-installs NCCL in one RUN. Fix: run CSCS's `purge` step as its own layer first (stage 0 in `env/build_image.sbatch`), skip nvshmem4py (pip reinstall risk, unused); build 3 = job 3700780 |

## 5. Slurm tests

| Test | Job | Result |
|---|---|---|
| hardware probes | 3700028, 3700038 | done (section 2) |
| NeMo image inspection | 3700381 | done (section 2) |
| overlay delete/re-create experiments (ubuntu base, NeMo base) | 3700681, 3700693 | same-layer re-create fails on NeMo base, split layers OK |
| image build | 3700126, 3700426, 3700780 → 3700830 | 1+2: NCCL .deb install (section 4); 3: stage-0 script layout (scripts expect CSCS's flat `/opt/alps/` layout); 4: **image complete** (31 GB squashfs; enroot exit-code quirk, section 4), used by all jobs since |
| fabric + NCCL inspector (`tools/test-infra.sbatch`, 2 nodes) | 3701550 | OK. busbw (large msgs): all-reduce 335 GB/s intra / 158 GB/s inter-node; all-to-all 334 GB/s intra (EP=4) / 40 GB/s inter-node. Inspector: 1 file/rank, every all-to-all as P2P Send/Recv records, no measurable overhead |
| tokenization, one full file (CC-v2 High-Quality) | 3701552 | OK: 982,598 docs, 0.736B tokens in 16.2 s = **45.4M tok/s/node** (~4.7 bytes/token) |
| tokenization smoke (all 23 sources, 5000 rows) | 3701551 → 3701581 | 1st: `--verify` could not import megatron.core from the image (see section 4) → fixed, rerun |
| training smoke, 2 nodes (`./launch.sh smoke`) | 3701554, 3701619 → **3701756** | 1: node with hung prolog had no GPUs → per-node health check. 2: smoke override `num_layers=4` left the 40-entry GDN/attention pattern → also set `linear_attention_freq=4`. **3: PASSED** (6.4 min): 20 steps, loss 12.8→0.02 (mock data), MTP + aux loss OK, ~550 ms/step steady (~240 TFLOP/s/GPU toy size), eval at 10/20, checkpoints at 15/20 (no GH200 ckpt crash), nsys window costs ~1.4 s at start/stop only, NCCL inspector 149k records (70 MB → 4.6 MB gz), netmon + per-step NIC logs + summary written |
| tokenization smoke, all 23 sources × 5000 rows | **3701581** | **PASSED**: every shard read back with Megatron's IndexedDataset (int32, EOS 248044, text decodes) → smoke blend `lightning35_qwen35_smoke.json` (24 prefixes, 4.35%/source) |
| dry run of the full 35B-A3B config at 8 nodes (`DRYRUN=1 ./launch.sh throughput 8`) | **3701757** | **PASSED** `world_size=32 data_parallel_size=32` |
| full-model throughput, 8 nodes: A EP4/TP1/MBS1 + nsys | 3701785 | **OOM** in the first backward (fla Gated-DeltaNet backward kernel, during Triton autotune). 10.74B params/GPU (2.49B dense + 33.0B/4 experts) = 43 GB bf16 weights+grads + optimizer shards; activations at MBS 1 ≈ 50 GB (two 248k-vocab logits: main + MTP, 30 GDN layers, 41 MoE layers). B/C/D (same or larger footprint) cancelled. Dry runs of C (MBS2 + recompute) and D (TP2+SP MBS2) had passed config validation |
| memory probes: real width (256 experts, 248k vocab, MTP), 8 and 20 layers, 2 nodes (EP4, expert-DP 2), MBS1, TP1 vs TP2+SP | 3701850, 3701851, 3701895, 3701896 | static (params+grads+optimizer) after iteration 2 / peak allocated: TP1 8L 31.3 / 47.3 GB, 20L 63.0 / 82.1 GB; TP2+SP 8L 28.0 / 34.0 GB, 20L 58.8 / 63.9 GB. Steady iteration (GBS 32): TP1 1.03 s → 2.11 s, TP2 1.44 s → 2.65 s. See "Memory model" below |
| dry run PP2 MBS2 at 8 nodes | 3701943 | **PASSED** `world_size=32 data_parallel_size=16` |
| PP2 MBS2 smoke (2 nodes, 4-layer model) | 3701942 → **3701976** | 1st landed on nid007210 again (Slurm prolog hung, `Reason=Prolog`) → cancelled, resubmitted with `--exclude=nid007210`. **PASSED** (8.7 min): 20 steps, eval at 10/20, checkpoints at 15/20 saved per PP stage, nsys report, netlog summary. New traffic in the logs: 2-rank/2-node P2P = pipeline activations (33 MB per microbatch each way), 2-rank/2-node AllReduce = word-embedding grads between first and last stage (the MTP layer on the last stage uses the embedding), ~1 GB per rank per step |
| HybridEP dispatcher, EP4 inside a node (2 nodes, 20 layers, 256 experts, MBS1) | 3701972 | **works** (its InfiniBand path is only used when the expert group spans nodes); same memory as alltoall (82.0 vs 82.1 GB peak); iteration 6: 1.48 s vs 2.11 s with alltoall, iteration 3: 1.92 vs 2.15 s (short run, to be confirmed at 8 nodes); ~35 s extra JIT compile in the first iteration |
| W&B offline smoke (2 nodes, PP1) | 3702103 | **PASSED** (6.4 min): offline run written under `$RUN_DIR/wandb`, contains `lm loss`, `mtp_1 loss`, LR, grad norm, `throughput/*`, `memory/*` and the per-step `net/*` metrics; run closed with exit code 0. Also re-tested the job-script edits (pre-step lock) |
| **full model, 8 nodes, PP2 MBS1 alltoall**, GBS 512, NETLOG=1 | 3702069 | **fits**: peak allocated 62.5 GB (stage 0, rank 0) / 68.1 GB (stage 1, rank 16), static 39.0 / 44.4 GB. Iterations 11–30: **12.86 s mean (11.6–15.4) = 115 TFLOP/s/GPU = 5,100 tokens/s/GPU**. Then ran a needless ~6 min final validation + test evaluation (fixed below) and, after that, a **second copy of the training** on every rank (see section 4) → cancelled |
| full model, 8 nodes, PP2 **MBS2** alltoall | 3702070 | **OOM** in iteration 1 (NCCL `cuMemAlloc` failed in the MoE-metrics all-reduce) — as the 3702069 numbers predict (~86 / ~92 GB) |
| full model, 8 nodes, PP2 MBS1 / MBS2 **HybridEP** | 3702071 / 3702072 | **MBS1: 11.68 s/it (its 11–30) = 127 TFLOP/s/GPU = 5,612 tokens/s/GPU, +10% over alltoall**, same memory (62.1 / 68.2 GB peak), first iteration +60 s (HybridEP JIT). MBS2 cancelled (same memory as the OOM above) |
| full model, 8 nodes, **PP4 MBS2** alltoall | 3702285 → 3702328 | queued (estimate: ~74 GB stage 0, ~69 GB last stage; pipeline bubble ~9% at GBS 512); resubmitted without the final evaluation (below) |
| dry run, train mode, 8 nodes, 1T tokens with the default batch ramp | 3702267 | **PASSED**: Hydra parsed the quoted schedule, patched validation passed (`global_batch_size: 2560`), `train_iters: 99958`, `lr_warmup_steps: 256000` samples (= recipe's 500 × 512), `lr_decay_steps: 244142080` samples (= 1T tokens) |
| batch-ramp smoke (2 nodes, PP2, `GBS_SCHEDULE="0:16 400K:32 1.2M:48"`, 20 its: steps after it 7 and 13; graphs captured for 4, 8, 12 microbatches), W&B offline | 3702268 | **first step OK**: Megatron-Core printed the 3-step schedule (thresholds 97 / 292 samples, as `tools/batch_schedule.py`); its 1–7 at GBS 16 with LR warmup; `global batch size 16 -> 32 after iteration 7: checkpoint, exit, restart`; checkpoint at 7; one `Pretrain stopped (exit condition reached)` line per rank (patch 0002), srun exit 0. **In-job restart not done**: 31 min left < my 35 min threshold → threshold lowered to 25 min (5 min start + 15 min exit margin), message fixed |
| same, **W&B online** (API key now in config.sh) | 3702279 → 3702315 → 3702333 | 3702279 cancelled (old threshold baked into its script); 3702315 (60 min) never starts: debug QOS allows 90 node-minutes per job (`QOSMaxNodeMinutesPerJob`) → 3702333 with 45 min. **W&B online works** from compute nodes (logged in from `WANDB_API_KEY`, run in project `qwen35-35b-a3b-cscs`). **Ramp works across in-job restarts**: its 1–7 at GBS 16 (LR warmup 2e-5 → 1.4e-4) → checkpoint + stop → restart with 37 min left, `wandb: Resuming run` → its 8–13 at GBS 32, LR advancing per sample (1.8e-4 … peak 3e-4 when 240 samples were consumed, then cosine: 2.954e-4, 2.819e-4) → stop at consumed 304 (= 112 + 6 × 32) → restart with 30 min left → loaded the iteration-13 checkpoint, its 14–20 at GBS 48, LR cosine down to exactly `min_lr` 3.0e-5 at the last iteration (decay length in samples is right), checkpoints at 15 and 20, eval + validation/test, W&B run resumed in all 3 processes and synced, every srun exit 0: **PASSED** (21.5 min). Each restart ~4 min (first iteration ~240 s: Triton autotune of the GDN kernels is redone per process) |
| full model, 8 nodes, PP2 MBS1, **constant GBS 2560** (final ramp size: 160 microbatches per rank, CUDA graphs captured for all of them), 8 steps | 3702312 → 3702329 | queued: memory + capture time + step time at the final batch |
| full model, 8 nodes, PP2 MBS1, GBS 512, **CUDA graphs off** | 3702313 → 3702330 | queued: what the TE-scoped graphs buy (vs 3702069) |
| smoke (2 nodes, PP2 default) with the new job body: scripts from the submission snapshot, `$0` copy for resubmission | 3702363 | **PASSED** (8 min): `CMD: python .../code/src-c15e44d82e8a/train/pretrain.py --profile .../src-c15e44d82e8a/conf/...`, `$RUN_DIR/logs/sbatch-3702363.sh` written (resubmission source), checkpoints 15/20, eval, srun exit 0 |
| full model, 8 nodes, default layout (PP2 MBS1 HybridEP), **NETLOG=2 + nsys** steps 15–17 on ranks 0 and 16 (one per pipeline stage), 25 steps | 3702414 | queued: logging overhead vs 3702071 and NCCL-inspector volume at full scale |
| **real-data smoke**: 2 nodes, smoke blend `lightning35_qwen35_smoke.json` (all 23 sources, 24 prefixes), new defaults (HybridEP, snapshot scripts, Triton autotune cache) | 3702441 | **PASSED** (first training on the tokenized Lightning data): blend weights 23 × 4.35%, GPTDataset splits built (640/192/64 samples), HybridEP active; loss on real text 12.83 → 7.67 in 20 steps (mock data goes to ~0), validation 8.71 (it 10) → 7.64 (it 20), test 7.72; first iteration 308 s |
| **full-scale lifecycle test** (train mode, 8 nodes, real blend 594 shards, default layout PP2/HybridEP, `GBS_SCHEDULE="0:512 60M:1024"`: step after it 29, 50 its = 149M tokens, checkpoints every 10, eval every 20, 40-min jobs → expected: in-job restart at the batch step, walltime exit, auto-resubmit, resume, completion; W&B online, NETLOG=2) | 3702526 | queued |
| **full tokenization** of the Lightning blend (`sbatch data/tokenize.sbatch`: 23 one-node tasks, ≤ 25B tokens per source; resumable, extend with `TARGET_TOKENS`) | 3702316_[0-22] | **23/23 COMPLETED** (7–14 min each, up to 55M tokens/s per node): **530B tokens, 594 shards, 2.0 TB** on iopsstor. 21 sources ~25B each; Wiki-Rewrite 7.9B, Scientific-Coding 1.25B, InfiniByte-Reasoning 19.8B, CC-v2.1 MHQ 16.8B (whole subsets) |
| blend (`python3.11 data/build_blend.py --train-tokens 1e12`, login node, reads receipts only) | – | `$DATA_ROOT/blends/lightning35_qwen35.json`: uniform 4.35% per source (Lightning weights). At 1T tokens: most sources ~1.7 epochs of what is tokenized, Wiki-Rewrite 5.5, Scientific-Coding 34.7 (the recipe's uniform weighting of a 1.25B-token source) |

Findings from the smoke-run network logs: communicators are unnamed (`comm_name: unknown`) but identifiable by size/span — 4 ranks/1 node P2P = MoE all-to-all (NVLink), 8 ranks/2 nodes AG+RS = dense DP, 2 ranks/2 nodes AG+RS = expert DP, small AllGather/AllReduce = token counts / loss. A few P2P records carry negative sizes wrapped to uint64 (inspector quirk) → counted as `invalid_size_records`.

### Memory model (MBS1, seq 4096) — corrected with the 8-node measurements

**Full model, 8 nodes, PP2 (job 3702069):** static 39.0 GB (stage 0) / 44.4 GB (stage 1; + output
layer, MTP layer, embedding copy for MTP); activations 23.5 GB on stage 0 (2 microbatches in flight × 20
layers → **~0.57 GB per layer per 4096-token microbatch**) and 23.8 GB on stage 1 (21 layers + **~12 GB
for the two 248k-vocab logit tensors**). The per-layer value of 0.26 GB derived from the 2-node probes
below was too low (probe peaks are not a clean slope); use the 8-node numbers. Consequences: MBS2 at
PP2 ≈ 86 / 92 GB (OOM confirmed, job 3702070); MBS2 needs PP4.

Earlier analysis from the 2-node probes (kept for the record):

- **Static memory is dominated by the routed experts**, which TP does not shard (ETP=1): +2.6 GB per
  layer at EP4 / expert-DP 2 for both TP1 and TP2. TP2 only halves the dense part (~7% of the parameters).
- **Activations**: the two 248k-vocab logit tensors (main head + MTP head, fp32 softmax kept for the
  backward) cost ~13.7 GB per microbatch at TP1; each transformer layer adds only ~0.26 GB.
  (Measured as peak minus static; the Bridge formula estimator does not model GDN layers or top-k routing.)
- **Time**: ~23 ms per layer per microbatch at TP1 (~70 TFLOP/s on the layers), while the 248k-vocab
  output GEMMs run at ~600 TFLOP/s. TP2 is 25–40% slower per GPU (more, smaller kernels plus TP
  communication on a hidden size of only 2048), so it is not a good trade.
- Extrapolated to the full model on 8 nodes (dense DP 32, expert DP 8): static ≈ 1.37 GB/layer
  → ~61 GB at PP1, + ~24 GB activations ≈ 85 GB peak. Job 3701785 died with `Triton Error [CUDA]: out of
  memory` (a kernel load during the GDN autotune in the first backward, not a PyTorch allocation): the
  CUDA context, NCCL buffers of ~10 communicators and autotune scratch live outside PyTorch's numbers,
  so **~80 GB peak allocated is the practical ceiling** on a 95.6 GB GH200 (also leaves room for routing
  imbalance with learned routing). At PP2: ~30 GB (stage 0) / ~33 GB (stage 1) static.
  PP2 + MBS2: stage 0 ≈ 30 + 2 microbatches × 20 layers × 0.52 GB ≈ 51 GB; stage 1 ≈ 33 + 21 × 0.52 + 27 (logits)
  ≈ 72 GB. PP2 + MBS4 does not fit on the last stage (logits alone ≈ 55 GB).

## 6. Docs and commits

- `3859f95` (first milestone commit, on `main`, not pushed): everything up to the PP2 default and the
  W&B offline mode; `core*` (0-byte core files of crashed ranks, written to the job's working
  directory) added to `.gitignore` on request.
- README: `./launch.sh train <nodes> <tokens>`, "Global batch size ramp" section, `SBATCH_EXCLUDE` for broken nodes.
- README "Parallel layout on GH200" rewritten: parallel folding (EP for experts + TP/SP for attention is supported, measured, not used), memory model from the 8-node runs, MBS, measured layout table.
