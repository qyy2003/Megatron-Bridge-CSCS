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
| Layout on GH200 | EP=4 (intra-node), TP=PP=CP=1, MBS 1, GBS 512, seq 4096, bf16 grad reduce, ≥ 8 nodes | GB200 layout (EP=8 on 8 GPUs) needs ~93 GB static/GPU; EP=4 keeps all-to-all on NVLink |
| MoE dispatcher | `alltoall` instead of HybridEP | HybridEP/DeepEP cross-node need InfiniBand GPU-initiated RDMA |
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
| Image builds 1+2 (jobs 3700126, 3700426) failed: NCCL .deb `unable to install new version of './usr/include/nccl_device': File exists`. Removing the leftover paths (build 2) did not help | root cause found with a podman experiment on the NeMo base (job 3700693): **rootless podman on Alps cannot re-create, in the same build step, a directory deleted from a lower layer** (`mv: cannot overwrite non-directory ... with directory`); delete in one layer + create in the next works. CSCS's Alps step purges NeMo's NCCL .debs and re-installs NCCL in one RUN. Fix: run CSCS's `purge` step as its own layer first (stage 0 in `env/build_image.sbatch`), skip nvshmem4py (pip reinstall risk, unused); build 3 = job 3700780 |

## 5. Slurm tests

| Test | Job | Result |
|---|---|---|
| hardware probes | 3700028, 3700038 | done (section 2) |
| NeMo image inspection | 3700381 | done (section 2) |
| overlay delete/re-create experiments (ubuntu base, NeMo base) | 3700681, 3700693 | same-layer re-create fails on NeMo base, split layers OK |
| image build | 3700126, 3700426, 3700780 → 3700830 | 1+2: NCCL .deb install (section 4); 3: stage-0 script layout (scripts expect CSCS's flat `/opt/alps/` layout); 4: stage 0 OK, NCCL 2.30.7 .debs install OK, rest running |
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

Findings from the smoke-run network logs: communicators are unnamed (`comm_name: unknown`) but identifiable by size/span — 4 ranks/1 node P2P = MoE all-to-all (NVLink), 8 ranks/2 nodes AG+RS = dense DP, 2 ranks/2 nodes AG+RS = expert DP, small AllGather/AllReduce = token counts / loss. A few P2P records carry negative sizes wrapped to uint64 (inspector quirk) → counted as `invalid_size_records`.

### Memory model (from the probes, MBS1, seq 4096)

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
