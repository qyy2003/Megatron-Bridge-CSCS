#!/usr/bin/env python3
"""Pretrain Qwen3.5/3.6-35B-A3B from scratch with Megatron-Bridge on CSCS Alps (GH200).

A thin entry point around the official Megatron-Bridge recipe, following the same
pattern as the Nemotron 3.5 Lightning stage-0 ``train.py``:

  1. build the official recipe (default ``qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config``),
  2. merge the cluster profile YAML (``conf/*.yaml``): every deviation from the GB200
     recipe needed on GH200 + Slingshot lives there, documented,
  3. apply the data blend produced by ``data/build_blend.py`` (no blend = mock data),
  4. apply Hydra-style ``KEY=VALUE`` overrides; unknown keys are an error (Bridge itself only
     warns and silently skips them),
  5. optionally enable the nsys capture window and the per-step network-traffic callback,
  6. ``pretrain(cfg, forward_step, callbacks)``.

Launched once per GPU by ``srun`` (see ``train/rank_wrapper.sh``); Megatron-Bridge derives
RANK / WORLD_SIZE / LOCAL_RANK / MASTER_* from the Slurm environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path


logger = logging.getLogger("mbcscs.pretrain")

DEFAULT_RECIPE = "qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config"
CXI_TELEMETRY = "/sys/class/cxi/{dev}/device/telemetry/{counter}"
CXI_BYTE_COUNTERS = ("hni_sts_tx_ok_octets", "hni_sts_rx_ok_octets")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse runner arguments; everything of the form KEY=VALUE is a config override."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE, help="Exported megatron.bridge.recipes function.")
    parser.add_argument("--profile", action="append", default=[], help="YAML merged onto the recipe (repeatable).")
    parser.add_argument("--blend", help="Blend JSON from data/build_blend.py. Omit for mock data.")
    parser.add_argument("--nsys-steps", help="START:END global steps captured by nsys (needs the nsys wrapper).")
    parser.add_argument("--nsys-ranks", default="0", help="Comma-separated global ranks run under nsys.")
    parser.add_argument("--net-log-dir", help="Write per-step timestamps + NIC byte deltas here (per node).")
    parser.add_argument("--batch-step-marker", help="With train.step_batch_size_schedule: stop before each "
                        "batch-size step and write this file (the job restarts training, see BatchSizeStepCallback).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the config, then exit.")
    args, rest = parser.parse_known_args(argv)
    overrides = []
    for item in rest:
        if item.startswith("-") or "=" not in item:
            parser.error(f"unexpected argument {item!r}; config overrides must be KEY=VALUE")
        overrides.append(item)
    return args, overrides


def load_recipe(name: str):
    """Return the ConfigContainer built by an exported Megatron-Bridge recipe function."""
    recipes = importlib.import_module("megatron.bridge.recipes")
    if not hasattr(recipes, name):
        raise SystemExit(f"recipe {name!r} is not exported by megatron.bridge.recipes")
    return getattr(recipes, name)()


def check_override_keys(cfg, overrides: list[str]) -> None:
    """Fail on override paths that do not exist in the config (typos would be skipped silently)."""
    import dataclasses

    for item in overrides:
        key = item.split("=", 1)[0].lstrip("+~")
        obj = cfg
        parts = key.split(".")
        for i, part in enumerate(parts):
            if not hasattr(obj, part):
                where = ".".join(parts[:i]) or "<root>"
                raise SystemExit(f"override {item!r}: {type(obj).__name__} at {where!r} has no field {part!r}")
            obj = getattr(obj, part)
            if obj is None and i < len(parts) - 1:
                raise SystemExit(f"override {item!r}: {'.'.join(parts[: i + 1])} is None; set it in Python")
            if i < len(parts) - 1 and not dataclasses.is_dataclass(obj):
                break


def yaml_keys(path: str) -> list[str]:
    """Flatten a profile YAML into KEY=VALUE-style paths for check_override_keys."""
    from omegaconf import OmegaConf

    keys: list[str] = []

    def walk(prefix: str, node) -> None:
        if isinstance(node, dict) and node and "_target_" not in node:
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            keys.append(f"{prefix}=")

    walk("", OmegaConf.to_container(OmegaConf.load(path), resolve=False))
    return keys


def apply_blend(cfg, blend_path: str) -> None:
    """Point the dataset at the tokenized Lightning blend written by data/build_blend.py."""
    from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths

    spec = json.loads(Path(blend_path).read_text())
    if {"train", "valid", "test"} & spec.keys():
        blend, blend_per_split, split = get_blend_fields_from_data_paths(per_split_data_args_path=blend_path)
    else:
        blend, blend_per_split, _ = get_blend_fields_from_data_paths(data_paths=spec["data_paths"])
        split = spec.get("split", "9990,8,2")
    cfg.dataset.blend = blend
    cfg.dataset.blend_per_split = blend_per_split
    cfg.dataset.split = split
    n_prefixes = len(blend[0]) if blend else sum(len(b[0]) for b in blend_per_split if b)
    logger.info("data blend %s: %d prefixes, split=%s", blend_path, n_prefixes, split)


def widen_blend_surplus(cfg) -> None:
    """Give the blended validation/test splits room for Megatron's per-prefix rounding.

    Megatron builds every prefix's dataset with ``ceil(N * w) * (1 + mid_level_dataset_surplus)``
    samples but sizes the blended split as ``sum_p ceil(N * w_p)``, i.e. up to one sample per
    prefix more than N. With ~600-1100 prefixes and an evaluation split of a few thousand samples
    the default surplus (0.005) is far too small and the build fails with "The test blend
    oversamples the contributing datasets" (job 3702526). Allow one extra sample per prefix in
    the smallest evaluation split, +25%.
    """
    ds, val = cfg.dataset, cfg.validation
    if ds.blend is not None:
        n_prefixes = len(ds.blend[0])
    elif ds.blend_per_split is not None:
        n_prefixes = max(len(b[0]) for b in ds.blend_per_split if b)
    else:
        return
    gbs = val.eval_global_batch_size or cfg.train.global_batch_size
    sizes = []
    if val.eval_iters:
        sizes.append(val.eval_iters * gbs)  # test split
        if val.eval_interval:
            sizes.append((cfg.train.train_iters // val.eval_interval + 1) * val.eval_iters * gbs)
    if not sizes:
        return
    needed = 1.25 * n_prefixes / min(sizes)
    if needed > ds.mid_level_dataset_surplus:
        logger.info("dataset.mid_level_dataset_surplus %.4f -> %.4f (%d prefixes, smallest eval split %d samples)",
                    ds.mid_level_dataset_surplus, needed, n_prefixes, min(sizes))
        ds.mid_level_dataset_surplus = needed


def build_config(args: argparse.Namespace, overrides: list[str]):
    """Recipe -> cluster profile -> data blend -> CLI overrides -> profiling."""
    from megatron.bridge.training.config import ProfilingConfig
    from megatron.bridge.training.mixed_precision import get_mixed_precision_config
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    cfg = load_recipe(args.recipe)
    # Recipes may name the precision recipe as a string ("bf16_mixed"); at setup Bridge copies its
    # fields over cfg.ddp/cfg.optimizer, so materialize it to make its fields overridable.
    if isinstance(cfg.mixed_precision, str):
        cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    for profile in args.profile:
        check_override_keys(cfg, yaml_keys(profile))
        cfg = process_config_with_overrides(cfg, config_filepath=profile)
        logger.info("merged profile %s", profile)
    if args.blend:
        apply_blend(cfg, args.blend)
    if overrides:
        check_override_keys(cfg, overrides)
        cfg = process_config_with_overrides(cfg, cli_overrides=overrides)

    # The dataset owns the sequence length (same rule as scripts/training/run_recipe.py).
    if cfg.dataset.seq_length is not None and cfg.model.seq_length != cfg.dataset.seq_length:
        cfg.model.seq_length = cfg.dataset.seq_length
    widen_blend_surplus(cfg)

    if args.nsys_steps:
        start, end = (int(x) for x in args.nsys_steps.split(":"))
        cfg.profiling = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=start,
            profile_step_end=end,
            profile_ranks=[int(r) for r in args.nsys_ranks.split(",") if r != ""],
        )
    return cfg


def _gpu_numa_nodes() -> tuple[str, ...]:
    """CPU-less NUMA nodes with memory, in node order (= GPU order): the GPUs' HBM on GH200."""
    global _GPU_NUMA_NODES
    if _GPU_NUMA_NODES is None:
        root, nodes = "/sys/devices/system/node", []
        try:
            names = sorted((n for n in os.listdir(root) if n.startswith("node") and n[4:].isdigit()),
                           key=lambda n: int(n[4:]))
        except OSError:
            names = []
        for n in names:
            try:
                with open(f"{root}/{n}/cpulist") as f:
                    if f.read().strip():
                        continue
                with open(f"{root}/{n}/meminfo") as f:
                    if any(len(p) >= 4 and p[2] == "MemTotal:" and int(p[3]) > 0 for p in map(str.split, f)):
                        nodes.append(n)
            except (OSError, ValueError):
                pass
        _GPU_NUMA_NODES = tuple(nodes)
    return _GPU_NUMA_NODES


_GPU_NUMA_NODES: tuple[str, ...] | None = None


class NetTrafficCallback:
    """Per-step wall-clock boundaries and Slingshot NIC byte counters.

    * local rank 0 of every node appends one JSON line per step to
      ``<log_dir>/<hostname>.jsonl`` with the step's start/end time (ns since epoch) and the
      TX/RX bytes each Slingshot NIC of that node moved during the step. These timestamps are
      what aligns the NCCL-inspector and netmon logs with training iterations.
    * the W&B rank (N-1) additionally logs ``net/*`` metrics for its own node, so network
      throughput sits next to the loss curve.

    The counters are the NIC hardware totals (NCCL + Lustre + everything else on the node).
    Each record also carries host memory: the node's MemAvailable / Shmem and this rank's
    VmRSS / VmHWM (GB), to catch host-memory growth before the kernel OOM-kills a rank.
    """

    def __init__(self, log_dir: str) -> None:
        from megatron.bridge.utils.common_utils import get_local_rank_preinit, get_rank_safe, get_world_size_safe

        self.enabled = get_local_rank_preinit() == 0 or get_rank_safe() == get_world_size_safe() - 1
        self.write_file = get_local_rank_preinit() == 0
        self.nics = sorted(d for d in os.listdir("/sys/class/cxi")) if os.path.isdir("/sys/class/cxi") else []
        self.fh = None
        if self.enabled and self.write_file:
            os.makedirs(log_dir, exist_ok=True)
            self.fh = open(os.path.join(log_dir, f"{socket.gethostname()}.jsonl"), "a", buffering=1)
        self.t0 = 0
        self.c0: dict[str, list[int]] = {}
        self.step = 0

    def _counters(self) -> dict[str, list[int]]:
        out = {}
        for dev in self.nics:
            vals = []
            for name in CXI_BYTE_COUNTERS:
                try:
                    with open(CXI_TELEMETRY.format(dev=dev, counter=name)) as f:
                        vals.append(int(f.read().split("@", 1)[0]))
                except (OSError, ValueError):
                    vals.append(-1)
            out[dev] = vals
        return out

    @staticmethod
    def _children_private_gb() -> dict[str, float]:
        """Private dirty memory of this rank's child processes (dataloader workers, checkpoint
        writers). Their per-prefix attention-mask caches once ran the nodes out of host memory
        (see dataset.create_attention_mask in the profile); constant since that fix.
        """
        pids: set[str] = set()
        try:
            for tid in os.listdir("/proc/self/task"):
                with open(f"/proc/self/task/{tid}/children") as f:
                    pids.update(f.read().split())
        except OSError:
            return {}
        kb = 0
        for pid in pids:
            try:
                with open(f"/proc/{pid}/smaps_rollup") as f:
                    for line in f:
                        if line.startswith("Private_Dirty:"):
                            kb += int(line.split()[1])
            except OSError:
                pass
        return {"workers_private": round(kb / 1e6, 2), "workers": len(pids)}

    @staticmethod
    def _gpu_memory_gb() -> dict[str, float]:
        """This rank's GPU memory (the W&B/TensorBoard memory metrics come from the last rank only;
        with PP the first stage can be the fuller one)."""
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_reserved": round(torch.cuda.memory_reserved() / 1e9, 2),
            "gpu_max_reserved": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "gpu_max_allocated": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }

    @staticmethod
    def _gpu_hbm_free_gb() -> dict[str, float]:
        """Lowest free HBM over the node's GPUs. On GH200 each GPU's memory is a CPU-less NUMA
        node (4/12/20/28) and CUDA allocations show up there as used memory, so this covers every
        rank of the node, also the expert-parallel ranks that get the busiest experts (dropless
        MoE: their activations grow with routing imbalance), unlike the counters above."""
        free = []
        for node in _gpu_numa_nodes():
            try:
                with open(f"/sys/devices/system/node/{node}/meminfo") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 4 and parts[2] == "MemFree:":
                            free.append(int(parts[3]) / 1e6)  # kB -> GB
            except OSError:
                pass
        return {"gpu_hbm_min_free": round(min(free), 2)} if free else {}

    @staticmethod
    def _host_memory_gb() -> dict[str, float]:
        out: dict[str, float] = {}
        for path, keys in (("/proc/meminfo", ("MemAvailable", "Shmem")), ("/proc/self/status", ("VmRSS", "VmHWM"))):
            try:
                with open(path) as f:
                    for line in f:
                        key, _, rest = line.partition(":")
                        if key in keys:
                            out[key] = round(int(rest.split()[0]) / 1e6, 2)  # kB -> GB
            except OSError:
                pass
        return out

    def on_train_step_start(self, context) -> None:
        if not self.enabled:
            return
        self.step = context.state.train_state.step + 1
        self.c0 = self._counters()
        self.t0 = time.time_ns()

    def on_train_step_end(self, context) -> None:
        if not self.enabled:
            return
        t1 = time.time_ns()
        c1 = self._counters()
        dt = max(t1 - self.t0, 1) / 1e9
        delta = {d: [b - a if a >= 0 and b >= 0 else None for a, b in zip(self.c0[d], c1[d])] for d in c1}
        mem = {**self._host_memory_gb(), **self._children_private_gb(), **self._gpu_memory_gb(),
               **self._gpu_hbm_free_gb()}
        if self.fh is not None:
            rec = {"iter": self.step, "t0_ns": self.t0, "t1_ns": t1, "nic_tx_rx_bytes": delta, "host_mem_gb": mem}
            self.fh.write(json.dumps(rec) + "\n")
        wb = context.state.wandb_logger
        if wb is not None:
            tx = sum(v[0] for v in delta.values() if v[0] is not None)
            rx = sum(v[1] for v in delta.values() if v[1] is not None)
            wb.log(
                {
                    "net/node_tx_GB": tx / 1e9,
                    "net/node_rx_GB": rx / 1e9,
                    "net/node_tx_GBps": tx / 1e9 / dt,
                    "net/node_rx_GBps": rx / 1e9 / dt,
                    "net/step_wallclock_s": dt,
                    **{f"node/{k}_GB" if k != "workers" else "node/workers": v for k, v in mem.items()},
                },
                step=self.step,
            )


def freeze_gc(context=None) -> None:
    """Move every object that exists before the dataloader workers fork into the permanent GC
    generation, so the workers' garbage collector never touches (and copy-on-write copies) the
    parent's heap: model, optimizer and checkpoint state."""
    import gc

    gc.collect()
    gc.freeze()


def trim_host_memory(context=None) -> None:
    """Return freed heap memory to the OS (registered after the checkpoint load and every save).

    Loading and saving a checkpoint stage tens of GB per rank in host memory, and glibc keeps the
    freed arenas; with 4 ranks per node the host ran out of memory a few checkpoint operations
    later (job 3703471: rank OOM-killed by the kernel).
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class BatchSizeStepCallback:
    """End the run cleanly at the last iteration before each step of the batch-size schedule.

    With PP > 1 the TE-scoped CUDA graphs are captured once per microbatch slot, in the 1F1B
    order of the microbatch count at capture time, and share input/output buffers according
    to that order; replaying them for more microbatches is not safe. So instead of growing the
    batch in-process, this sets ``train.exit_interval`` to the current iteration: Bridge saves
    a checkpoint and returns normally, rank 0 writes ``marker``, and ``train/job_body.sh``
    starts training again in the same allocation. The new process loads the checkpoint, takes
    the new batch size from ``consumed_train_samples`` and captures its graphs for it.
    """

    def __init__(self, marker: str) -> None:
        from megatron.bridge.utils.common_utils import get_rank_safe

        self.marker = marker
        self.rank = get_rank_safe()

    def on_train_step_end(self, context) -> None:
        from megatron.core import num_microbatches_calculator as nmc

        calc = nmc._GLOBAL_NUM_MICROBATCHES_CALCULATOR
        if not isinstance(calc, nmc.StepBatchsizeNumMicroBatchesCalculator):
            return
        train_cfg, ts = context.state.cfg.train, context.state.train_state
        # Bridge advances step and consumed_train_samples after this callback.
        current = calc.get_current_global_batch_size()
        upcoming = calc._get_batch_size_for_samples(ts.consumed_train_samples + current)
        if upcoming == current or ts.step + 1 >= train_cfg.train_iters:
            return
        train_cfg.exit_interval = ts.step + 1
        if self.rank == 0:
            info = {"iteration": ts.step + 1, "consumed_samples": ts.consumed_train_samples + current,
                    "global_batch_size": current, "next_global_batch_size": upcoming}
            Path(self.marker).write_text(json.dumps(info) + "\n")
            logger.info("global batch size %d -> %d after iteration %d: checkpoint, exit, restart",
                        current, upcoming, ts.step + 1)


def main(argv: list[str] | None = None) -> None:
    """Build the config and run pretraining."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    args, overrides = parse_args(argv)
    cfg = build_config(args, overrides)

    if args.dry_run:
        # Same validation the real run performs before initializing (parallel sizes, batch
        # divisibility, precision/overlap resolution), for the WORLD_SIZE given by launch.sh.
        from megatron.bridge.training.config import runtime_config_update

        runtime_config_update(cfg)
        cfg.print_yaml()
        logger.info("dry run OK: world_size=%s data_parallel_size=%s", os.environ.get("WORLD_SIZE"),
                    getattr(cfg, "data_parallel_size", "?"))
        return

    from megatron.bridge.training.callbacks import CallbackManager
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    callbacks = CallbackManager()
    if os.environ.get("MBCSCS_NO_MALLOC_TRIM") != "1":  # only for A/B memory tests
        callbacks.register("on_train_start", trim_host_memory)
        callbacks.register("on_checkpoint_save", trim_host_memory)
        callbacks.register("on_data_init_start", freeze_gc)
    if args.net_log_dir:
        net = NetTrafficCallback(args.net_log_dir)
        callbacks.register("on_train_step_start", net.on_train_step_start)
        callbacks.register("on_train_step_end", net.on_train_step_end)
    if args.batch_step_marker and cfg.train.step_batch_size_schedule:
        if not (cfg.checkpoint.save and cfg.checkpoint.load):
            raise SystemExit("--batch-step-marker needs checkpoint.save and checkpoint.load (the restart resumes)")
        callbacks.register("on_train_step_end", BatchSizeStepCallback(args.batch_step_marker).on_train_step_end)

    pretrain(cfg, forward_step, callbacks=callbacks)

    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
