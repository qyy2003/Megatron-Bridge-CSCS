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

    if args.nsys_steps:
        start, end = (int(x) for x in args.nsys_steps.split(":"))
        cfg.profiling = ProfilingConfig(
            use_nsys_profiler=True,
            profile_step_start=start,
            profile_step_end=end,
            profile_ranks=[int(r) for r in args.nsys_ranks.split(",") if r != ""],
        )
    return cfg


class NetTrafficCallback:
    """Per-step wall-clock boundaries and Slingshot NIC byte counters.

    * local rank 0 of every node appends one JSON line per step to
      ``<log_dir>/<hostname>.jsonl`` with the step's start/end time (ns since epoch) and the
      TX/RX bytes each Slingshot NIC of that node moved during the step. These timestamps are
      what aligns the NCCL-inspector and netmon logs with training iterations.
    * the W&B rank (N-1) additionally logs ``net/*`` metrics for its own node, so network
      throughput sits next to the loss curve.

    The counters are the NIC hardware totals (NCCL + Lustre + everything else on the node).
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
        if self.fh is not None:
            rec = {"iter": self.step, "t0_ns": self.t0, "t1_ns": t1, "nic_tx_rx_bytes": delta}
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
                },
                step=self.step,
            )


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
