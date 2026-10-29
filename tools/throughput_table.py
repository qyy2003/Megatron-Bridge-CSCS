#!/usr/bin/env python3
"""Throughput table of finished jobs: tools/throughput_table.py [--first N] JOBID [JOBID ...]

Per job (from logs/*-<job>.log and the run's netlogs/job_<id>/steps/*.jsonl): run name, nodes,
MBS / GBS / GLOBAL_DP / optimizer from the name, median step time over steps >= --first (default 6),
tokens/s per GPU, TFLOP/s per GPU (Megatron's count), node-hours per billion tokens, the smallest
free HBM of any GPU (steps logs), and whether it crashed (OOM, Traceback).
"""
import argparse
import glob
import json
import re
import statistics
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]
RUN_ROOT = "/capstor/scratch/cscs/yqin/mbcscs/runs"


def one(job: str, first: int) -> dict:
    logs = glob.glob(str(WORKDIR / f"logs/*-{job}.log"))
    if not logs:
        return {"job": job, "error": "no log"}
    log = logs[0]
    name = Path(log).name[len("mbcscs-"): -len(f"-{job}.log")]
    text = open(log, errors="replace").read()
    m = re.search(r"START .*?nodes=(\d+)", text) or re.search(r"-(\d+)n-", name)
    nodes = int(m.group(1))
    steps, tflops, gbs, seq = {}, {}, None, 4096
    for m in re.finditer(r"\] iteration +(\d+)/.*?per iteration \(ms\): ([\d.]+).*?TFLOP/s/GPU\): ([\d.]+).*?global batch size: +(\d+)", text):
        it = int(m.group(1))
        steps[it] = float(m.group(2)) / 1000
        tflops[it] = float(m.group(3))
        gbs = int(m.group(4))
    sl = re.search(r"-sl(\d+)", name)
    seq = int(sl.group(1)) if sl else seq
    row = {"job": job, "run": name, "nodes": nodes}
    row["oom"] = bool(re.search(r"OutOfMemoryError|out of memory", text))
    row["traceback"] = "Traceback" in text
    good = [steps[i] for i in steps if i >= first]
    if good:
        t = statistics.median(good)
        gpus = nodes * 4
        row.update(step_s=round(t, 3), steps=len(good), gbs=gbs,
                   tok_s_gpu=round(gbs * seq / t / gpus),
                   tflops=round(statistics.median([tflops[i] for i in steps if i >= first]), 1),
                   node_h_per_btok=round(nodes * t / 3600 / (gbs * seq / 1e9), 2))
    hbm = []
    for f in glob.glob(f"{RUN_ROOT}/{name}/netlogs/job_{job}/steps/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["iter"] >= first:
                hbm.append(r["host_mem_gb"].get("gpu_hbm_min_free", float("nan")))
    if hbm:
        row["hbm_min_free_gb"] = round(min(hbm), 1)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=6, help="first step counted (after CUDA-graph capture)")
    ap.add_argument("jobs", nargs="+")
    args = ap.parse_args()
    rows = [one(j, args.first) for j in args.jobs]
    cols = ["job", "run", "nodes", "gbs", "step_s", "tok_s_gpu", "tflops", "node_h_per_btok", "hbm_min_free_gb", "oom"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
