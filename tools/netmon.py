#!/usr/bin/env python3
"""Per-node network counter sampler: Slingshot NICs (CXI) and NVLink, as JSON lines.

Every --interval seconds one record with the cumulative hardware counters of every
Slingshot NIC on the node:
  * bytes on the wire, total and split into HPC-protocol ("opt": NCCL/libfabric, Lustre)
    and Ethernet/IP ("ieee") frames,
  * packets sent/received per Slingshot traffic class,
plus the IP-level byte counters of the hsn netdevs, host memory from /proc/meminfo (kB:
MemAvailable, Shmem, AnonPages, Cached; a node running out of host memory gets OOM-killed), the free
HBM of every GPU (kB; on GH200 each GPU's memory is a CPU-less NUMA node and CUDA allocations show up
there as used, so this covers all ranks of the node) and,
every --nvlink-every samples, the per-link NVLink TX/RX counters (KiB) of each local GPU via
`nvidia-smi nvlink -gt d`.

Counters are cumulative since boot; consumers difference consecutive records
(tools/summarize_netlogs.py does). Standard library only, Python >= 3.6, so it runs
inside the container or on the host.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time


CXI_ROOT = "/sys/class/cxi"
CXI_COUNTERS = [
    "hni_sts_tx_ok_octets",
    "hni_sts_rx_ok_octets",
    "hni_sts_tx_octets_opt",
    "hni_sts_rx_octets_opt",
    "hni_sts_tx_octets_ieee",
    "hni_sts_rx_octets_ieee",
] + ["hni_pkts_sent_by_tc_%d" % i for i in range(8)] + ["hni_pkts_recv_by_tc_%d" % i for i in range(8)]
NVLINK_LINE = re.compile(r"Link (\d+): Data (Tx|Rx): (\d+) KiB")
NVLINK_GPU = re.compile(r"^GPU (\d+):")

_stop = False


def now_ns():
    return time.time_ns() if hasattr(time, "time_ns") else int(time.time() * 1e9)


def read_counter(path):
    with open(path) as f:
        return int(f.read().split("@", 1)[0])


def read_cxi(devices, names):
    out = {}
    for dev in devices:
        vals = {}
        for name in names:
            try:
                vals[name] = read_counter(os.path.join(CXI_ROOT, dev, "device", "telemetry", name))
            except (OSError, ValueError):
                pass
        out[dev] = vals
    return out


def read_hsn():
    out = {}
    base = "/sys/class/net"
    for dev in sorted(d for d in os.listdir(base) if d.startswith("hsn")):
        try:
            out[dev] = [read_counter(os.path.join(base, dev, "statistics", "tx_bytes")),
                        read_counter(os.path.join(base, dev, "statistics", "rx_bytes"))]
        except (OSError, ValueError):
            pass
    return out


def read_nvlink():
    """{gpu: [[tx_kib, rx_kib] per link]} or None if nvidia-smi is unavailable."""
    try:
        res = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=30, universal_newlines=True)
    except (OSError, subprocess.SubprocessError):
        return None
    gpus, cur = {}, None
    for line in res.stdout.splitlines():
        m = NVLINK_GPU.match(line.strip())
        if m:
            cur = m.group(1)
            gpus[cur] = {}
            continue
        m = NVLINK_LINE.search(line)
        if m and cur is not None:
            link, kind, val = m.groups()
            gpus[cur].setdefault(int(link), [0, 0])[0 if kind == "Tx" else 1] = int(val)
    return {g: [links[k] for k in sorted(links)] for g, links in gpus.items()} or None


MEMINFO_FIELDS = ("MemTotal", "MemAvailable", "Shmem", "AnonPages", "Cached")


def gpu_numa_nodes():
    """CPU-less NUMA nodes with memory, in node order (= GPU order): the GPUs' HBM on GH200."""
    root = "/sys/devices/system/node"
    try:
        names = sorted((n for n in os.listdir(root) if n.startswith("node") and n[4:].isdigit()),
                       key=lambda n: int(n[4:]))
    except OSError:
        return []
    nodes = []
    for n in names:
        try:
            with open(os.path.join(root, n, "cpulist")) as f:
                if f.read().strip():
                    continue
            if read_node_meminfo(n).get("MemTotal", 0) > 0:
                nodes.append(n)
        except OSError:
            pass
    return nodes


def read_node_meminfo(node):
    """{MemTotal, MemFree} in kB of one NUMA node ("Node 4 MemFree:   22343680 kB")."""
    out = {}
    with open(os.path.join("/sys/devices/system/node", node, "meminfo")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[2] in ("MemTotal:", "MemFree:"):
                out[parts[2][:-1]] = int(parts[3])
    return out


def read_gpu_hbm_free(nodes):
    out = []
    for n in nodes:
        try:
            out.append(read_node_meminfo(n).get("MemFree"))
        except OSError:
            out.append(None)
    return out


def read_meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in MEMINFO_FIELDS:
                    out[key] = int(rest.split()[0])
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="JSONL file to append to")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--nvlink-every", type=int, default=5, help="sample NVLink every N samples (0 = never)")
    ap.add_argument("--extra-counters", default="", help="regex of additional CXI telemetry counters")
    args = ap.parse_args()

    def stop(*_):
        global _stop
        _stop = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    devices = sorted(os.listdir(CXI_ROOT)) if os.path.isdir(CXI_ROOT) else []
    names = list(CXI_COUNTERS)
    if args.extra_counters and devices:
        rx = re.compile(args.extra_counters)
        names += [n for n in sorted(os.listdir(os.path.join(CXI_ROOT, devices[0], "device", "telemetry")))
                  if rx.search(n) and n not in names]

    gpu_nodes = gpu_numa_nodes()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", buffering=1) as fh:
        fh.write(json.dumps({"type": "meta", "host": socket.gethostname(), "t_ns": now_ns(),
                             "interval_s": args.interval, "cxi_devices": devices, "cxi_counters": names,
                             "gpu_numa_nodes": gpu_nodes,
                             "gpu_hbm_total_kb": [read_node_meminfo(n).get("MemTotal") for n in gpu_nodes],
                             "slurm_job_id": os.environ.get("SLURM_JOB_ID")}) + "\n")
        start, k = time.monotonic(), 0
        while not _stop:
            rec = {"type": "sample", "t_ns": now_ns(), "cxi": read_cxi(devices, names), "hsn": read_hsn(),
                   "mem_kb": read_meminfo(), "gpu_hbm_free_kb": read_gpu_hbm_free(gpu_nodes)}
            if args.nvlink_every and k % args.nvlink_every == 0:
                rec["nvlink_kib"] = read_nvlink()
            rec["t_end_ns"] = now_ns()
            fh.write(json.dumps(rec) + "\n")
            k += 1
            delay = start + k * args.interval - time.monotonic()
            while delay > 0 and not _stop:
                time.sleep(min(delay, 0.2))
                delay = start + k * args.interval - time.monotonic()


if __name__ == "__main__":
    main()
