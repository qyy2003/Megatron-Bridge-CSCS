#!/usr/bin/env python3
"""Summarize the network logs of one job: $RUN_DIR/netlogs/job_<id>.

Reads
  steps/<host>.jsonl          per-step wall-clock window + Slingshot NIC byte deltas (trainer callback)
  netmon/<host>.jsonl         NIC / NVLink counter samples (tools/netmon.py)
  nccl-inspector/*.log[.gz]   one record per NCCL collective / P2P op on every rank (NETLOG>=2)
Writes <dir>/summary/
  steps_nic.csv               iter, host, seconds, NIC tx/rx bytes and GB/s per node
  nccl_ops.csv                per communicator x op: records, bytes, GPU time, mean bus bandwidth
  nccl_steps.csv              per iteration x communicator x op (attributed by timestamp; exact
                              with NETLOG=3, otherwise by dump time, i.e. within ~1 dump interval)
  netmon_rates.csv            per host and sample: NIC tx/rx GB/s, NVLink tx/rx GB/s per GPU
  summary.json                headline numbers

NCCL sizes are logical message sizes (what the op moves per rank); bytes on the wire come
from the NIC counters. Standard library only.

Usage: summarize_netlogs.py <NETLOG_DIR>
"""

import bisect
import collections
import csv
import gzip
import json
import os
import sys
from pathlib import Path


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass  # truncated last line of a killed job


def iter_json_objects(path, chunk=1 << 26):
    """Stream concatenated JSON objects (any whitespace in between) from a possibly gzipped file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    dec = json.JSONDecoder()
    buf, pos = "", 0
    with opener(path, "rt") as f:
        while True:
            data = f.read(chunk)
            buf = buf[pos:] + data
            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos >= len(buf):
                    break
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break  # incomplete object: read more
                yield obj
                pos = end
            if not data:
                return


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    out = root / "summary"
    out.mkdir(exist_ok=True)
    summary = {"netlog_dir": str(root)}

    # ---- per-step windows and NIC deltas (trainer callback) ----
    windows = collections.defaultdict(list)  # host -> [(t0_us, t1_us, iter)]
    step_rows = []
    for f in sorted((root / "steps").glob("*.jsonl")):
        host = f.stem
        for r in read_jsonl(f):
            dt = (r["t1_ns"] - r["t0_ns"]) / 1e9
            tx = sum(v[0] for v in r["nic_tx_rx_bytes"].values() if v[0] is not None)
            rx = sum(v[1] for v in r["nic_tx_rx_bytes"].values() if v[1] is not None)
            windows[host].append((r["t0_ns"] // 1000, r["t1_ns"] // 1000, r["iter"]))
            step_rows.append([r["iter"], host, round(dt, 4), tx, rx, round(tx / 1e9 / dt, 3), round(rx / 1e9 / dt, 3)])
    for host in windows:
        windows[host].sort()
    with open(out / "steps_nic.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "host", "seconds", "nic_tx_bytes", "nic_rx_bytes", "nic_tx_GBps", "nic_rx_GBps"])
        w.writerows(sorted(step_rows))
    if step_rows:
        n_hosts = len(windows)
        steady = [r for r in step_rows if r[0] > 3] or step_rows  # skip warm-up/compile steps
        summary["steps"] = {
            "iterations": len({r[0] for r in step_rows}), "hosts": n_hosts,
            "mean_step_s": round(sum(r[2] for r in steady) / len(steady), 3),
            "mean_node_tx_GB_per_step": round(sum(r[3] for r in steady) / len(steady) / 1e9, 3),
            "mean_node_rx_GB_per_step": round(sum(r[4] for r in steady) / len(steady) / 1e9, 3),
            "mean_node_tx_GBps": round(sum(r[5] for r in steady) / len(steady), 3),
        }

    def step_of(host, t_us, exact):
        ws = windows.get(host)
        if not ws:
            return None
        i = bisect.bisect_right(ws, (t_us, float("inf"), 0)) - 1
        if i < 0:
            return None
        t0, t1, it = ws[i]
        return it if (not exact or t_us <= t1) else None

    # ---- NCCL inspector ----
    ops = collections.defaultdict(lambda: [0, 0, 0, 0.0])        # (comm, n_ranks, nnodes, kind, op) -> n, bytes, us, busbw
    per_step = collections.defaultdict(lambda: [0, 0, 0])        # (iter, comm, kind, op) -> n, bytes, us
    n_records, n_invalid, exact_any, files = 0, 0, False, sorted((root / "nccl-inspector").glob("*.log*"))
    for f in files:
        for rec in iter_json_objects(f):
            hdr, meta = rec.get("header", {}), rec.get("metadata", {})
            if "coll_perf" in rec:
                kind, p = "coll", rec["coll_perf"]
                op, size, us, bw = p.get("coll"), p.get("coll_msg_size_bytes", 0), p.get("coll_exec_time_us", 0), p.get("coll_busbw_gbs", 0.0)
                ts = (p.get("event_trace_ts") or {}).get("coll_start_ts")
            elif "p2p_perf" in rec:
                kind, p = "p2p", rec["p2p_perf"]
                op, size, us, bw = p.get("p2p"), p.get("p2p_msg_size_bytes", 0), p.get("p2p_exec_time_us", 0), p.get("p2p_busbw_gbs", 0.0)
                ts = (p.get("event_trace_ts") or {}).get("p2p_start_ts")
            else:
                continue
            n_records += 1
            if size >= 1 << 63:  # negative size wrapped to uint64 (seen on a few small P2P ops)
                n_invalid += 1
                continue
            comm = hdr.get("comm_name", "unknown")
            key = (comm, hdr.get("n_ranks"), hdr.get("nnodes"), kind, op)
            a = ops[key]
            a[0] += 1; a[1] += size; a[2] += us; a[3] += bw
            exact = ts is not None
            exact_any |= exact
            it = step_of(meta.get("hostname"), ts if exact else meta.get("dump_timestamp_us", 0), exact)
            if it is not None:
                s = per_step[(it, comm, kind, op)]
                s[0] += 1; s[1] += size; s[2] += us
    with open(out / "nccl_ops.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comm_name", "n_ranks", "nnodes", "kind", "op", "records", "bytes", "gpu_time_s", "mean_busbw_GBps"])
        for k, (n, b, us, bw) in sorted(ops.items(), key=lambda kv: -kv[1][1]):
            w.writerow([*k, n, b, round(us / 1e6, 3), round(bw / max(n, 1), 3)])
    with open(out / "nccl_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "comm_name", "kind", "op", "records", "bytes", "gpu_time_s"])
        for k, (n, b, us) in sorted(per_step.items()):
            w.writerow([*k, n, b, round(us / 1e6, 3)])
    if files:
        summary["nccl_inspector"] = {
            "files": len(files), "records": n_records, "invalid_size_records": n_invalid, "bytes_on_disk": sum(f.stat().st_size for f in files),
            "step_attribution": "exact (verbose)" if exact_any else "by dump time (approximate)",
            "top_comm_ops_by_bytes": [
                {"comm": k[0], "n_ranks": k[1], "nnodes": k[2], "kind": k[3], "op": k[4], "records": v[0],
                 "GB": round(v[1] / 1e9, 3), "gpu_time_s": round(v[2] / 1e6, 3)}
                for k, v in sorted(ops.items(), key=lambda kv: -kv[1][1])[:15]],
        }

    # ---- netmon samples -> rates ----
    rate_rows = []
    for f in sorted((root / "netmon").glob("*.jsonl")):
        prev, prev_nv = None, None
        for r in read_jsonl(f):
            if r.get("type") != "sample":
                continue
            tx = sum(c.get("hni_sts_tx_ok_octets", 0) for c in r["cxi"].values())
            rx = sum(c.get("hni_sts_rx_ok_octets", 0) for c in r["cxi"].values())
            row = None
            if prev is not None:
                dt = (r["t_ns"] - prev[0]) / 1e9
                row = [f.stem, r["t_ns"], round(dt, 3), round((tx - prev[1]) / 1e9 / dt, 3), round((rx - prev[2]) / 1e9 / dt, 3), ""]
            nv = r.get("nvlink_kib")
            if nv:
                if prev_nv is not None and row is not None:
                    dtn = (r["t_ns"] - prev_nv[0]) / 1e9
                    per_gpu = {g: round(sum(l[0] for l in links) - sum(l[0] for l in prev_nv[1].get(g, links)), 0)
                               for g, links in nv.items()}
                    row[5] = json.dumps({g: round(v * 1024 / 1e9 / dtn, 3) for g, v in per_gpu.items()})
                prev_nv = (r["t_ns"], nv)
            if row:
                rate_rows.append(row)
            prev = (r["t_ns"], tx, rx)
    with open(out / "netmon_rates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host", "t_ns", "dt_s", "nic_tx_GBps", "nic_rx_GBps", "nvlink_tx_GBps_per_gpu"])
        w.writerows(rate_rows)
    if rate_rows:
        busy = [r for r in rate_rows if r[3] > 0.01] or rate_rows
        summary["netmon"] = {"samples": len(rate_rows), "peak_node_tx_GBps": max(r[3] for r in rate_rows),
                             "mean_busy_node_tx_GBps": round(sum(r[3] for r in busy) / len(busy), 3)}

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
