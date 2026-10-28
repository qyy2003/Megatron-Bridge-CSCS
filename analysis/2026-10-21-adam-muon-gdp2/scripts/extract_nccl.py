"""NCCL-inspector records (NETLOG=2) of the three runs over the same 50 clean steps (401-450).

Writes, per run:
  data/nccl_<run>.csv     per communicator shape / op / size class: ops, GB and GPU kernel seconds per GPU per step
  data/comms_<run>.json   every communicator: size, members (global ranks), hosts of its members, and per op /
                          size class its mean ops, GB and kernel seconds per member per step
  data/placement_<run>.csv  share of each op class completing in each tenth of the step (rank 0's node)
coll_exec_time_us is GPU kernel time: it includes waiting for the slowest peer and overlaps compute.
"""
import bisect, collections, csv, glob, gzip, json, os, re, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = "/capstor/scratch/cscs/yqin/mbcscs/runs"
RUNS = {"adam": "train-q35a3b-gbs512to2560-sl4096/netlogs/job_3705060",
        "muon": "train-q35a3b-gbs512to2560-sl4096-muon-max500/netlogs/job_3706874",
        "muon-zero1-gdp2": "train-q35a3b-gbs512to2560-sl4096-muon-zero1-gdp2-max500/netlogs/job_3708048"}
STEPS = (401, 450)
DATA = Path(__file__).resolve().parents[1] / "data"
RX = re.compile(r'"id":"(\w+)","comm_name":"[^"]*","rank":(\d+),"n_ranks":(\d+),"nnodes":(\d+)\}.*?'
                r'"dump_timestamp_us":(\d+).*?"(?:coll|p2p)":"(\w+)".*?_msg_size_bytes":(\d+),"(?:coll|p2p)_exec_time_us":(\d+)')

def step_bounds(run):
    """(t0, t1) per step 401..450 from the first node's step log (all nodes agree within ms)."""
    f = sorted(glob.glob(f"{ROOT}/{RUNS[run]}/steps/*.jsonl"))[0]
    rec = [json.loads(l) for l in open(f)]
    rec = {r["iter"]: r for r in rec if STEPS[0] <= r["iter"] <= STEPS[1]}
    return [(rec[i]["t0_ns"] / 1e9, rec[i]["t1_ns"] / 1e9) for i in sorted(rec)]

def one_rank(args):
    path, bounds = args
    starts, ends = [b[0] for b in bounds], [b[1] for b in bounds]
    comms, agg, place = {}, collections.Counter(), collections.Counter()
    host = os.path.basename(path).split("-")[0]
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            m = RX.search(line)
            if not m:
                continue
            cid, rk, n, nn, ts, op, size, us = m.groups()
            comms[cid] = int(rk)
            t = int(ts) / 1e6
            i = bisect.bisect_left(ends, t)
            if i >= len(ends) or t < starts[i]:
                continue
            cls = "small" if int(size) < 65536 else "large"
            agg[(cid, op, cls, "n")] += 1; agg[(cid, op, cls, "bytes")] += int(size); agg[(cid, op, cls, "us")] += int(us)
            place[(cid, op, cls, min(int(10 * (t - starts[i]) / (ends[i] - starts[i])), 9))] += 1
    return host, comms, agg, place

def main():
    for run, sub in RUNS.items():
        bounds = step_bounds(run)
        files = sorted(glob.glob(f"{ROOT}/{sub}/nccl-inspector/*.log*"))
        with Pool(16) as p:
            res = p.map(one_rank, [(f, bounds) for f in files])
        # global rank of each file = its rank in the 32-rank communicator
        sizes = collections.Counter(cid for _, comms, _, _ in res for cid in comms)
        world = [cid for cid, c in sizes.items() if c == 32]
        ranks = [next(comms[c] for c in world if c in comms) for _, comms, _, _ in res]
        members = collections.defaultdict(dict)
        for (host, comms, _, _), g in zip(res, ranks):
            for cid in comms:
                members[cid][g] = host
        def shape(cid):
            m = sorted(members[cid]); hosts = {members[cid][r] for r in m}
            stride = m[1] - m[0] if len(m) > 1 else 0
            return f"n{len(m)}/nodes{len(hosts)}/stride{stride}"
        nsteps = len(bounds)
        per_comm = collections.defaultdict(lambda: collections.Counter())
        for (host, comms, agg, _), g in zip(res, ranks):
            for (cid, op, cls, k), v in agg.items():
                per_comm[(cid, op, cls)][k] += v
        info = {cid: {"shape": shape(cid), "members": {str(r): h for r, h in sorted(v.items())}, "ops": {}}
                for cid, v in members.items()}
        for (cid, op, cls), c in per_comm.items():
            k = len(members[cid])
            info[cid]["ops"][f"{op}/{cls}"] = {"ops": round(c["n"] / k / nsteps, 3),
                                               "GB": round(c["bytes"] / k / nsteps / 1e9, 5),
                                               "kernel_s": round(c["us"] / k / nsteps / 1e6, 5)}
        json.dump(info, open(DATA / f"comms_{run}.json", "w"), indent=1)
        tot = collections.defaultdict(lambda: collections.Counter()); who = collections.defaultdict(set)
        for (host, comms, agg, _), g in zip(res, ranks):
            for (cid, op, cls, k), v in agg.items():
                tot[(shape(cid), op, cls)][k] += v; who[(shape(cid), op, cls)].add(g)
        with open(DATA / f"nccl_{run}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["shape", "op", "size_class", "ranks", "ops_per_step", "GB_per_step", "kernel_s_per_step"])
            for key, c in sorted(tot.items(), key=lambda kv: -kv[1]["us"]):
                n = len(who[key])
                w.writerow([*key, n, round(c["n"] / n / nsteps, 2), round(c["bytes"] / n / nsteps / 1e9, 4),
                            round(c["us"] / n / nsteps / 1e6, 4)])
        rank0 = res[ranks.index(0)]
        pl = collections.defaultdict(lambda: [0] * 10)
        for (cid, op, cls, b), v in rank0[3].items():
            pl[(shape(cid), op, cls)][b] += v
        with open(DATA / f"placement_{run}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["shape", "op", "size_class", "ops"] + [f"tenth_{b}" for b in range(10)])
            for key, v in sorted(pl.items()):
                w.writerow([*key, sum(v)] + [round(x / sum(v), 3) for x in v])
        print(run, "files", len(files), "comms", len(members), "steps", nsteps)

if __name__ == "__main__":
    main()
