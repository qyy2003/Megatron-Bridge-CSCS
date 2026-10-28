"""NIC send rate through a step, phase-folded over steps 401-450 -> data/nic_phase.csv.

netmon samples each node's Slingshot NIC byte counters (4 NICs, hni_sts_tx_ok_octets) once per second.
Each 1-s interval's rate is assigned to the step phase of its midpoint; 50 steps of ~11-14 s land at
different phases, so 20 phase bins are well covered. Stage 0 = first 4 nodes of the job, stage 1 = last 4.
"""
import bisect, collections, csv, glob, json, os
from pathlib import Path

ROOT = "/capstor/scratch/cscs/yqin/mbcscs/runs"
RUNS = {"adam": ("train-q35a3b-gbs512to2560-sl4096/netlogs/job_3705060",
                 ["nid006575", "nid006586", "nid006591", "nid006620", "nid006634", "nid006643", "nid006644", "nid006646"]),
        "muon": ("train-q35a3b-gbs512to2560-sl4096-muon-max500/netlogs/job_3706874",
                 ["nid006045", "nid006110", "nid007148", "nid007157", "nid007169", "nid007225", "nid007260", "nid007289"]),
        "muon-zero1-gdp2": ("train-q35a3b-gbs512to2560-sl4096-muon-zero1-gdp2-max500/netlogs/job_3708048",
                            ["nid006082", "nid006103", "nid007156", "nid007180", "nid007183", "nid007186", "nid007190", "nid007220"])}
BINS = 20
out = Path(__file__).resolve().parents[1] / "data" / "nic_phase.csv"
rows = []
for run, (sub, nodes) in RUNS.items():
    for idx, node in enumerate(nodes):
        st = [json.loads(l) for l in open(f"{ROOT}/{sub}/steps/{node}.jsonl")]
        st = sorted((r["t0_ns"] / 1e9, r["t1_ns"] / 1e9) for r in st if 401 <= r["iter"] <= 450)
        t0s = [s[0] for s in st]
        prev = None
        acc = collections.defaultdict(list)
        for line in open(f"{ROOT}/{sub}/netmon/{node}.jsonl"):
            d = json.loads(line)
            if d.get("type") != "sample":
                continue
            t = d["t_ns"] / 1e9
            if t < t0s[0] - 2 or t > st[-1][1] + 2:
                prev = None if t > st[-1][1] + 2 else (t, d)
                continue
            tx = sum(c["hni_sts_tx_ok_octets"] for c in d["cxi"].values())
            if prev is not None:
                pt, ptx = prev
                mid = (t + pt) / 2
                i = bisect.bisect_right(t0s, mid) - 1
                if 0 <= i < len(st) and mid < st[i][1]:
                    phase = (mid - st[i][0]) / (st[i][1] - st[i][0])
                    acc[min(int(phase * BINS), BINS - 1)].append((tx - ptx) / (t - pt) / 1e9)
            prev = (t, tx)
        for b in range(BINS):
            v = acc.get(b, [])
            rows.append({"run": run, "node": node, "stage": idx // 4, "bin": b, "phase_mid": (b + 0.5) / BINS,
                         "tx_GBps": round(sum(v) / len(v), 3) if v else "", "samples": len(v)})
    print(run, "done")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
