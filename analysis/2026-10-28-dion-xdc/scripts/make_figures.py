"""Figures of the Dion check (job logs + data/comms_*.json from extract_nccl.py) -> figures/.

    python scripts/make_figures.py      # matplotlib
"""
import csv, json, re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

# first 40 steps: the three 500-step runs (train mode) and the two Dion checks (same data, LR warmup 500)
steps = {}
for r in csv.DictReader(open(REPO / "analysis/2026-10-21-adam-muon-gdp2/data/steps.csv")):
    if int(r["iter"]) <= 40:
        steps.setdefault(r["run"], {})[int(r["iter"])] = (float(r["lm_loss"]), float(r["step_ms"]) / 1000)
for name, job in (("dion-r1/4-gdp2", "3786160"), ("dion-r1/16-gdp2", "3786472")):
    log = next((REPO / "logs").glob(f"*-{job}.log"))
    for line in open(log, errors="replace"):
        m = re.search(r"^ *31: .*iteration +(\d+)/.*per iteration \(ms\): ([\d.]+).*lm loss: ([\dE.+-]+)", line)
        if m:
            steps.setdefault(name, {})[int(m.group(1))] = (float(m.group(3)), float(m.group(2)) / 1000)
with open(HERE / "data/steps_first40.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "iter", "lm_loss", "step_s"])
    for run, d in steps.items():
        for it in sorted(d):
            w.writerow([run, it, *d[it]])
LABEL = {"adam": "Adam", "muon": "Muon", "muon-zero1-gdp2": "Muon ZeRO-1 + GDP2",
         "dion-r1/4-gdp2": "Dion r=1/4 + GDP2", "dion-r1/16-gdp2": "Dion r=1/16 + GDP2"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
for c, run in zip(SLOT, LABEL):
    d = steps[run]; its = sorted(d)
    ax.plot(its, [d[i][0] for i in its], color=c, lw=1.6, label=LABEL[run],
            ls=(0, (4, 2)) if run == "muon-zero1-gdp2" else "-")  # lies on top of Muon
ax.set_xlabel("step (GBS 512, LR warmup 500 steps)"); ax.set_title("LM loss, first 40 steps", loc="left")
ax.legend(loc="lower left")

# cross-instance all-reduce per GPU per step, by pipeline stage
def per_stage(path):
    out = {0: 0.0, 1: 0.0}
    for v in json.load(open(path)).values():
        if v["shape"].startswith("n2/nodes2/stride8"):
            stage = 0 if min(map(int, v["members"])) < 16 else 1
            out[stage] += sum(o["GB"] for k, o in v["ops"].items() if k.startswith("AllReduce")) / 8
    return out
src = {"Muon ZeRO-1 + GDP2": REPO / "analysis/2026-10-21-adam-muon-gdp2/data/comms_muon-zero1-gdp2.json",
       "Dion r=1/4 + GDP2": HERE / "data/comms_dion-r0.25-gdp2.json",
       "Dion r=1/16 + GDP2": HERE / "data/comms_dion-r0.0625-gdp2.json"}
ax = axes[1]
names = list(src); vals = {n: per_stage(p) for n, p in src.items()}
for k, stage in enumerate((0, 1)):
    xs = [i + (k - 0.5) * 0.36 for i in range(len(names))]
    bars = ax.bar(xs, [vals[n][stage] for n in names], width=0.34, color=SLOT[k], edgecolor=SURFACE, lw=2,
                  label=f"stage {stage} GPUs")
    for b, n in zip(bars, names):
        ax.annotate(f"{vals[n][stage]:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
ax.set_ylabel("GB per GPU per step"); ax.grid(axis="x", visible=False)
ax.set_title("Cross-datacenter all-reduce (NCCL inspector)", loc="left"); ax.legend(loc="upper right")
fig.savefig(HERE / "figures/dion_loss_and_bytes.png", dpi=150, bbox_inches="tight")
print("wrote figures/dion_loss_and_bytes.png")
