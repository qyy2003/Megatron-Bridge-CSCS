"""9B Dion vs Muon figure (job logs + data/comms_9b-*.json from extract_nccl.py) -> figures/.

    python scripts/make_figures.py      # matplotlib
"""
import csv, json, math, re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SLOT = ["#2a78d6", "#eb6834"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
RUNS = {"Muon ZeRO-1 + GDP2": ("3789784", "9b-muon-zero1-gdp2"), "Dion r=1/4 + GDP2": ("3789785", "9b-dion-zero1-r0.25-gdp2")}
steps, val = {}, {}
for name, (job, _) in RUNS.items():
    log = next((REPO / "logs").glob(f"*-{job}.log"))
    d = steps.setdefault(name, {})
    for line in open(log, errors="replace"):
        m = re.search(r"^ *15: .*iteration +(\d+)/.*per iteration \(ms\): ([\d.]+).*?lm loss: ([\dE.+-]+).*?mtp_1 loss: ([\dE.+-]+)", line)
        if m:
            d[int(m.group(1))] = (float(m.group(2)) / 1000, float(m.group(3)), float(m.group(4)))
        m = re.search(r"^ *15: +validation loss at iteration (\d+) \| lm loss value: ([\dE.+-]+)", line)
        if m:
            val[name] = (int(m.group(1)), float(m.group(2)))
with open(HERE / "data/steps_9b_dion_vs_muon.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "iter", "step_s", "lm_loss", "lm_ppl", "mtp_loss"])
    for name, d in steps.items():
        for it in sorted(d):
            w.writerow([name, it, d[it][0], d[it][1], round(math.exp(min(20, d[it][1])), 4), d[it][2]])


def smooth(d, k, w=10):
    its = sorted(d)
    return its, [sum(d[j][k] for j in its[max(0, i - w + 1): i + 1]) / len(its[max(0, i - w + 1): i + 1]) for i in range(len(its))]


fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
for c, name in zip(SLOT, RUNS):
    its, y = smooth(steps[name], 1)
    ax.plot(its, y, color=c, lw=1.6, label=name)
    it, v = val[name]
    ax.plot([it], [v], marker="D", ms=6, color=c, mec=SURFACE, mew=1.5, ls="none")
ax.text(0.30, 0.52, "validation at step 500 (diamonds):\n" + "\n".join(
    f"{n.split(' ')[0]}: {val[n][1]:.3f}  (PPL {math.exp(val[n][1]):.2f})" for n in RUNS),
    transform=ax.transAxes, fontsize=8.5, color=INK2)
ax.set_yscale("log"); ax.set_yticks([2.5, 3, 4, 6, 8, 13]); ax.set_yticklabels(["2.5", "3", "4", "6", "8", "13"])
ax.minorticks_off(); ax.set_xlabel("step (GBS 512 × 4096 tokens)"); ax.set_title("9B: LM loss (10-step mean)", loc="left")
ax.legend(loc="upper right")
ax = axes[1]
m_its, m = smooth(steps["Muon ZeRO-1 + GDP2"], 1)
d_its, dd = smooth(steps["Dion r=1/4 + GDP2"], 1)
common = sorted(set(m_its) & set(d_its))
mm, dm = dict(zip(m_its, m)), dict(zip(d_its, dd))
ax.plot(common, [dm[i] - mm[i] for i in common], color=SLOT[1], lw=1.6)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("step"); ax.set_title("Dion − Muon, LM loss (above 0 = Dion behind)", loc="left")


def per_stage(path):
    out = {0: 0.0, 1: 0.0}
    for v in json.load(open(path)).values():
        if v["shape"].startswith("n2/nodes2/stride4"):
            stage = 0 if min(map(int, v["members"])) < 8 else 1
            out[stage] += sum(o["GB"] for k, o in v["ops"].items() if k.startswith("AllReduce")) / 4
    return out


ax = axes[2]
vals = {name: per_stage(HERE / f"data/comms_{tag}.json") for name, (_, tag) in RUNS.items()}
names = list(RUNS)
for k, stage in enumerate((0, 1)):
    xs = [i + (k - 0.5) * 0.36 for i in range(len(names))]
    bars = ax.bar(xs, [vals[n][stage] for n in names], width=0.34, color=("#1baf7a", "#6250d6")[k], edgecolor=SURFACE, lw=2,
                  label=f"stage {stage} GPUs")
    for b, n in zip(bars, names):
        ax.annotate(f"{vals[n][stage]:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.grid(axis="x", visible=False)
ax.set_ylabel("GB per GPU per step"); ax.set_title("Cross-datacenter all-reduce (steps 401–450)", loc="left")
ax.legend(loc="upper right")
fig.savefig(HERE / "figures/9b_dion_vs_muon.png", dpi=150, bbox_inches="tight")
print("wrote figures/9b_dion_vs_muon.png")
