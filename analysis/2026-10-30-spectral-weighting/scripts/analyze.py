"""0.8B spectral-weighting experiments: pull W&B histories, tabulate, plot.

    WANDB_API_KEY=... python scripts/analyze.py        # needs wandb + matplotlib

Runs: the seven 0.8B 500-step runs of WORKLOG §8 (W&B project qwen35-0.8b-cscs). Writes data/runs.csv
(per step: lm loss, mtp_1 loss, acceptance, step time), data/summary.csv and figures/*.png.
"""
import csv
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import wandb  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
(HERE / "data").mkdir(exist_ok=True)
(HERE / "figures").mkdir(exist_ok=True)
PROJECT = "yuyqin-eth-zurich/qwen35-0.8b-cscs"
BASE = "train-q35-0.8b-gbs512to2560-sl4096-"
RUNS = [  # label, run-name middle, colour
    ("A Muon (Newton-Schulz)", "muon-zero1-gdp2-max500", "#2a78d6"),
    ("F exact polar (p=0)", "muon-zero1-spec-polar-gdp2-max500", "#1baf7a"),
    ("E power p=0.5", "muon-zero1-spec-power0.5-gdp2-max500", "#eda100"),
    ("D top-1/4 only", "muon-zero1-spec-top0.25-gdp2-max500", "#e87ba4"),
    ("G Dion r=1/4", "dion-zero1-r0.25-gdp2-max500", "#eb6834"),
    ("C Dion r=1/4 x2", "dion-zero1-r0.25-x2-gdp2-max500", "#6250d6"),
    ("B Dion r=1", "dion-zero1-r1.0-gdp2-max500", "#008300"),
]
KEYS = ["_step", "lm loss", "mtp_1 loss", "mtp_1_acceptance_rate", "iteration-time", "optimizer-time"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

api = wandb.Api(timeout=120)
by_name = {r.name: r for r in api.runs(PROJECT, filters={"display_name": {"$regex": "max500$"}})}
hist, val, meta = {}, {}, {}
for label, mid, _ in RUNS:
    run = by_name.get(BASE + mid)
    if run is None:
        print("missing:", label)
        continue
    rows = {}
    for row in run.scan_history(keys=["_step", "lm loss", "mtp_1 loss"]):
        rows[row["_step"]] = dict(row)
    for extra in ("mtp_1_acceptance_rate", "iteration-time", "optimizer-time"):
        for row in run.scan_history(keys=["_step", extra]):
            rows.setdefault(row["_step"], {})[extra] = row[extra]
    hist[label] = rows
    val[label] = {row["_step"]: row["lm loss validation"] for row in run.scan_history(keys=["_step", "lm loss validation"])}
    meta[label] = run.state
with open(HERE / "data/runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "step", "lm_loss", "mtp_1_loss", "mtp_1_acceptance_rate", "iteration_time", "optimizer_time"])
    for label, rows in hist.items():
        for s in sorted(rows):
            r = rows[s]
            w.writerow([label, s, r.get("lm loss"), r.get("mtp_1 loss"), r.get("mtp_1_acceptance_rate"),
                        r.get("iteration-time"), r.get("optimizer-time")])


def mean(label, key, s, w=10):
    v = [hist[label][i][key] for i in range(s - w + 1, s + 1) if i in hist[label] and hist[label][i].get(key) is not None]
    return sum(v) / len(v) if v else float("nan")


summary = []
for label, _, _ in RUNS:
    if label not in hist:
        continue
    v = val[label]
    times = [hist[label][i]["iteration-time"] for i in hist[label] if i >= 20 and hist[label][i].get("iteration-time")]
    opt = [hist[label][i]["optimizer-time"] for i in hist[label] if i >= 20 and hist[label][i].get("optimizer-time")]
    summary.append({"run": label, "state": meta[label], "val250": v.get(250), "val500": v.get(500),
                    **{f"lm{s}": mean(label, "lm loss", s) for s in (25, 50, 100, 200, 300, 400, 500)},
                    "mtp500": mean(label, "mtp_1 loss", 500), "acc500": mean(label, "mtp_1_acceptance_rate", 500),
                    "step_s": st.median(times) if times else None, "opt_s": st.median(opt) if opt else None})
with open(HERE / "data/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
for r in summary:
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ref = "A Muon (Newton-Schulz)"
for label, _, c in RUNS:
    if label not in hist:
        continue
    steps = sorted(s for s in hist[label] if s >= 10)
    ls = (0, (4, 2)) if label.startswith(("B", "C", "G")) else "-"
    axes[0].plot(steps, [mean(label, "lm loss", s) for s in steps], color=c, lw=1.5, ls=ls, label=label)
    if ref in hist:
        common = [s for s in steps if s in hist[ref]]
        axes[1].plot(common, [mean(label, "lm loss", s) - mean(ref, "lm loss", s) for s in common], color=c, lw=1.5, ls=ls)
        axes[2].plot(common, [mean(label, "mtp_1 loss", s) - mean(ref, "mtp_1 loss", s) for s in common], color=c, lw=1.5, ls=ls)
axes[0].set_yscale("log"); axes[0].set_title("LM loss (10-step mean)", loc="left"); axes[0].legend(fontsize=8.5)
axes[0].set_yticks([3, 4, 5, 6, 8, 10, 12]); axes[0].set_yticklabels(["3", "4", "5", "6", "8", "10", "12"]); axes[0].minorticks_off()
for ax, t in ((axes[1], "LM loss − Muon (A); above 0 = worse"), (axes[2], "MTP-1 loss − Muon (A)")):
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title(t, loc="left"); ax.set_xlabel("step")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_yticks([-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1]); ax.set_yticklabels(["-1", "-0.5", "-0.2", "-0.1", "0", "0.1", "0.2", "0.5", "1"])
    ax.set_ylim(-1.2, 1.2); ax.minorticks_off()
axes[0].set_xlabel("step")
fig.savefig(HERE / "figures/loss_vs_muon.png", dpi=150, bbox_inches="tight")
fig, ax = plt.subplots(figsize=(9, 3.8))
labels = [r["run"] for r in summary if r["val500"]]
vals = [r["val500"] for r in summary if r["val500"]]
cols = [c for label, _, c in RUNS if label in labels]
bars = ax.barh(range(len(labels))[::-1], vals, color=cols, height=0.55, edgecolor=SURFACE)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.3f} (PPL {math.exp(v):.2f})", (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(labels))[::-1]); ax.set_yticklabels(labels)
lo = min(vals) - 0.15
ax.set_xlim(lo, max(vals) + 0.12); ax.grid(axis="y", visible=False)
ax.set_title("Validation LM loss at step 500 (lower is better)", loc="left")
fig.savefig(HERE / "figures/val500.png", dpi=150, bbox_inches="tight")
print("figures written")
