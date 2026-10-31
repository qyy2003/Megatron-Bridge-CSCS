"""Momentum spectra per module (data/spectra/<run>/spectra.csv from tools/ckpt_spectra.py) -> figures/spectra.png."""
import collections, csv, re, statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
RUNS = [("9B Muon momentum", "9b-muon", "#2a78d6"), ("0.8B Muon momentum", "0.8b-muon", "#1baf7a"),
        ("0.8B Dion r=1/4 backlog", "0.8b-dion-r0.25", "#eb6834")]
MODS = ["linear_qkv", "linear_proj", "in_proj.value", "in_proj.z", "out_proj", "linear_fc1", "linear_fc2", "MTP linear_fc1", "MTP linear_fc2"]


def load(tag):
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(HERE / f"data/spectra/{tag}/spectra.csv")):
        sub = re.search(r"weight\.(\w+)$", r["key"])
        g[("MTP " if r["mtp"] == "True" else "") + r["module"] + (f".{sub.group(1)}" if sub else "")].append(r)
    return g


data = {tag: load(tag) for _, tag, _ in RUNS}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
for ax, key, title in ((axes[0], "energy_top1/4", "Share of ||M||² in the top 1/4 of singular directions"),
                       (axes[1], "muon_alignment", "cos(U Vᵀ, M): alignment of a fully whitened step with M")):
    for k, (label, tag, c) in enumerate(RUNS):
        xs = [i + (k - 1) * 0.27 for i in range(len(MODS))]
        ys = [st.median(float(r[key]) for r in data[tag][m]) if data[tag].get(m) else 0 for m in MODS]
        ax.bar(xs, ys, width=0.25, color=c, edgecolor=SURFACE, lw=1.5, label=label)
    ax.set_xticks(range(len(MODS))); ax.set_xticklabels(MODS, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
axes[0].legend(loc="lower left", fontsize=8.5)
fig.savefig(HERE / "figures/spectra.png", dpi=150, bbox_inches="tight")
print("wrote figures/spectra.png")
