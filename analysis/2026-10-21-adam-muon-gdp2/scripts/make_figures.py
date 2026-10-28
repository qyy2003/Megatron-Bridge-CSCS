"""All figures of the report from data/*.csv|json (run the extract_*.py scripts first).

    python scripts/make_figures.py      # needs matplotlib + numpy
"""
import collections
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)

# Reference categorical palette (light), fixed order; validated with the dataviz validator (adjacent pairs).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
RUNS = ["adam", "muon", "muon-zero1-gdp2"]
LABEL = {"adam": "Adam (baseline)", "muon": "Muon", "muon-zero1-gdp2": "Muon ZeRO-1 + GLOBAL_DP=2"}
COLOR = dict(zip(RUNS, SLOT))
NODES = {"adam": ["nid006575", "nid006586", "nid006591", "nid006620", "nid006634", "nid006643", "nid006644", "nid006646"],
         "muon": ["nid006045", "nid006110", "nid007148", "nid007157", "nid007169", "nid007225", "nid007260", "nid007289"],
         "muon-zero1-gdp2": ["nid006082", "nid006103", "nid007156", "nid007180", "nid007183", "nid007186", "nid007190", "nid007220"]}
# Slurm topology (scontrol show topology): Slingshot dragonfly group of each node
DF_GROUP = {"nid006575": 28, "nid006586": 28, "nid006591": 28, "nid006620": 28, "nid006634": 28, "nid006643": 28,
            "nid006644": 28, "nid006646": 28, "nid006045": 23, "nid006110": 23, "nid007148": 33, "nid007157": 33,
            "nid007169": 33, "nid007225": 33, "nid007260": 34, "nid007289": 34, "nid006082": 23, "nid006103": 23,
            "nid007156": 33, "nid007180": 33, "nid007183": 33, "nid007186": 33, "nid007190": 33, "nid007220": 33}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load_steps():
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(DATA / "steps.csv")):
        d[r["run"]][int(r["iter"])] = r
    return d


def series(d, run, key, lo=1, hi=10**9):
    its = [i for i in sorted(d[run]) if lo <= i <= hi and d[run][i][key]]
    return np.array(its), np.array([float(d[run][i][key]) for i in its])


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", color=INK2, fontsize=8.5, va="center")


# ---------------------------------------------------------------- 1. loss curves
def fig_loss(d):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in [(axes[0], "lm_loss", "LM loss (next token)"), (axes[1], "mtp_loss", "MTP-1 loss (token t+2)")]:
        for run in RUNS:
            x, y = series(d, run, key, hi=500)
            ax.plot(x, y, color=COLOR[run], label=LABEL[run], lw=1.4 if run != "muon-zero1-gdp2" else 1.0,
                    ls="-" if run != "muon-zero1-gdp2" else (0, (4, 2)))
        ax.set_yscale("log"); ax.set_ylim(2.35, 13.8)
        ax.set_yticks([2.5, 3, 4, 5, 6, 8, 10, 12.8]); ax.set_yticklabels(["2.5", "3", "4", "5", "6", "8", "10", "12.8"])
        ax.minorticks_off()
        ax.set_xlabel("step (GBS 512 × 4,096 tokens = 2.1M tokens)"); ax.set_title(title, loc="left")
        ax.set_xlim(0, 505)
    for run, val in [("adam", 2.820), ("muon", 2.536)]:
        axes[0].plot([500], [val], marker="D", ms=6, color=COLOR[run], mec=SURFACE, mew=1.5, ls="none")
        end_label(axes[0], 500, val, f"val {val:.3f}", dy=8 if run == "adam" else -8)
    axes[0].legend(loc="upper right")
    # 3rd panel: Muon - Adam (negative = Muon better)
    ax = axes[2]
    for key, ls, name in [("lm_loss", "-", "LM loss"), ("mtp_loss", (0, (4, 2)), "MTP-1 loss")]:
        xa, ya = series(d, "adam", key, hi=500); xm, ym = series(d, "muon", key, hi=500)
        common = sorted(set(xa) & set(xm))
        a = dict(zip(xa, ya)); m = dict(zip(xm, ym))
        diff = np.array([m[i] - a[i] for i in common])
        ax.plot(common, diff, color=COLOR["muon"], ls=ls, label=f"Muon − Adam, {name}")
        cross = next(i for i, v in zip(common, diff) if i > 20 and v < 0)
        ax.axvline(cross, color=MUTED, lw=0.8)
        ax.annotate(f"{name}: Muon ahead\nfrom step {cross}", (cross, 2.1 if key == "lm_loss" else 1.2),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Muon − Adam (below 0 = Muon better)", loc="left"); ax.set_xlabel("step"); ax.set_xlim(0, 505)
    ax.legend(loc="upper right")
    save(fig, "fig1_loss.png")


# ---------------------------------------------------------------- 2. early phase
def fig_early(d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    ax = axes[0]
    for run in ["adam", "muon"]:
        x, y = series(d, run, "grad_norm", hi=200)
        ax.plot(x, y, color=COLOR[run], label=LABEL[run])
    x, lr = series(d, "adam", "lr", hi=200)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("Gradient norm (before clipping at 1.0)", loc="left")
    ax.legend(loc="lower left")
    ax2 = ax.inset_axes([0.50, 0.50, 0.30, 0.25])
    ax2.plot(x, lr * 1e4, color=INK2, lw=1.0); ax2.set_title("LR ×1e-4 (same for both)", fontsize=8, loc="left",
                                                             fontweight="normal", color=INK2)
    ax2.tick_params(labelsize=7); ax2.grid(False)
    ax = axes[1]
    rows = [r for r in csv.DictReader(open(DATA / "update_alignment.csv")) if r["rank"] == "4"]
    share = [float(r["signal_share"]) for r in rows]
    for key, run, name in [("adam_sign_cos", "adam", "Adam, first steps: sign(g)"),
                           ("muon_ns5_cos", "muon", "Muon: Newton–Schulz(g), spectral scale")]:
        y = [float(r[key]) for r in rows]
        ax.plot(share, y, color=COLOR[run], marker="o", ms=5, mec=SURFACE, mew=1.5, label=name)
    ax.set_xlabel("share of ‖g‖² in a rank-4 signal (rest: isotropic noise)")
    ax.set_ylabel("cos(update, gradient)"); ax.set_ylim(0, 1)
    ax.set_title("Toy 1024×1024 matrix: alignment of the step with g", loc="left")
    ax.invert_xaxis(); ax.legend(loc="lower right")
    ax.annotate("early training:\ngradient ≈ low rank", (0.98, 0.17), xytext=(0.93, 0.03), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    save(fig, "fig2_early_phase.png")


# ---------------------------------------------------------------- network data helpers
def purpose(run, shape, op, cls):
    n, nodes, stride = (int(p.lstrip("nodestrid")) for p in shape.split("/"))
    if cls == "small" or n == 32:
        return "Small reductions"
    if op in ("Send", "Recv"):
        return "Pipeline P2P"
    if n == 2 and stride == 16:
        return "Embedding sync"
    if nodes == 1:
        return "EP routing metadata"
    if run == "muon-zero1-gdp2" and stride == 8:
        return "Grad sync across instances"
    if op in ("ReduceScatter", "AllReduce"):
        return "Grad sync in DP group"
    if op == "AllGather":
        return "Param all-gather"
    return "Other"


PURPOSES = ["Grad sync in DP group", "Grad sync across instances", "Param all-gather", "Pipeline P2P",
            "Embedding sync", "Small reductions", "EP routing metadata"]
PCOLOR = dict(zip(PURPOSES, SLOT))


def load_nccl(run):
    return list(csv.DictReader(open(DATA / f"nccl_{run}.csv")))


def median_step(d, run):
    x, y = series(d, run, "step_ms", lo=401, hi=450)
    return float(np.median(y)) / 1000


# ---------------------------------------------------------------- 3. step time
def fig_step_time(d):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.55})
    ax = axes[0]
    for run in RUNS:
        x, y = series(d, run, "step_ms", lo=2, hi=500)
        y = np.sort(y / 1000)
        ax.step(y, np.arange(1, len(y) + 1) / len(y), where="post", color=COLOR[run], label=LABEL[run])
        med = np.median(y)
        ax.plot([med], [0.5], marker="o", ms=6, color=COLOR[run], mec=SURFACE, mew=1.5)
    ax.set_xscale("log"); ax.set_xticks([10, 15, 20, 50, 100, 300, 1000, 2000])
    ax.set_xticklabels(["10", "15", "20", "50", "100", "300", "1000", "2000"]); ax.minorticks_off()
    ax.set_xlabel("step time [s] (steps 2–500, log scale)"); ax.set_ylabel("share of steps ≤ x")
    ax.set_title("Step-time distribution (dots: medians)", loc="left"); ax.legend(loc="lower right")
    ax = axes[1]
    yb = np.arange(len(RUNS))[::-1]
    for i, run in enumerate(RUNS):
        tot = collections.Counter()
        for r in load_nccl(run):
            tot[purpose(run, r["shape"], r["op"], r["size_class"])] += float(r["kernel_s_per_step"])
        left = 0.0
        for p in PURPOSES:
            v = tot.get(p, 0.0)
            if v <= 0:
                continue
            ax.barh(yb[i], v, left=left, height=0.5, color=PCOLOR[p], edgecolor=SURFACE, linewidth=2,
                    label=p if i == 2 or (p != "Grad sync across instances" and i == 0) else None)
            left += v
        ax.annotate(f"{left:.2f} s in collectives  |  median step {median_step(d, run):.1f} s",
                    (left, yb[i]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(yb); ax.set_yticklabels([LABEL[r].replace(" + ", "\n+ ") for r in RUNS]); ax.set_xlim(0, 5.4)
    ax.set_xlabel("GPU kernel time inside NCCL per GPU per step [s] (steps 401–450; overlaps compute)")
    ax.set_title("Where communication time goes", loc="left"); ax.grid(axis="y", visible=False)
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(p) for p in PURPOSES if p in lab]
    ax.legend([h[k] for k in order], [lab[k] for k in order], loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=4, fontsize=8.5)
    save(fig, "fig3_step_time.png")


# ---------------------------------------------------------------- 4. NIC traffic through a step
def fig_nic():
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(DATA / "nic_phase.csv")):
        if r["tx_GBps"]:
            acc[(r["run"], int(r["stage"]), float(r["phase_mid"]))].append(float(r["tx_GBps"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for stage, ax in enumerate(axes):
        for run in RUNS:
            ph = sorted({k[2] for k in acc if k[0] == run and k[1] == stage})
            y = [np.mean(acc[(run, stage, p)]) for p in ph]
            ax.plot(ph, y, color=COLOR[run], marker="o", ms=4, mec=SURFACE, mew=1, label=LABEL[run])
        ax.axhline(100, color=MUTED, lw=0.8)
        ax.annotate("node NIC peak 4 × 25 GB/s", (0.02, 100), xytext=(0, -10), textcoords="offset points",
                    fontsize=8, color=INK2)
        ax.set_xlabel("position in the step (0 = start, 1 = end)")
        ax.set_title(f"Stage {stage} nodes ({'1–4' if stage == 0 else '5–8'}): NIC send rate per node", loc="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 108)
    axes[0].set_ylabel("GB/s per node (Slingshot, 4 NICs)")
    axes[0].annotate("pipeline activations only\n(~0.2 GB/s)", (0.5, 1), xytext=(0.32, 25), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].annotate("gradient sync burst", (0.87, 60), xytext=(0.45, 70), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].legend(loc="center left", bbox_to_anchor=(0.03, 0.55))
    save(fig, "fig4_nic_through_step.png")


# ---------------------------------------------------------------- traffic model (node x node)
SEND_FACTOR = {"AllReduce": lambda k: 2 * (k - 1) / k, "ReduceScatter": lambda k: k - 1,
               "AllGather": lambda k: k - 1, "Send": lambda k: 1.0, "Recv": lambda k: 0.0}


def traffic(run, only=None):
    """GB per step sent from node i to node j (inter-node only), per purpose.

    Ring model (NCCL's default for these sizes): the members of a communicator form one ring that visits each
    node's GPUs consecutively, nodes in rank order, so each node sends one ring hop to the next node. Every hop
    carries what one rank sends: NCCL count x (2(k-1)/k for AllReduce, k-1 for ReduceScatter/AllGather, whose
    count is the per-rank part). Send = its bytes to the peer. Checked against the NIC counters (nic_vs_model.json).
    """
    comms = json.load(open(DATA / f"comms_{run}.json"))
    nodes = NODES[run]
    mats = collections.defaultdict(lambda: np.zeros((8, 8)))
    for cid, c in comms.items():
        mem = {int(k): v for k, v in c["members"].items()}
        k = len(mem)
        ring = []
        for r in sorted(mem):
            if mem[r] not in ring:
                ring.append(mem[r])
        if k < 2 or len(ring) < 2:
            continue
        for key, st in c["ops"].items():
            op, cls = key.split("/")
            if op not in SEND_FACTOR:
                continue
            p = purpose(run, c["shape"], op, cls)
            if only and p not in only:
                continue
            if op == "Send":  # 2-rank P2P: rank -> the other rank
                src, dst = sorted(mem)
                vol = st["GB"]  # mean over the 2 members; both send in 1F1B
                mats[p][nodes.index(mem[src]), nodes.index(mem[dst])] += vol
                mats[p][nodes.index(mem[dst]), nodes.index(mem[src])] += vol
                continue
            hop = st["GB"] * SEND_FACTOR[op](k)
            for a, b in zip(ring, ring[1:] + ring[:1]):
                mats[p][nodes.index(a), nodes.index(b)] += hop
    total = sum(mats.values()) if mats else np.zeros((8, 8))
    return total, dict(mats)


def fig_matrix():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    mats = {run: traffic(run)[0] for run in RUNS}
    vmax = max(m.max() for m in mats.values())
    for ax, run in zip(axes, RUNS):
        m = mats[run]
        im = ax.imshow(np.where(m > 0, m, np.nan), cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        lab = [f"{i + 1}·g{DF_GROUP[n]}" for i, n in enumerate(NODES[run])]
        ax.set_xticklabels(lab, rotation=90, fontsize=8); ax.set_yticklabels(lab, fontsize=8)
        ax.grid(False); ax.set_xlabel("receiving node · dragonfly group"); ax.set_ylabel("sending node")
        for i in range(8):
            for j in range(8):
                if m[i, j] >= 0.5:
                    ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color=SURFACE if m[i, j] > 0.55 * vmax else INK)
        ax.set_title(f"{LABEL[run]}\n{m.sum() / 8:.0f} GB out per node per step", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.03); cb.set_label("GB per step (inter-node NCCL traffic)")
    save(fig, "fig6_traffic_matrix.png")
    return mats


# ---------------------------------------------------------------- 5. who talks to whom (gradient sync)
def fig_topology():
    """Who talks to whom in the end-of-step gradient sync: nodes in rank order (stage 0 top), dragonfly group as box
    shade, edge width/label = GB per step between the two nodes (both directions, ring model)."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 15.5))
    W, H, DX = 2.0, 1.25, 3.0          # node box size, horizontal pitch
    Y0, Y1 = 3.6, 0.0                  # stage 0 / stage 1 rows
    for ax, run in zip(axes, RUNS):
        _, per = traffic(run)
        pos = {i: (i % 4 * DX, Y0 if i < 4 else Y1) for i in range(8)}
        ax.set_xlim(-1.0, 3 * DX + W + 0.4); ax.set_ylim(-1.9, Y0 + H + 1.6); ax.axis("off")
        groups = sorted({DF_GROUP[n] for n in NODES[run]})
        shade = dict(zip(groups, ["#efeeea", "#d9d7cf", "#c2bfb5"]))
        for mat, color in [(per.get("Grad sync in DP group", np.zeros((8, 8))), SLOT[0]),
                           (per.get("Grad sync across instances", np.zeros((8, 8))), SLOT[1])]:
            sym = mat + mat.T
            for i in range(8):
                for j in range(i + 1, 8):
                    gb = sym[i, j]
                    if gb < 0.05:
                        continue
                    (x1, y1), (x2, y2) = pos[i], pos[j]
                    top = y1 == Y0
                    dist = abs(j - i)
                    if dist == 1:   # neighbours: straight line between the boxes
                        ya = y1 + H / 2
                        ax.plot([x1 + W, x2], [ya, ya], color=color, lw=1 + 6 * gb / 60, solid_capstyle="butt", zorder=2)
                        ax.text((x1 + W + x2) / 2, ya + (0.22 if top else -0.22), f"{gb:.0f}", ha="center",
                                va="center", fontsize=8, color=INK2)
                        continue
                    # farther pairs: arc above stage 0 / below stage 1
                    ya = y1 + H if top else y1
                    h = (0.55 if dist == 2 else 1.05) * (1 if top else -1)
                    xs = np.linspace(x1 + W / 2 + 0.25, x2 + W / 2 - 0.25, 50)
                    t = (xs - xs[0]) / (xs[-1] - xs[0])
                    ax.plot(xs, ya + h * 4 * t * (1 - t), color=color, lw=1 + 6 * gb / 60, zorder=2)
                    shift = (i % 2) * 0.35 * (1 if top else -1) if dist == 2 else 0
                    ax.text((xs[0] + xs[-1]) / 2, ya + h + (0.18 if top else -0.18) + shift, f"{gb:.0f}",
                            ha="center", va="center", fontsize=8, color=INK2)
        for i in range(4):  # pipeline links
            x, _ = pos[i]
            ax.plot([x + W / 2] * 2, [Y1 + H, Y0], color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        for i, node in enumerate(NODES[run]):
            x, y = pos[i]
            g = DF_GROUP[node]
            ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08", fc=shade[g],
                                        ec=INK2, lw=0.8, zorder=3))
            inst = f" · instance {'A' if i in (0, 1, 4, 5) else 'B'}" if run == "muon-zero1-gdp2" else ""
            ax.text(x + W / 2, y + H - 0.2, f"node {i + 1}{inst}", ha="center", fontsize=8.5, color=INK,
                    fontweight="bold", zorder=4)
            ax.text(x + W / 2, y + H - 0.45, f"{node} · group {g}", ha="center", fontsize=7.5, color=INK2, zorder=4)
            for gpu in range(4):
                gx = x + 0.14 + gpu * 0.44
                ax.add_patch(Rectangle((gx, y + 0.12), 0.36, 0.45, fc=SURFACE, ec=MUTED, lw=0.6, zorder=4))
                ax.text(gx + 0.18, y + 0.345, str(4 * i + gpu), ha="center", va="center", fontsize=7, color=INK2,
                        zorder=5)
        ax.text(-0.75, Y0 + H / 2, "stage 0", rotation=90, va="center", fontsize=9, color=INK2)
        ax.text(-0.75, Y1 + H / 2, "stage 1", rotation=90, va="center", fontsize=9, color=INK2)
        gtxt = ", ".join(f"group {g} ({sum(DF_GROUP[n] == g for n in NODES[run])} nodes)" for g in groups)
        ax.set_title(f"{LABEL[run]}    —    dragonfly groups (box shade): {gtxt}", loc="left", fontsize=10.5)
    handles = [Line2D([], [], color=SLOT[0], lw=3), Line2D([], [], color=SLOT[1], lw=3),
               Line2D([], [], color=MUTED, lw=1, ls=(0, (2, 2)))]
    fig.legend(handles, ["gradient sync inside the DP group (GLOBAL_DP=1) / inside an instance (=2)",
                         "gradient all-reduce across instances (\"datacenters\")",
                         "pipeline stage 0 ↔ 1 (activations, all step long)"],
               loc="lower center", ncol=1, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("End-of-step gradient sync: GB per step between two nodes (both directions; ring model from NCCL logs). "
                 "GPU boxes: global rank.", fontsize=9.5, color=INK2, x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(hspace=0.12, bottom=0.07, top=0.97)
    save(fig, "fig5_who_talks_to_whom.png")


def check_against_nic(mats):
    """Compare the modelled per-node send volume with the NIC counters (steps 401-450)."""
    root = "/capstor/scratch/cscs/yqin/mbcscs/runs"
    sub = {"adam": "train-q35a3b-gbs512to2560-sl4096/netlogs/job_3705060",
           "muon": "train-q35a3b-gbs512to2560-sl4096-muon-max500/netlogs/job_3706874",
           "muon-zero1-gdp2": "train-q35a3b-gbs512to2560-sl4096-muon-zero1-gdp2-max500/netlogs/job_3708048"}
    out = {}
    for run in RUNS:
        tx = []
        for f in glob.glob(f"{root}/{sub[run]}/steps/*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if 401 <= r["iter"] <= 450:
                    tx.append(sum(v[0] for v in r["nic_tx_rx_bytes"].values()) / 1e9)
        out[run] = (float(np.mean(tx)), float(mats[run].sum() / 8))
        print(f"{run}: NIC counters {out[run][0]:.1f} GB/node/step, model {out[run][1]:.1f}")
    json.dump(out, open(DATA / "nic_vs_model.json", "w"), indent=1)


if __name__ == "__main__":
    d = load_steps()
    fig_loss(d)
    fig_early(d)
    fig_step_time(d)
    fig_nic()
    mats = fig_matrix()
    fig_topology()
    check_against_nic(mats)
