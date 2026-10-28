"""Toy: how much of a unit-RMS update points along the gradient, for Adam's first steps (sign(g)) and
Muon (5 quintic Newton-Schulz steps, spectral scale sqrt(max(m, n)) as in emerging_optimizers),
when the gradient is a low-rank signal plus an isotropic bulk. -> data/update_alignment.csv

alignment = <g, U> / (||g||_F * ||U||_F) (cosine); first-order loss decrease per step = lr * <g, U>.
"""
import csv
from pathlib import Path
import numpy as np

def ns5(g, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315  # quintic coefficients (Muon / emerging_optimizers default)
    x = g / (np.linalg.norm(g) + 1e-7)
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return x

rng = np.random.default_rng(0)
m = n = 1024
rows = []
for signal_share in [0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]:   # share of ||g||_F^2 in the rank-4 signal
    for rank in [1, 4]:
        u, _ = np.linalg.qr(rng.standard_normal((m, rank))); v, _ = np.linalg.qr(rng.standard_normal((n, rank)))
        sig = u @ np.diag(np.linspace(2, 1, rank)) @ v.T
        noise = rng.standard_normal((m, n))
        sig *= np.sqrt(signal_share) / np.linalg.norm(sig); noise *= np.sqrt(1 - signal_share) / np.linalg.norm(noise)
        g = sig + noise
        upd = {"adam_sign": np.sign(g), "muon_ns5": ns5(g) * np.sqrt(max(m, n))}
        row = {"signal_share": signal_share, "rank": rank}
        for k, U in upd.items():
            row[k + "_rms"] = float(np.sqrt((U ** 2).mean()))
            row[k + "_cos"] = float((g * U).sum() / np.linalg.norm(g) / np.linalg.norm(U))
            row[k + "_cos_signal"] = float((sig * U).sum() / (np.linalg.norm(sig) + 1e-30) / np.linalg.norm(U)) if signal_share else 0.0
        rows.append(row)
        print({k: round(v, 3) for k, v in row.items()})
out = Path(__file__).resolve().parents[1] / "data" / "update_alignment.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
