"""Toy: which singular-value weighting does a Dion step apply (one power iteration per step, Q column-normalized)?

    python scripts/dion_full_rank_sim.py        # numpy only, ~1 min on one CPU core; run it through Slurm

Gradients = a slowly rotating rank-32 signal with power-law strengths + isotropic noise. For each step
the update D = P Q^T is projected on the singular vectors of the matrix it was computed from
(w_i = u_i^T D v_i) and log w is fitted against log s: exponent 0 = Muon's polar factor, 1 = the
momentum itself. Also reported: cos(D, U V^T), cos(D, B) and the share of ||D||^2 in the top 1/4 directions.
"""
import numpy as np

rng = np.random.default_rng(0)
m, n, mu, steps, burn_in = 512, 256, 0.95, 300, 100


def orth(x):
    return np.linalg.qr(x)[0]


def cos(a, b):
    return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))


print(f"matrix {m}x{n}, mu {mu}, {steps} steps (stats after {burn_in})")
for drift in (0.0, 0.02, 0.1):
    for r in (n, n // 4):
        U0, V0 = orth(rng.standard_normal((m, 32))), orth(rng.standard_normal((n, 32)))
        sig = 3.0 * np.arange(1, 33) ** -1.0
        M = np.zeros((m, n))
        Q = orth(rng.standard_normal((n, r)))
        res = []
        for t in range(steps):
            U0 = orth(U0 + drift * 4 * rng.standard_normal((m, 32)) / np.sqrt(m))
            V0 = orth(V0 + drift * 4 * rng.standard_normal((n, 32)) / np.sqrt(n))
            G = U0 @ np.diag(sig) @ V0.T + 0.02 * rng.standard_normal((m, n))
            B = M + G
            P = orth(B @ Q)
            R = B.T @ P
            M = B - (1 - mu) * P @ R.T
            Q = R / np.linalg.norm(R, axis=0, keepdims=True)
            D = P @ Q.T
            if t >= burn_in:
                u, s, vt = np.linalg.svd(B, full_matrices=False)
                w = ((u.T @ D) * vt).sum(axis=1)
                keep = s > 1e-3 * s[0]
                p = np.polyfit(np.log(s[keep]), np.log(np.clip(np.abs(w[keep]), 1e-12, None)), 1)[0]
                top = int(np.ceil(0.25 * len(s)))
                res.append((cos(D, u @ vt), cos(D, B), p, (w[:top] ** 2).sum() / (w ** 2).sum()))
        a = np.array(res).mean(axis=0)
        print(f"drift {drift:4} rank {r:3}: cos(step, UV^T) {a[0]:.3f}  cos(step, B) {a[1]:.3f}  "
              f"weight ~ s^{a[2]:+.2f}  top-1/4 share of the step {a[3]:.2f}", flush=True)
print("reference: Muon polar s^0, share 0.25 (cos with B = the momentum's alignment); momentum itself s^1")
