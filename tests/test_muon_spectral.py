"""Unit tests of the muon-spectral patch set (spectral_update in emerging_optimizers.py), 1 GPU.

    SNAP=<snapshot with muon-zero1,muon-spectral> TEST=tests/test_muon_spectral.py NPROC=1 sbatch tests/run_dion_tests.sbatch
"""
import math
import sys
import time

import torch

from megatron.core.optimizer.emerging_optimizers import spectral_update
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def svals(x):
    return torch.linalg.svdvals(x.double())


def cos(a, b):
    return float((a.double() * b.double()).sum() / (a.double().norm() * b.double().norm()))


torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
for m, n in ((512, 256), (256, 512), (300, 300)):
    k = min(m, n)
    full = torch.randn(m, n, device=dev)
    low = torch.randn(m, 8, device=dev) @ torch.randn(8, n, device=dev) * 3 + 0.3 * torch.randn(m, n, device=dev)
    for gname, g in (("random", full), ("low-rank+noise", low)):
        tag = f"{m}x{n} {gname}"
        sg = svals(g)
        kept = sg > 1e-4 * sg.max()  # directions the update keeps (smaller ones count as numerically zero)
        nk = int(kept.sum())
        pol = spectral_update(g, "polar", 0.0)
        s = svals(pol)[:nk]
        check(f"polar {tag}: the {nk} kept singular values all equal", float(s.max() - s.min()) < 1e-3 * float(s.max()),
              f"(min {float(s.min()):.4f}, max {float(s.max()):.4f})")
        check(f"polar {tag}: Frobenius norm sqrt(k)", abs(float(pol.norm()) - math.sqrt(k)) < 1e-3 * math.sqrt(k))
        ns = newton_schulz(g, steps=5)
        c = cos(pol, ns)
        if gname == "random" and m != n:
            check(f"polar {tag}: same direction as Newton-Schulz (5 quintic steps)", c > 0.95, f"(cos {c:.4f})")
        else:  # NS under-weights directions below ~1e-3 of the largest: no exact match expected
            print(f"[INFO] polar {tag}: cos with Newton-Schulz {c:.4f}", flush=True)
        p1 = spectral_update(g, "power", 1.0)
        c = cos(p1, g)
        check(f"power p=1 {tag}: = normalized gradient", c > 0.9999, f"(cos {c:.6f})")
        ph = spectral_update(g, "power", 0.5)
        sh = svals(ph)
        ratio = sh[:nk] / sg[:nk].sqrt()
        check(f"power p=0.5 {tag}: singular values proportional to sqrt(s)",
              float(ratio.std() / ratio.mean()) < 1e-3, f"(rel. spread {float(ratio.std() / ratio.mean()):.1e})")
        r = math.ceil(0.25 * k)
        top = spectral_update(g, "top", 0.25)
        st = svals(top)
        check(f"top 1/4 {tag}: exactly r={r} equal non-zero singular values, norm sqrt(k)",
              int((st > 1e-3 * st.max()).sum()) == r and float(st[:r].std() / st[:r].mean()) < 1e-3
              and abs(float(top.norm()) - math.sqrt(k)) < 1e-3 * math.sqrt(k),
              f"(non-zero {int((st > 1e-3 * st.max()).sum())}, value {float(st[0]):.3f} = sqrt(k/r) {math.sqrt(k / r):.3f})")
        # top directions = those of the gradient: projection of top onto the top-r subspace of g
        u, _, vh = torch.linalg.svd(g.double(), full_matrices=False)
        proj = u[:, :r] @ (u[:, :r].T @ top.double() @ vh[:r].T) @ vh[:r]
        check(f"top 1/4 {tag}: lies in the gradient's top-r singular subspace",
              float((proj - top.double()).norm() / top.norm()) < 1e-3)
z = spectral_update(torch.zeros(64, 32, device=dev), "polar", 0.0)
check("zero gradient -> zero update", float(z.abs().max()) == 0.0)
b = spectral_update(torch.randn(256, 128, device=dev, dtype=torch.bfloat16), "power", 0.5)
check("bf16 input -> bf16 output, finite", b.dtype == torch.bfloat16 and bool(torch.isfinite(b).all()))

# time per matrix (TF32 on, as in training) vs Newton-Schulz
torch.backends.cuda.matmul.allow_tf32 = True
for m, n in ((7168, 1024), (1024, 3584), (2048, 1024), (24576, 4096), (4096, 12288)):
    g = torch.randn(m, n, device=dev)
    for name, fn in (("spectral", lambda x: spectral_update(x, "power", 0.5)), ("NS5", lambda x: newton_schulz(x, steps=5))):
        fn(g)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            fn(g)
        torch.cuda.synchronize()
        print(f"[INFO] {m}x{n} {name}: {(time.time() - t0) / 5 * 1e3:.1f} ms", flush=True)
print("ALL PASSED" if not FAILED else f"FAILED: {FAILED}", flush=True)
sys.exit(1 if FAILED else 0)
