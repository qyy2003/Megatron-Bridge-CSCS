"""Unit tests of megatron/core/optimizer/dion.py (optional patch set "dion"), 4 GPUs of one node.

    srun ... torchrun --nproc_per_node=4 tests/test_dion.py      (tests/run_dion_tests.sbatch)

1. single rank vs an independent fp64 implementation of the paper's algorithm;
2. two optimizer instances (ranks 0+1 and 2+3, each pair an inter-instance group), every rank with
   its own gradients g_i: weights bitwise identical within the pair, equal to single-instance
   Dion on sum_i g_i; momentum averaging (done before checkpoints) changes nothing;
3. bf16 communication: still bitwise identical within the pair, close to the reference;
4. a matrix with zero gradients stays finite; mismatched matrices across instances are refused;
5. time of one optimizer step on a realistic per-rank set of matrices: Dion (r = 1/4, 1/16)
   vs Muon (emerging_optimizers, 5 Newton-Schulz steps).
"""

import math
import os
import sys
import time

import torch
import torch.distributed as dist

from megatron.core.optimizer.dion import Dion, dion_scale_factor, orthonormalize

SHAPES = [(256, 512), (512, 128), (300, 200), (256, 512)]
LR, MU, WD = 0.02, 0.95, 0.1
FAILED = []


def check(name, ok, detail=""):
    rank = dist.get_rank()
    flag = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flag)
    if rank == 0:
        print(f"[{'PASS' if flag.item() == 0 else 'FAIL'}] {name} {detail}", flush=True)
    if flag.item():
        FAILED.append(name)


def make_params(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(m, n, device="cuda", generator=g) * 0.02 for m, n in SHAPES]


def grads_for(step, instance, seed=100):
    g = torch.Generator(device="cuda").manual_seed(seed + 1000 * step + instance)
    return [torch.randn(m, n, device="cuda", generator=g) for m, n in SHAPES]


def reference(xs0, grad_fn, steps, rank_fraction, mult=8, q_seed=1234):
    """Dion from the paper, fp64, one process; grad_fn(step) = the global gradient."""
    xs = [x.double().clone() for x in xs0]
    ms = [torch.zeros_like(x) for x in xs]
    qs = []
    for i, x in enumerate(xs):
        k = min(x.shape)
        r = max(1, min(k, math.ceil(math.ceil(rank_fraction * k) / mult) * mult))
        gen = torch.Generator(device="cuda").manual_seed(q_seed + i)
        q = torch.randn(x.shape[1], r, device="cuda", dtype=torch.float32, generator=gen)
        qs.append((q / q.norm(dim=0, keepdim=True)).double())
    for t in range(steps):
        gs = grad_fn(t)
        for i, x in enumerate(xs):
            b = ms[i] + gs[i].double()
            p, _ = torch.linalg.qr(b @ qs[i])
            r_ = b.t() @ p
            ms[i] = b - (1 - MU) * p @ r_.t()
            qs[i] = r_ / (r_.norm(dim=0, keepdim=True) + 1e-8)
            x.mul_(1 - LR * WD)
            x.add_(p @ qs[i].t(), alpha=-LR * dion_scale_factor(x.shape[0], x.shape[1], "spectral"))
    return xs


def run_dion(xs0, grad_fn, steps, rank_fraction, group, comm_dtype="fp32", average_at=None):
    xs = [torch.nn.Parameter(x.clone()) for x in xs0]
    opt = Dion(xs, lr=LR, mu=MU, weight_decay=WD, rank_fraction=rank_fraction,
               comm_dtype=comm_dtype, inter_instance_group=group)
    for t in range(steps):
        for x, g in zip(xs, grad_fn(t)):
            x.grad = g.clone()
        opt.step()
        if average_at is not None and t == average_at:
            opt.average_momentum_across_instances()
    return [x.detach() for x in xs], opt


def _chol_qr2(p):
    for _ in range(2):
        chol, _ = torch.linalg.cholesky_ex(p.mT @ p)
        p = torch.linalg.solve_triangular(chol, p.mT, upper=False).mT
    return p


def max_rel(a_list, b_list):
    return max(float((a.double() - b.double()).norm() / b.double().norm()) for a, b in zip(a_list, b_list))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    assert world == 4, "run with 4 ranks"
    if rank == 0:
        print(f"[INFO] TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')} "
              f"allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
              f"fp32 matmul precision={torch.get_float32_matmul_precision()}", flush=True)
    # Exactness checks in true fp32 (the container enables TF32 for fp32 GEMMs by default).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    pairs = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    pair = pairs[rank // 2]
    instance = rank % 2  # position inside the pair = optimizer instance
    steps, frac = 8, 0.25

    # 1. single rank (no instances) vs fp64 reference, global gradient = grads_for(t, 0)
    xs0 = make_params(7)
    ours, _ = run_dion(xs0, lambda t: grads_for(t, 0), steps, frac, group=None)
    ref = reference(xs0, lambda t: grads_for(t, 0), steps, frac)
    err = max_rel(ours, ref)
    check("single instance == fp64 reference", err < 1e-4, f"(max rel. error {err:.2e})")
    ours_full, _ = run_dion(xs0, lambda t: grads_for(t, 0), steps, 1.0, group=None)
    ref_full = reference(xs0, lambda t: grads_for(t, 0), steps, 1.0)
    err = max_rel(ours_full, ref_full)
    check("full rank == fp64 reference", err < 1e-4, f"(max rel. error {err:.2e})")

    # 2. two instances: instance i sees g_i; the global gradient is g_0 + g_1
    def local(t):
        return grads_for(t, instance)

    def global_(t):
        return [a + b for a, b in zip(grads_for(t, 0), grads_for(t, 1))]

    ours2, opt2 = run_dion(xs0, local, steps, frac, group=pair)
    same = True
    for x in ours2:
        buf = [torch.empty_like(x) for _ in range(2)]
        dist.all_gather(buf, x.contiguous(), group=pair)
        same &= torch.equal(buf[0], buf[1])
    check("2 instances: weights bitwise identical across instances", same)
    check("2 instances: sync check reports 0", opt2.check_instances_in_sync() == 0.0)
    ref2 = reference(xs0, global_, steps, frac)
    err = max_rel(ours2, ref2)
    check("2 instances == single-instance Dion on the summed gradient", err < 1e-4, f"(max rel. error {err:.2e})")
    ours_avg, _ = run_dion(xs0, local, steps, frac, group=pair, average_at=3)
    err = max_rel(ours_avg, ours2)
    check("momentum averaging across instances is a no-op", err < 1e-5, f"(max rel. diff {err:.2e})")

    # 3. bf16 communication
    ours_bf, _ = run_dion(xs0, local, steps, frac, group=pair, comm_dtype="bf16")
    same = True
    for x in ours_bf:
        buf = [torch.empty_like(x) for _ in range(2)]
        dist.all_gather(buf, x.contiguous(), group=pair)
        same &= torch.equal(buf[0], buf[1])
    check("bf16 comm: weights bitwise identical across instances", same)
    err = max_rel(ours_bf, ref2)
    check("bf16 comm: close to the fp64 reference", err < 3e-2, f"(max rel. error {err:.2e})")

    # orthonormalize: random, rank-deficient (rank r/2) and zero matrices
    g = torch.Generator(device="cuda").manual_seed(5)
    full = torch.randn(6, 300, 40, device="cuda", generator=g)
    low = torch.randn(6, 300, 20, device="cuda", generator=g) @ torch.randn(6, 20, 40, device="cuda", generator=g)
    cases = {"random": full, "rank-deficient": low, "zero": torch.zeros(6, 300, 40, device="cuda")}
    for name, mat in cases.items():
        q = orthonormalize(mat)
        ortho = float((q.mT @ q - torch.eye(40, device="cuda")).abs().max())
        span = float((q @ (q.mT @ mat) - mat).norm() / (mat.norm() + 1e-30))
        check(f"orthonormalize {name}: orthonormal columns spanning P", ortho < 1e-4 and span < 1e-4,
              f"(|QtQ-I| {ortho:.1e}, span residual {span:.1e})")

    # 4. zero gradient for one matrix (an expert without tokens) / mismatched layouts
    def zero_first(t):
        g = local(t)
        g[0].zero_()
        return g

    ours_z, opt_z = run_dion(xs0, zero_first, steps, frac, group=pair)
    finite = all(torch.isfinite(x).all() for x in ours_z) and all(torch.isfinite(q).all() for q in opt_z._q.values())
    check("zero-gradient matrix stays finite", bool(finite))
    bad = [torch.nn.Parameter(torch.randn(64 if instance == 0 else 72, 32, device="cuda"))]
    opt_bad = Dion(bad, lr=LR, inter_instance_group=pair)
    bad[0].grad = torch.randn_like(bad[0])
    try:
        opt_bad.step()
        refused = False
    except RuntimeError:
        refused = True
    check("mismatched matrices across instances are refused", refused)

    # TF32 (the training default in this container): deviation from the fp64 reference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    ours_tf, _ = run_dion(xs0, local, steps, frac, group=pair)
    err = max_rel(ours_tf, ref2)
    check("TF32: 2 instances close to the fp64 reference", err < 1e-2, f"(max rel. error {err:.2e})")

    # 5. step time on a realistic per-rank set (8 nodes, GLOBAL_DP=2, stage 0: ~640 fc1 + 640 fc2
    #    expert matrices + dense matrices); one rank, no communication
    if rank == 0:
        shapes = [(1024, 2048)] * 640 + [(2048, 512)] * 640 + [(4096, 2048)] * 8 + [(2048, 4096)] * 8
        params = [torch.nn.Parameter(torch.randn(m, n, device="cuda") * 0.02) for m, n in shapes]
        for p in params:
            p.grad = torch.randn_like(p) * 1e-3
        numel = sum(p.numel() for p in params)
        results = []
        # where the time goes (one step, r = 1/4)
        opt = Dion(params, lr=1e-4, rank_fraction=0.25)
        opt.step()
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            opt.step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14), flush=True)
        del opt
        torch.cuda.empty_cache()
        # orthonormalization alone on one shape run: Householder QR vs Cholesky QR (2 passes)
        for k, m, r in ((640, 1024, 256), (640, 2048, 128)):
            pm = torch.randn(k, m, r, device="cuda")
            for name, fn in (("householder", lambda x: torch.linalg.qr(x)[0]), ("cholesky2", _chol_qr2)):
                fn(pm)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3):
                    q = fn(pm)
                torch.cuda.synchronize()
                ortho = float((q.mT @ q - torch.eye(r, device="cuda")).abs().max())
                results.append(f"QR {name} {k}x{m}x{r}: {(time.time() - t0) / 3 * 1e3:.0f} ms (|QtQ-I| {ortho:.1e})")
        for frac_b in (0.25, 1 / 16):
            opt = Dion(params, lr=1e-4, rank_fraction=frac_b)
            opt.step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 3
            comm = sum((p.shape[0] + p.shape[1]) * opt.rank_for(p) for p in params)
            results.append(f"Dion r={frac_b:.4g}: {dt * 1e3:.0f} ms/step, factors {comm / numel:.3f} of the matrix values")
            del opt
            torch.cuda.empty_cache()
        try:
            from emerging_optimizers import registry

            muon_cls = registry.get_optimizer_cls("muon")
            opt = muon_cls(params, lr=1e-4)
            opt.step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                opt.step()
            torch.cuda.synchronize()
            results.append(f"Muon (emerging_optimizers, NS5): {(time.time() - t0) / 3 * 1e3:.0f} ms/step")
        except Exception as e:  # noqa: BLE001
            results.append(f"Muon timing skipped: {e!r}")
        print(f"[INFO] {len(params)} matrices, {numel / 1e9:.2f}B values:\n  " + "\n  ".join(results), flush=True)
    dist.barrier()
    if rank == 0:
        print("ALL PASSED" if not FAILED else f"FAILED: {FAILED}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
