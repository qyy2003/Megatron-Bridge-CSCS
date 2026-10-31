#!/usr/bin/env python3
"""Singular-value spectra of the optimizer momentum (or weights) stored in a torch_dist checkpoint.

    python tools/ckpt_spectra.py --ckpt <run>/checkpoints/iter_0000500 --out <dir> [--layers 0,8,16,24,31]

Runs inside the training container (torch); no process group needed. Writes <out>/keys.json (every
tensor key with its shape) and <out>/spectra.csv with, per selected matrix:

  energy_top{1/16,1/4,1/2}  share of ||M||_F^2 in the top 1/16, 1/4, 1/2 of the singular directions
  stable_rank               ||M||_F^2 / sigma_1^2
  effective_rank            exp(entropy of sigma_i^2 / sum sigma^2)
  muon_alignment            cos(U V^T, M) = sum(sigma) / (sqrt(k) ||M||_F): how much of a fully whitened
                            (Muon) step points along M (1 = flat spectrum, small = few strong directions)
  sigma_q{50,90}            the median and 90th-percentile singular value relative to sigma_1

Stacked tensors (Megatron stores per-layer weights as [layers, out, in]) are split per layer.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

MODULES = ("linear_fc1", "linear_fc2", "linear_qkv", "linear_proj", "in_proj", "out_proj", "eh_proj")


def stats(m: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(m.float().cuda()).double()
    k = s.numel()
    e = s.pow(2)
    tot = float(e.sum())
    if tot == 0:
        return {"k": k, "zero": True}
    p = e / tot
    out = {"k": k, "fro": math.sqrt(tot), "sigma1": float(s[0])}
    for name, f in (("1/16", 1 / 16), ("1/4", 1 / 4), ("1/2", 1 / 2)):
        out[f"energy_top{name}"] = float(e[: max(1, math.ceil(f * k))].sum() / tot)
    out["stable_rank"] = tot / float(e[0])
    out["effective_rank"] = float(torch.exp(-(p * torch.log(p.clamp(min=1e-300))).sum()))
    out["muon_alignment"] = float(s.sum() / (math.sqrt(k) * math.sqrt(tot)))
    out["sigma_q50"] = float(s[k // 2] / s[0])
    out["sigma_q90"] = float(s[min(k - 1, int(0.1 * k))] / s[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default="momentum_buffer", help="substring of the keys to analyse ('' = weights)")
    ap.add_argument("--layers", default="0,8,16,24,31")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = FileSystemReader(args.ckpt)
    md = reader.read_metadata().state_dict_metadata
    keys = {k: list(v.size) for k, v in md.items() if isinstance(v, TensorStorageMetadata)}
    json.dump(keys, open(out / "keys.json", "w"), indent=0)
    want_layers = [int(x) for x in args.layers.split(",")]
    sel = []
    for k, shape in keys.items():
        if args.state and args.state not in k:
            continue
        if not args.state and ("optimizer" in k or not k.endswith("weight")):
            continue
        if len(shape) < 2 or not any(mod in k for mod in MODULES) or "expert" in k:
            continue
        sel.append(k)
    print(f"{len(keys)} tensor keys, {len(sel)} selected:", *sel[:40], sep="\n  ")
    rows = []
    for k in sel:
        shape = keys[k]
        t = torch.empty(shape, dtype=md[k].properties.dtype)
        dcp.load({k: t}, storage_reader=FileSystemReader(args.ckpt), no_dist=True)
        mats = []
        if len(shape) == 3:  # [layers, out, in]
            for li in want_layers:
                if li < shape[0]:
                    mats.append((li, t[li]))
        else:
            m = re.search(r"layers\.(\d+)\.", k)
            mats.append((int(m.group(1)) if m else -1, t))
        for li, mat in mats:
            st = stats(mat)
            row = {"key": k, "layer": li, "shape": "x".join(map(str, mat.shape)),
                   "module": next(mod for mod in MODULES if mod in k), "mtp": "mtp" in k, **st}
            rows.append(row)
            print(f"{row['module']:12s} layer {li:3d} {row['shape']:>12s} mtp={row['mtp']} "
                  f"top1/16 {st.get('energy_top1/16', 0):.3f} top1/4 {st.get('energy_top1/4', 0):.3f} "
                  f"stable {st.get('stable_rank', 0):7.1f} eff {st.get('effective_rank', 0):7.1f} "
                  f"muon_align {st.get('muon_alignment', 0):.3f}", flush=True)
        del t
    if rows:
        fields = sorted({f for r in rows for f in r}, key=lambda f: (f not in ("key", "layer", "module", "mtp", "shape"), f))
        with open(out / "spectra.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out}/spectra.csv ({len(rows)} matrices)")


if __name__ == "__main__":
    main()
