#!/usr/bin/env python3
"""Build the Megatron data blend for train/pretrain.py from the tokenized shards.

Weighting (--weights):
  lightning  (default) the Nemotron 3.5 Lightning open-data recipe: every source gets its
             weight from data/lightning35_sources.json (all 1.0 -> a uniform mixture over the
             23 sources, whatever their size). Small sources are therefore repeated: the table
             printed below shows the implied epochs for --train-tokens.
  natural    proportional to tokens (each source by its tokenized size).
--source-weights FILE.json ({"<source>": weight, ...}) overrides per-source weights, e.g. for a
second, quality-heavier phase (the Lightning recipe's phase 2 upweights High-Quality,
High-Quality-Synthetic and Specialized-v1; NVIDIA does not publish the exact numbers).

Within a source, each shard is weighted by its token count. Train/valid/test are drawn from
every shard with --split, so validation loss is measured on the same mixture.

Output: {"data_paths": [w1, prefix1, ...], "split": "...", "meta": {...}} for --blend.
Standard library only (runs on the login node).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="tokenized root [$DATA_ROOT/qwen35]")
    ap.add_argument("--sources", default=str(Path(__file__).with_name("lightning35_sources.json")))
    ap.add_argument("--weights", choices=["lightning", "natural"], default="lightning")
    ap.add_argument("--source-weights", help="JSON {source: weight} overriding the per-source weights")
    ap.add_argument("--split", default="9990,8,2")
    ap.add_argument("--train-tokens", type=float, default=500e9, help="planned training tokens (for the epoch report)")
    ap.add_argument("--out", default=None, help="output JSON [$DATA_ROOT/blends/lightning35_qwen35.json]")
    ap.add_argument("--allow-partial", action="store_true", help="accept smoke-test shards (MAX_ROWS)")
    args = ap.parse_args()

    import os

    data_root = Path(args.data_root or os.path.join(os.environ["DATA_ROOT"], "qwen35"))
    out = Path(args.out or os.path.join(os.environ["DATA_ROOT"], "blends", "lightning35_qwen35.json"))
    sources = json.loads(Path(args.sources).read_text())["sources"]
    overrides = json.loads(Path(args.source_weights).read_text()) if args.source_weights else {}
    unknown = set(overrides) - {s["name"] for s in sources}
    if unknown:
        sys.exit(f"--source-weights: unknown sources {sorted(unknown)}")

    rows, missing, partial = [], [], 0
    for s in sources:
        receipts = [json.loads(p.read_text()) for p in sorted((data_root / s["name"]).glob("*.json"))]
        partial += sum(r.get("partial", False) for r in receipts)
        shards = [(r["prefix"], r["tokens"], r["docs"]) for r in receipts if r["tokens"] > 0]
        if not shards:
            missing.append(s["name"])
            continue
        w = float(overrides.get(s["name"], s.get("weight", 1.0)))
        rows.append({"name": s["name"], "weight": w, "shards": shards,
                     "tokens": sum(t for _, t, _ in shards), "docs": sum(d for _, _, d in shards)})
    if partial and not args.allow_partial:
        sys.exit(f"{partial} shards come from a MAX_ROWS smoke test; pass --allow-partial to use them")
    if not rows:
        sys.exit(f"no tokenized shards under {data_root}")
    if missing:
        print(f"WARNING: {len(missing)} sources have no shards yet and are left out: {missing}")

    total_tokens = sum(r["tokens"] for r in rows)
    if args.weights == "natural" and not overrides:
        for r in rows:
            r["weight"] = r["tokens"]
    wsum = sum(r["weight"] for r in rows)
    data_paths = []
    for r in rows:
        r["share"] = r["weight"] / wsum
        for prefix, tokens, _ in r["shards"]:
            data_paths += [f"{r['share'] * tokens / r['tokens']:.12g}", prefix]

    print(f"{'source':62s} {'shards':>6s} {'tokens':>9s} {'share':>7s} {'epochs@' + format(args.train_tokens / 1e9, '.0f') + 'B':>12s}")
    for r in rows:
        epochs = r["share"] * args.train_tokens / r["tokens"]
        print(f"{r['name']:62s} {len(r['shards']):6d} {r['tokens'] / 1e9:8.2f}B {100 * r['share']:6.2f}% {epochs:12.2f}")
    print(f"{'total':62s} {sum(len(r['shards']) for r in rows):6d} {total_tokens / 1e9:8.2f}B")

    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"weights": args.weights, "source_weights": overrides, "data_root": str(data_root),
            "total_tokens": total_tokens, "sources": {r["name"]: {"tokens": r["tokens"], "docs": r["docs"],
                                                                 "shards": len(r["shards"]), "share": r["share"]}
                                                     for r in rows}}
    out.write_text(json.dumps({"data_paths": data_paths, "split": args.split, "meta": meta}, indent=1))
    print(f"wrote {out} ({len(data_paths) // 2} prefixes)")


if __name__ == "__main__":
    main()
