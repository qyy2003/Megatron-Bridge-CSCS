#!/usr/bin/env python3
"""Tokenize one source of the Lightning blend (parquet on the CSCS mirror) into Megatron bin/idx.

One output shard per input parquet file:
    <out-root>/<source>/<parquet stem>.{bin,idx}   int32 tokens, one document = text + EOS
    <out-root>/<source>/<parquet stem>.json        receipt (docs, tokens, tokenizer, timing)
The receipt is written last, so an interrupted run simply redoes that file, and a rerun with
a larger --target-tokens continues where the previous one stopped. Files are processed in a
seeded random order per source, so a partial tokenization is a random sample of the source.

Tokenizer: the Qwen3.5 tokenizer (byte-level BPE, no BOS, EOS <|endoftext|>), matching the
Lightning recipe's add_bos=false / add_eos=true. Megatron's IndexedDataset format is written
directly (same layout as megatron.core.datasets.indexed_dataset, verified by --verify).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import struct
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np


_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INT32_CODE = 4  # megatron.core.datasets.indexed_dataset.DType.int32

_TOK = None
_EOS = None


def _init_worker(tokenizer_json: str, eos_id: int) -> None:
    global _TOK, _EOS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer

    _TOK = Tokenizer.from_file(tokenizer_json)
    _EOS = eos_id


def _encode(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encs = _TOK.encode_batch(texts, add_special_tokens=False)
    lens = np.fromiter((len(e.ids) + 1 for e in encs), dtype=np.int32, count=len(encs))
    out = np.empty(int(lens.sum()), dtype=np.int32)
    pos = 0
    for e in encs:
        n = len(e.ids)
        out[pos : pos + n] = e.ids
        out[pos + n] = _EOS
        pos += n + 1
    return out, lens


def write_idx(path: str, lengths: np.ndarray) -> None:
    """Megatron IndexedDataset .idx: one sequence per document."""
    n = len(lengths)
    pointers = np.zeros(n, dtype=np.int64)
    if n > 1:
        pointers[1:] = np.cumsum(lengths[:-1], dtype=np.int64) * 4
    with open(path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", _INT32_CODE))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<Q", n + 1))
        f.write(lengths.astype(np.int32).tobytes(order="C"))
        f.write(pointers.tobytes(order="C"))
        f.write(np.arange(n + 1, dtype=np.int64).tobytes(order="C"))


def tokenize_file(pool: Pool, parquet: Path, text_field: str, prefix: Path, batch_rows: int,
                  max_rows: int | None, inflight: int) -> dict:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet)
    if text_field not in pf.schema_arrow.names:
        raise SystemExit(f"{parquet}: no column {text_field!r} (columns: {pf.schema_arrow.names})")
    t0 = time.time()
    rows = skipped = 0
    lengths = []
    pending = collections.deque()
    tmp_bin = Path(f"{prefix}.bin.tmp")  # not with_suffix: stems may contain dots
    with open(tmp_bin, "wb") as fbin:

        def drain(limit: int) -> None:
            while len(pending) > limit:
                toks, lens = pending.popleft().get()
                fbin.write(toks.tobytes(order="C"))
                lengths.append(lens)

        for batch in pf.iter_batches(batch_size=batch_rows, columns=[text_field]):
            texts = batch.column(0).to_pylist()
            if max_rows is not None:
                texts = texts[: max(0, max_rows - rows)]
            rows += len(texts)
            kept = [t for t in texts if t]
            skipped += len(texts) - len(kept)
            if kept:
                pending.append(pool.apply_async(_encode, (kept,)))
                drain(inflight)
            if max_rows is not None and rows >= max_rows:
                break
        drain(0)
    lens = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int32)
    write_idx(f"{prefix}.idx", lens)
    os.replace(tmp_bin, f"{prefix}.bin")
    return {"rows": rows, "docs": int(len(lens)), "empty_skipped": skipped, "tokens": int(lens.sum()),
            "seconds": round(time.time() - t0, 1)}


def verify(prefix: Path, tokenizer_json: str, n: int = 3) -> None:
    """Read the shard back with Megatron's IndexedDataset and decode a few documents."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from tokenizers import Tokenizer

    ds = IndexedDataset(str(prefix), mmap=True)
    tok = Tokenizer.from_file(tokenizer_json)
    print(f"[verify] {prefix.name}: {len(ds)} docs, dtype={ds.index.dtype}, docs_idx={len(ds.document_indices)}")
    for i in (0, len(ds) // 2, len(ds) - 1)[:n]:
        ids = ds[i]
        print(f"[verify]  doc {i}: {len(ids)} tokens, last={ids[-1]}, text={tok.decode(ids[:40].tolist())!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", required=True, help="data/lightning35_sources.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="source name")
    g.add_argument("--source-index", type=int, help="index into the sources list (Slurm array id)")
    ap.add_argument("--mirror", required=True, help="$SOURCE_MIRROR")
    ap.add_argument("--out-root", required=True, help="output root, e.g. $DATA_ROOT/qwen35")
    ap.add_argument("--tokenizer-dir", required=True, help="HF snapshot dir with tokenizer.json")
    ap.add_argument("--target-tokens", type=float, default=25e9, help="stop once this source has this many tokens")
    ap.add_argument("--max-files", type=int, default=None, help="process at most N new files")
    ap.add_argument("--max-rows-per-file", type=int, default=None, help="testing: only the first N rows per file")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 8))
    ap.add_argument("--batch-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--verify", action="store_true", help="read back each new shard with Megatron's reader")
    args = ap.parse_args()

    sources = json.loads(Path(args.sources).read_text())["sources"]
    src = sources[args.source_index] if args.source_index is not None else next(
        s for s in sources if s["name"] == args.source)
    in_dir = Path(args.mirror) / src["mirror"]
    out_dir = Path(args.out_root) / src["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = Path(args.tokenizer_dir)
    tokenizer_json = str(tok_dir / "tokenizer.json")
    tok_cfg = json.loads((tok_dir / "tokenizer_config.json").read_text())
    added = {t["content"]: t["id"] for t in json.loads(Path(tokenizer_json).read_text())["added_tokens"]}
    eos_id = added[tok_cfg["eos_token"]]

    files = sorted(in_dir.glob("*.parquet"))
    rng = random.Random(f"{args.seed}:{src['name']}")
    rng.shuffle(files)

    def receipt(f: Path) -> Path:
        return out_dir / (f.stem + ".json")

    def is_done(f: Path) -> bool:
        # A partial (--max-rows-per-file) shard only counts as done for another partial run.
        if not receipt(f).exists():
            return False
        return args.max_rows_per_file is not None or not json.loads(receipt(f).read_text()).get("partial", False)

    done = [json.loads(receipt(f).read_text()) for f in files if is_done(f)]
    have = sum(r["tokens"] for r in done)
    print(f"[{src['name']}] {len(files)} parquet files in {in_dir}; {len(done)} done, {have / 1e9:.2f}B tokens; "
          f"target {args.target_tokens / 1e9:.1f}B; eos={eos_id}; workers={args.workers}", flush=True)
    todo = [f for f in files if not is_done(f)]
    if args.max_files is not None:
        todo = todo[: args.max_files]

    with Pool(args.workers, initializer=_init_worker, initargs=(tokenizer_json, eos_id)) as pool:
        for f in todo:
            if have >= args.target_tokens:
                break
            prefix = out_dir / f.stem
            stats = tokenize_file(pool, f, src.get("text_field", "text"), prefix, args.batch_rows,
                                  args.max_rows_per_file, inflight=4 * args.workers)
            have += stats["tokens"]
            rec = {"source": src["name"], "parquet": str(f), "parquet_bytes": f.stat().st_size,
                   "prefix": str(prefix), "partial": args.max_rows_per_file is not None,
                   "tokenizer": str(tok_dir), "eos_id": eos_id, "dtype": "int32",
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats}
            tmp = Path(f"{receipt(f)}.tmp")
            tmp.write_text(json.dumps(rec, indent=1))
            os.replace(tmp, receipt(f))
            rate = stats["tokens"] / max(stats["seconds"], 1e-3) / 1e6
            print(f"[{src['name']}] {f.name}: {stats['docs']} docs {stats['tokens'] / 1e9:.3f}B tok "
                  f"in {stats['seconds']}s ({rate:.1f}M tok/s) | source total {have / 1e9:.2f}B", flush=True)
            if args.verify:
                verify(prefix, tokenizer_json)
    print(f"[{src['name']}] finished with {have / 1e9:.2f}B tokens", flush=True)


if __name__ == "__main__":
    sys.exit(main())
