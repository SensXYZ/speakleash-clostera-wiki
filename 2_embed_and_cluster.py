#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "clostera",
#     "numpy",
#     "openai",
#     "python-dotenv",
# ]
# ///
"""Embed JSONL articles via LM Studio (OpenAI-compatible) and cluster them with clostera."""

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

import clostera
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", default="wiki_pl.jsonl",
                   help="input JSONL (output of 1_create_dataset.py). Default: wiki_pl.jsonl")
    p.add_argument("-o", "--output-dir", default="out",
                   help="directory for vectors.npy / labels.npy / *_clustered.jsonl. Default: out")
    p.add_argument("--base-url", default="http://localhost:1234/v1",
                   help="LM Studio OpenAI-compatible endpoint. Default: http://localhost:1234/v1")
    p.add_argument("--api-key", default=os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
                   help="API key for the server. Default: $LMSTUDIO_API_KEY or 'lm-studio'")
    p.add_argument("--model", required=True,
                   help="embedding model id loaded in LM Studio (e.g. text-embedding-bge-m3)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="texts per /v1/embeddings request. Default: 32")
    p.add_argument("--max-chars", type=int, default=2000,
                   help="truncate each article to N chars before sending. Default: 2000")
    p.add_argument("-k", "--clusters", type=int, default=64,
                   help="number of clusters for clostera. Default: 64")
    p.add_argument("--metric", default="cos", choices=["l2", "cos"],
                   help="clostera distance metric. Default: cos")
    p.add_argument("--algorithm", default="auto",
                   help="clostera algorithm name (see clostera.available_algorithms()). Default: auto")
    p.add_argument("--limit", type=int, default=None,
                   help="optional cap on number of articles (debug)")
    p.add_argument("--skip-embed", action="store_true",
                   help="skip embedding step and load vectors.npy from --output-dir")
    return p.parse_args()


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ckpt_meta_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".meta.json")


def _ckpt_meta(input_path: Path, n_records: int, model: str, max_chars: int, batch_size: int) -> dict:
    stat = input_path.stat()
    return {
        "input_path": str(input_path.resolve()),
        "input_mtime": stat.st_mtime,
        "input_size": stat.st_size,
        "n_records": n_records,
        "model": model,
        "max_chars": max_chars,
        "batch_size": batch_size,
    }


def _ckpt_valid(meta_path: Path, expected: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return saved == expected


def embed_all(records, client, model, batch_size, max_chars, checkpoint_path, input_path):
    """Returns float32 (N, D) array aligned with records.

    Resumes from checkpoint if checkpoint_path exists with matching metadata.
    Saves progress after each batch.
    """
    n = len(records)
    meta_path = _ckpt_meta_path(checkpoint_path)
    current_meta = _ckpt_meta(input_path, n, model, max_chars, batch_size)
    dim: int | None = None
    cursor = 0

    # Resume from checkpoint if available and config matches
    if checkpoint_path.exists() and _ckpt_valid(meta_path, current_meta):
        existing = np.load(checkpoint_path)
        if existing.shape[0] > 0 and existing.shape[0] < n:
            cursor = existing.shape[0]
            dim = existing.shape[1]
            print(f"  resuming from checkpoint: {cursor}/{n} already embedded",
                  file=sys.stderr)
    elif checkpoint_path.exists():
        print("  checkpoint config mismatch — starting from scratch", file=sys.stderr)
        checkpoint_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    vectors: np.ndarray | None = None
    t0 = time.time()
    for start in range(cursor, n, batch_size):
        batch = records[start : start + batch_size]
        inputs = [((r.get("text") or "")[:max_chars] or " ") for r in batch]
        resp = client.embeddings.create(model=model, input=inputs)
        # The API guarantees response order matches input order.
        for item in resp.data:
            v = np.asarray(item.embedding, dtype=np.float32)
            if vectors is None:
                if dim is None:
                    dim = v.shape[0]
                vectors = np.empty((n, dim), dtype=np.float32)
                # Pre-fill from checkpoint
                if cursor > 0:
                    existing = np.load(checkpoint_path)
                    vectors[:cursor] = existing
            vectors[start + item.index] = v
        # Save checkpoint + metadata after each batch
        if vectors is not None:
            done = min(start + batch_size, n)
            np.save(checkpoint_path, vectors[:done])
            meta_path.write_text(json.dumps(current_meta), encoding="utf-8")
        batch_num = start // batch_size
        if batch_num % 10 == 0 or start + batch_size >= n:
            done = min(start + batch_size, n)
            elapsed = time.time() - t0
            rate = (done - cursor) / elapsed if elapsed > 0 else 0
            print(f"  embedded {done}/{n}  ({rate:.1f} docs/s)", file=sys.stderr)
    return vectors


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "vectors.npy"
    labels_path = out_dir / "labels.npy"
    out_jsonl = out_dir / f"{in_path.stem}_clustered.jsonl"

    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 1

    records = []
    for rec in iter_jsonl(in_path):
        records.append(rec)
        if args.limit is not None and len(records) >= args.limit:
            break
    n_total = len(records)
    if n_total == 0:
        print("error: input is empty", file=sys.stderr)
        return 1
    print(f"loaded {n_total} records from {in_path}", file=sys.stderr)

    if args.skip_embed:
        if not vectors_path.exists():
            print(f"error: --skip-embed set but {vectors_path} not found", file=sys.stderr)
            return 1
        vectors = np.load(vectors_path).astype(np.float32, copy=False)
        if vectors.shape[0] != n_total:
            print(
                f"error: vectors.npy has {vectors.shape[0]} rows, JSONL has {n_total}",
                file=sys.stderr,
            )
            return 1
        print(f"loaded {vectors_path} shape={vectors.shape}", file=sys.stderr)
    else:
        print(f"embedding via {args.base_url} model={args.model!r}", file=sys.stderr)
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=5)
        ckpt_path = out_dir / "vectors_partial.npy"
        vectors = embed_all(
            records, client, args.model, args.batch_size, args.max_chars, ckpt_path, in_path
        )
        np.save(vectors_path, vectors)
        ckpt_path.unlink(missing_ok=True)
        _ckpt_meta_path(ckpt_path).unlink(missing_ok=True)
        print(f"wrote {vectors_path} shape={vectors.shape}", file=sys.stderr)

    print(
        f"clustering: k={args.clusters} metric={args.metric} algorithm={args.algorithm}",
        file=sys.stderr,
    )
    clusterer = clostera.Clusterer(
        k=args.clusters, metric=args.metric, algorithm=args.algorithm
    )
    labels = clusterer.fit_transform(vectors)
    print(f"clostera backend selected: {clusterer.algorithm_}", file=sys.stderr)

    np.save(labels_path, labels)
    print(f"wrote {labels_path}", file=sys.stderr)

    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)
    for old in clusters_dir.glob("cluster_*.jsonl"):
        old.unlink()

    per_cluster_files: dict[int, TextIO] = {}
    with ExitStack() as stack:
        combined = stack.enter_context(open(out_jsonl, "w", encoding="utf-8"))
        for rec, lbl in zip(records, labels, strict=True):
            cid = int(lbl)
            rec_out = dict(rec)
            rec_out["cluster"] = cid
            line = json.dumps(rec_out, ensure_ascii=False) + "\n"
            combined.write(line)
            fh = per_cluster_files.get(cid)
            if fh is None:
                fh = stack.enter_context(open(clusters_dir / f"cluster_{cid:03d}.jsonl", "w", encoding="utf-8"))
                per_cluster_files[cid] = fh
            fh.write(line)
    print(f"wrote {out_jsonl}", file=sys.stderr)
    print(f"wrote {len(per_cluster_files)} per-cluster files to {clusters_dir}/", file=sys.stderr)

    unique, counts = np.unique(labels, return_counts=True)
    print(
        f"cluster sizes: min={counts.min()} median={int(np.median(counts))} "
        f"max={counts.max()} (k={len(unique)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
