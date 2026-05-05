#!/usr/bin/env python3
"""Embed JSONL articles via LM Studio (OpenAI-compatible) and cluster them with clostera."""

import argparse
import json
import sys
import time
from pathlib import Path

import clostera
import numpy as np
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", default="wiki_pl.jsonl",
                   help="input JSONL (output of 1_create_dataset.py). Default: wiki_pl.jsonl")
    p.add_argument("-o", "--output-dir", default="out",
                   help="directory for vectors.npy / labels.npy / *_clustered.jsonl. Default: out")
    p.add_argument("--base-url", default="http://localhost:1234/v1",
                   help="LM Studio OpenAI-compatible endpoint. Default: http://localhost:1234/v1")
    p.add_argument("--api-key", default="lm-studio",
                   help="any non-empty string works for local LM Studio")
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
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def embed_all(records, client, model, batch_size, max_chars):
    """Returns float32 (N, D) array aligned with records."""
    n = len(records)
    vectors = None
    cursor = 0
    t0 = time.time()
    for start in range(0, n, batch_size):
        batch = records[start : start + batch_size]
        inputs = [((r.get("text") or "")[:max_chars] or " ") for r in batch]
        resp = client.embeddings.create(model=model, input=inputs)
        # The API guarantees response order matches input order.
        for item in resp.data:
            v = np.asarray(item.embedding, dtype=np.float32)
            if vectors is None:
                vectors = np.empty((n, v.shape[0]), dtype=np.float32)
            vectors[cursor] = v
            cursor += 1
        if start // batch_size % 10 == 0 or cursor == n:
            elapsed = time.time() - t0
            rate = cursor / elapsed if elapsed > 0 else 0
            print(f"  embedded {cursor}/{n}  ({rate:.1f} docs/s)", file=sys.stderr)
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
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        vectors = embed_all(
            records, client, args.model, args.batch_size, args.max_chars
        )
        np.save(vectors_path, vectors)
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

    combined = open(out_jsonl, "w", encoding="utf-8")
    per_cluster_files: dict[int, "object"] = {}
    try:
        for rec, lbl in zip(records, labels):
            cid = int(lbl)
            rec_out = dict(rec)
            rec_out["cluster"] = cid
            line = json.dumps(rec_out, ensure_ascii=False) + "\n"
            combined.write(line)
            fh = per_cluster_files.get(cid)
            if fh is None:
                fh = open(clusters_dir / f"cluster_{cid:03d}.jsonl", "w", encoding="utf-8")
                per_cluster_files[cid] = fh
            fh.write(line)
    finally:
        combined.close()
        for fh in per_cluster_files.values():
            fh.close()
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
