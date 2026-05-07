#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "speakleash",
# ]
# ///
"""Sample random Polish Wikipedia articles via the speakleash package and write JSONL."""

import argparse
import json
import os
import random
import sys

from speakleash import Speakleash


def parse_count(value: str) -> int:
    """Accept '10000', '10k', '100K', '1M' style values."""
    s = value.strip().lower().replace("_", "")
    multiplier = 1
    if s.endswith("k"):
        multiplier, s = 1_000, s[:-1]
    elif s.endswith("m"):
        multiplier, s = 1_000_000, s[:-1]
    n = int(float(s) * multiplier)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"count must be positive, got {value!r}")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample random Polish Wikipedia articles via speakleash.",
    )
    parser.add_argument(
        "-n", "--count", type=parse_count, default=parse_count("10k"),
        help="number of articles to sample (e.g. 10000, 10k, 100k). Default: 10k",
    )
    parser.add_argument(
        "-o", "--output", default="wiki_pl.jsonl",
        help="output JSONL path. Default: wiki_pl.jsonl",
    )
    parser.add_argument(
        "--dataset", default="plwiki",
        help="speakleash dataset name. Default: plwiki",
    )
    parser.add_argument(
        "--cache-dir", default="datasets",
        help="local cache directory for speakleash downloads. Default: ./datasets",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for reproducible sampling",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    sl = Speakleash(cache_dir)
    ds = sl.get(args.dataset)
    if ds is None:
        print(f"error: dataset {args.dataset!r} not found", file=sys.stderr)
        return 1

    total = int(ds.documents or 0)
    if total <= 0:
        print(f"error: dataset {args.dataset!r} reports {total} documents", file=sys.stderr)
        return 1

    n = min(args.count, total)
    if n < args.count:
        print(
            f"warning: requested {args.count} but dataset has only {total}; sampling all",
            file=sys.stderr,
        )

    print(f"sampling {n} of {total} articles from {args.dataset!r}", file=sys.stderr)
    chosen = set(random.sample(range(total), n))

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for idx, (text, meta) in enumerate(ds.ext_data):
            if idx not in chosen:
                continue
            record = {"text": text, "meta": meta or {}}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  {written}/{n}", file=sys.stderr)
            if written == n:
                break

    print(f"done: {written} articles -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
