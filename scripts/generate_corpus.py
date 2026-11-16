"""Regenerate data/corpus.jsonl and data/queries.jsonl (deterministic)."""

import argparse
from pathlib import Path

from agentic_search.synth.generate import write_corpus

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=457)
    args = ap.parse_args()
    n_docs, n_queries = write_corpus(Path(args.out), args.seed)
    print(f"wrote {n_docs} documents and {n_queries} labeled queries to {args.out}/")
