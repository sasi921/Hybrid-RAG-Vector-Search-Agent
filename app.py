"""Command line interface for indexing, retrieval, and evaluation."""
from __future__ import annotations

import argparse
import json

from hybrid_rag.agent import answer, ingest_folder
from hybrid_rag.embeddings import make_embedder
from hybrid_rag.evaluation import evaluate
from hybrid_rag.store import SQLiteStore


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid RAG Vector Search Agent")
    p.add_argument("--db", default="data/knowledge.sqlite", help="SQLite index path")
    p.add_argument("--embedder", choices=["hash", "semantic"], default="hash")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    sub = p.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Index .md/.txt files recursively")
    ingest.add_argument("--docs", default="examples/docs")
    ingest.add_argument("--chunk-words", type=int, default=110)
    ingest.add_argument("--overlap", type=int, default=20)
    ask = sub.add_parser("ask", help="Get source-cited extractive evidence")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=3)
    ev = sub.add_parser("evaluate", help="Evaluate source recall@k")
    ev.add_argument("--dataset", default="examples/eval.jsonl")
    ev.add_argument("--top-k", type=int, default=3)
    sub.add_parser("stats", help="Show indexed chunk count")
    args = p.parse_args()
    embedder = make_embedder(args.embedder, args.model)
    with SQLiteStore(args.db, embedder.signature) as store:
        if args.command == "ingest":
            result = ingest_folder(args.docs, store, embedder,
                                   max_words=args.chunk_words, overlap=args.overlap)
        elif args.command == "ask":
            result = answer(args.question, store, embedder, top_k=args.top_k)
        elif args.command == "evaluate":
            result = evaluate(args.dataset, store, embedder, top_k=args.top_k)
        else:
            result = {"chunks": store.count(), "embedder": embedder.signature}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
