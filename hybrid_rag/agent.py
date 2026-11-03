"""Hybrid retrieval agent: ingest, rank-fuse, and produce cited evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import chunk_text
from .embeddings import Embedder
from .store import SQLiteStore


def ingest_folder(
    folder: str | Path, store: SQLiteStore, embedder: Embedder,
    *, max_words: int = 110, overlap: int = 20,
) -> dict[str, int]:
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    docs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"})
    present = set()
    for doc in docs:
        if doc.stat().st_size > 2_000_000:
            raise ValueError(f"Document exceeds 2 MB safety limit: {doc}")
        source = doc.relative_to(root).as_posix()
        present.add(source)
        chunks = chunk_text(doc.read_text(encoding="utf-8"), max_words=max_words, overlap=overlap)
        vectors = embedder.embed(chunks)
        store.replace_source(source, chunks, vectors)
    store.remove_missing_sources(present)
    return {"documents": len(docs), "chunks": store.count()}


def retrieve(
    question: str, store: SQLiteStore, embedder: Embedder,
    *, top_k: int = 3, candidates: int = 25, rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if top_k < 1 or candidates < 1:
        raise ValueError("top_k and candidates must be >= 1")
    if not question.strip():
        return []
    vector = embedder.embed([question])[0]
    lexical = store.lexical_search(question, candidates)
    dense = [(i, score) for i, score in store.dense_search(vector, candidates)
             if score >= (0.10 if embedder.signature.startswith("hash-") else 0.20)]
    if embedder.signature.startswith("hash-"):
        # Hashing is lexical, not semantic. Only accept vector candidates that
        # also have an actual keyword match; abstain on out-of-domain queries.
        lexical_ids = {i for i, _ in lexical}
        dense = [(i, score) for i, score in dense if i in lexical_ids]
    scores: dict[int, float] = {}
    for ranking in (lexical, dense):
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    selected = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
    chunks = store.get_chunks(selected)
    return [
        {
            "source": chunks[i]["source"],
            "chunk": chunks[i]["ordinal"],
            "citation": f"{chunks[i]['source']}#chunk-{chunks[i]['ordinal']}",
            "score": round(scores[i], 6),
            "text": chunks[i]["body"],
        }
        for i in selected
    ]


def answer(question: str, store: SQLiteStore, embedder: Embedder, *, top_k: int = 3) -> dict:
    matches = retrieve(question, store, embedder, top_k=top_k)
    if not matches:
        return {"question": question, "answer": "No supporting evidence found.", "citations": [], "matches": []}
    # Extractive, not generative: never invent details not in source passages.
    excerpts = [f"{m['text'][:360]} [{m['citation']}]" for m in matches]
    return {
        "question": question,
        "answer": "\n\n".join(excerpts),
        "citations": [m["citation"] for m in matches],
        "matches": matches,
    }
