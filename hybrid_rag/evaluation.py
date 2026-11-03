"""Simple offline retrieval evaluation (source recall@k)."""
from __future__ import annotations

import json
from pathlib import Path

from .agent import retrieve
from .embeddings import Embedder
from .store import SQLiteStore


def evaluate(dataset: str | Path, store: SQLiteStore, embedder: Embedder, top_k: int = 3) -> dict:
    cases = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    hits = []
    details = []
    for case in cases:
        found = retrieve(case["question"], store, embedder, top_k=top_k)
        sources = {x["source"] for x in found}
        expected = case["expected_source"]
        hit = expected in sources
        hits.append(hit)
        details.append({"question": case["question"], "expected_source": expected, "hit": hit})
    return {"cases": len(cases), "recall_at_k": round(sum(hits) / len(hits), 4), "details": details}
