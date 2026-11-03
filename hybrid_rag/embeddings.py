"""Offline lexical baseline and optional genuine semantic sentence embeddings.

The hashing baseline is *not* semantic: it is provided for deterministic, free
smoke tests. For semantic retrieval install requirements-semantic.txt.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    @property
    def signature(self) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else values


class HashingEmbedder:
    """Deterministic signed feature hashing; lexical rather than semantic."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    @property
    def signature(self) -> str:
        return f"hash-v1:{self.dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1 if digest[4] & 1 else -1
                vec[idx] += sign
            result.append(normalize(vec))
        return result


class SentenceTransformerEmbedder:
    """Local neural embeddings; model downloads once unless already cached."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install optional semantic dependencies: pip install -r requirements-semantic.txt"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def signature(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]


def make_embedder(kind: str, model: str) -> Embedder:
    if kind == "hash":
        return HashingEmbedder()
    if kind == "semantic":
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"Unknown embedder: {kind}")
