"""Deterministic word-window chunking with overlap."""
from __future__ import annotations


def chunk_text(text: str, *, max_words: int = 110, overlap: int = 20) -> list[str]:
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("Expected max_words > 0 and 0 <= overlap < max_words")
    words = text.split()
    if not words:
        return []
    stride = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), stride)
            if i == 0 or len(words) - i > overlap]
