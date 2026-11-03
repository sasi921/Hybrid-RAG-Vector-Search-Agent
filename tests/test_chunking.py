import pytest
from hybrid_rag.chunking import chunk_text


def test_chunk_overlap_and_empty():
    assert chunk_text("a b c d e", max_words=3, overlap=1) == ["a b c", "c d e"]
    assert chunk_text(" \n ") == []


def test_invalid_chunk_config():
    with pytest.raises(ValueError):
        chunk_text("abc", max_words=5, overlap=5)
