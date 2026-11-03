import pytest
from hybrid_rag.embeddings import HashingEmbedder
from hybrid_rag.store import cosine


def test_hash_vectors_are_deterministic_and_normalized():
    e = HashingEmbedder(256)
    a, b = e.embed(["retrieval index", "retrieval index"])
    assert a == b
    assert cosine(a, b) == pytest.approx(1.0)
    assert e.signature == "hash-v1:256"


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])
