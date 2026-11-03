import pytest
from hybrid_rag.store import SQLiteStore
from hybrid_rag.embeddings import HashingEmbedder


def test_persistence_fts_and_source_replace(tmp_path):
    path = tmp_path / "index.sqlite"
    embed = HashingEmbedder()
    with SQLiteStore(path, embed.signature) as store:
        store.replace_source("doc.txt", ["Kubernetes supports containers"], embed.embed(["Kubernetes supports containers"]))
        assert store.count() == 1
        assert store.lexical_search("Kubernetes")
        assert store.lexical_search('Kubernetes" OR *')  # MATCH syntax safely escaped
        store.replace_source("doc.txt", ["Secrets require approval"], embed.embed(["Secrets require approval"]))
        assert store.count() == 1
        assert store.lexical_search("Kubernetes") == []
    with SQLiteStore(path, embed.signature) as store:
        assert store.count() == 1
    with pytest.raises(ValueError, match="Index uses"):
        SQLiteStore(path, "semantic:other")


def test_prunes_missing_sources(tmp_path):
    e = HashingEmbedder()
    with SQLiteStore(tmp_path / "index.sqlite", e.signature) as store:
        store.replace_source("gone.txt", ["text"], e.embed(["text"]))
        store.remove_missing_sources(set())
        assert store.count() == 0
