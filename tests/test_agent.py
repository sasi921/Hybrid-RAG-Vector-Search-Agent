from pathlib import Path
from hybrid_rag.agent import answer, ingest_folder, retrieve
from hybrid_rag.embeddings import HashingEmbedder
from hybrid_rag.evaluation import evaluate
from hybrid_rag.store import SQLiteStore


def test_ingest_answer_evaluate(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "security.txt").write_text("API secrets require human approval and auditing.")
    (docs / "deploy.md").write_text("Kubernetes deployments use continuous integration tests.")
    embed = HashingEmbedder()
    with SQLiteStore(tmp_path / "index.sqlite", embed.signature) as store:
        stats = ingest_folder(docs, store, embed, max_words=30, overlap=3)
        assert stats == {"documents": 2, "chunks": 2}
        result = answer("How are API secrets approved?", store, embed, top_k=1)
        assert result["citations"] == ["security.txt#chunk-0"]
        assert "human approval" in result["answer"]
        dataset = tmp_path / "eval.jsonl"
        dataset.write_text('{"question":"How are API secrets approved?","expected_source":"security.txt"}\n')
        metrics = evaluate(dataset, store, embed, top_k=1)
        assert metrics["recall_at_k"] == 1.0
        (docs / "security.txt").unlink()
        assert ingest_folder(docs, store, embed)["chunks"] == 1
        assert all(x["source"] != "security.txt" for x in retrieve("secrets", store, embed))


def test_empty_index_and_query(tmp_path):
    embed = HashingEmbedder()
    with SQLiteStore(tmp_path / "db.sqlite", embed.signature) as store:
        assert answer("", store, embed)["citations"] == []
        assert answer("unrelated galaxy nebula", store, embed)["citations"] == []
        assert answer("anything", store, embed)["answer"] == "No supporting evidence found."
