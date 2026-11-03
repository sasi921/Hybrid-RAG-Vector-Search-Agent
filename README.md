# Hybrid RAG Vector Search Agent

A **standalone, runnable AI retrieval agent** that indexes local documents in SQLite, combines **FTS5/BM25 keyword search** with **vector similarity**, ranks evidence using **reciprocal rank fusion (RRF)**, and returns source-cited passages. Includes a real optional neural embedding backend, tests, a reproducible evaluation dataset, CI, and Docker support.

**No API keys required.** The default hashing embedder is a fast, deterministic **lexical baseline, not a semantic model**. For real semantic embeddings, switch to the optional local `sentence-transformers` backend. The project returns **extractive evidence** rather than claiming an LLM-generated answer; this makes citations auditable and avoids unsupported synthesis.

## Architecture

```text
Markdown / TXT files
        │
        ▼
  Chunk + overlap
        │
        ├── SQLite FTS5 (BM25 keyword index)
        └── SQLite JSON vectors (exact cosine scan)
                  ▲
              Embedder
         hash baseline OR
        sentence-transformers

Question → keyword rank + vector rank → RRF → top-k evidence → citations
```

The vector store is persistent and uses an **exact linear scan**, which is appropriate for a small local corpus. For millions of chunks, replace it with an ANN index (FAISS, Qdrant, pgvector, etc.). This project intentionally does not claim to use an accelerated vector database.

## Quick start (Python 3.10+)

```bash
python -m pip install -r requirements.txt
python app.py ingest
python app.py ask "How does reciprocal rank fusion improve retrieval?"
python app.py evaluate
python app.py stats
python -m pytest -q
```

Run with real neural embeddings (first run may download model weights):

```bash
python -m pip install -r requirements-semantic.txt
python app.py --embedder semantic --db data/semantic.sqlite ingest
python app.py --embedder semantic --db data/semantic.sqlite ask "How do we protect customer secrets?"
python app.py --embedder semantic --db data/semantic.sqlite evaluate
```

**Important:** the index records the embedder signature. You cannot query a hash index with semantic embeddings. Use a separate `--db` for each model, or rebuild.

## Use your own documents

```bash
python app.py --db data/my-docs.sqlite ingest --docs /path/to/your/txt-and-md-files
python app.py --db data/my-docs.sqlite ask "What is the release process?" --top-k 3
```

The ingester recursively reads `.txt` and `.md` files, limits each file to 2 MB, updates changed documents, and removes deleted sources. Only index documents you have permission to process. The SQLite file stays local and is excluded from Git.

## Sample output

```json
{
  "question": "How do we protect customer secrets?",
  "answer": "The security program requires least-privilege credentials... [security.txt#chunk-0]",
  "citations": ["security.txt#chunk-0"],
  "matches": [{"source": "security.txt", "chunk": 0, "citation": "security.txt#chunk-0", "score": 0.032787, "text": "..."}]
}
```

Scores are relative **RRF ranking scores**, not probabilities or calibrated confidence values. Your actual output and score depend on indexed content.

## Evaluation

`examples/eval.jsonl` has question/source pairs. `evaluate` computes **source recall@k**, the fraction of questions where the expected source appears in the top-k retrieved chunks. It does not measure answer factuality or semantic similarity. Add more labeled cases and compare both embedding modes.

## Project layout

```text
app.py                         CLI: ingest / ask / evaluate / stats
hybrid_rag/chunking.py         Deterministic chunking with overlap
hybrid_rag/embeddings.py       Hash baseline + optional semantic embeddings
hybrid_rag/store.py            SQLite FTS5 + persistent vectors
hybrid_rag/agent.py            Hybrid RRF retrieval + extractive cited answers
hybrid_rag/evaluation.py       Source recall@k evaluation
examples/docs/                 Sample knowledge base
examples/eval.jsonl            Evaluation questions
.github/workflows/tests.yml    CI checks
Dockerfile                     Lightweight CLI container
```

## What to learn

Chunking/overlap, BM25, embeddings, persistent vector storage, cosine similarity, reciprocal rank fusion, source attribution, index/model compatibility, offline evaluation, and CI.

## Roadmap

- Add approximate-nearest-neighbor vector indexing for scale
- Add hybrid reranking and contextual chunking
- Add optional grounded LLM synthesis with claim-to-source verification
- Add ingestion provenance, metadata filters, and observability

## Portfolio context

Part of the **AI Agent Engineering Lab**, a collection of independent, recruiter-visible GitHub repositories. This project is separate from the old `100-Days-of-AI-Agents` repository.
