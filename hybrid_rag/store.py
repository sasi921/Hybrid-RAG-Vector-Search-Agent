"""Persistent SQLite chunk store with FTS5 BM25 and exact vector scan."""
from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match")
    denom = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denom if denom else 0.0


class SQLiteStore:
    def __init__(self, db_path: str | Path, embedder_signature: str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                body TEXT NOT NULL,
                vector TEXT NOT NULL,
                UNIQUE(source, ordinal)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                body, content='chunks', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, body) VALUES (new.id, new.body);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES ('delete', old.id, old.body);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES ('delete', old.id, old.body);
                INSERT INTO chunks_fts(rowid, body) VALUES (new.id, new.body);
            END;
        """)
        row = self.conn.execute("SELECT value FROM metadata WHERE key='embedder'").fetchone()
        if row and row["value"] != embedder_signature:
            self.close()
            raise ValueError(
                f"Index uses {row['value']}; requested {embedder_signature}. "
                "Use a different --db path or rebuild the index."
            )
        self.conn.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('embedder',?)",
                          (embedder_signature,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def replace_source(self, source: str, chunks: list[str], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE source=?", (source,))
            self.conn.executemany(
                "INSERT INTO chunks(source,ordinal,body,vector) VALUES(?,?,?,?)",
                [(source, i, text, json.dumps(vector)) for i, (text, vector) in enumerate(zip(chunks, vectors))]
            )

    def remove_missing_sources(self, present: set[str]) -> None:
        existing = {r[0] for r in self.conn.execute("SELECT DISTINCT source FROM chunks")}
        with self.conn:
            for source in existing - present:
                self.conn.execute("DELETE FROM chunks WHERE source=?", (source,))

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def lexical_search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not tokens:
            return []
        # Quoting tokens prevents MATCH syntax injection; OR is intentional.
        match = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
        rows = self.conn.execute(
            "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY score ASC LIMIT ?",
            (match, limit),
        )
        return [(int(row[0]), float(row[1])) for row in rows]

    def dense_search(self, vector: list[float], limit: int = 20) -> list[tuple[int, float]]:
        rows = self.conn.execute("SELECT id, vector FROM chunks")
        scored = [(int(r[0]), cosine(vector, json.loads(r[1]))) for r in rows]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, source, ordinal, body FROM chunks WHERE id IN ({placeholders})", ids
        )
        return {int(r["id"]): dict(r) for r in rows}
