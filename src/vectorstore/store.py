from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..ingestion.models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY,
    text        TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,
    page        INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    indexed_at  TEXT    NOT NULL
);
"""


class ChunkStore:
    """SQLite-backed store for chunk text and metadata.

    chunk_id values are assigned sequentially starting from the current
    row count, so they always match the FAISS internal vector positions
    when both systems receive data in the same order.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        # WAL mode allows reads while a write is in progress
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> list[int]:
        """Insert chunks and return their assigned chunk_ids.

        IDs are assigned sequentially from the current row count so they
        stay aligned with FAISS vector positions.
        """
        if not chunks:
            return []

        start_id = self.count()
        now = datetime.now(timezone.utc).isoformat()

        rows = [
            (
                start_id + i,
                chunk.text,
                chunk.source,
                chunk.file_type,
                chunk.page,
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                now,
            )
            for i, chunk in enumerate(chunks)
        ]

        self._conn.executemany(
            """
            INSERT INTO chunks
                (chunk_id, text, source, file_type, page,
                 chunk_index, start_char, end_char, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

        return [start_id + i for i in range(len(chunks))]

    def clear(self) -> None:
        """Remove all rows. Call before re-indexing so IDs reset to 0."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, chunk_ids: list[int]) -> list[Chunk]:
        """Return chunks for the given IDs in the order requested.

        IDs not found in the store are silently skipped. Preserving the
        requested order is important because FAISS returns IDs ranked by
        similarity score — we must not re-sort them.
        """
        if not chunk_ids:
            return []

        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

        by_id = {row["chunk_id"]: row for row in rows}

        result: list[Chunk] = []
        for cid in chunk_ids:
            if cid not in by_id:
                continue
            row = by_id[cid]
            result.append(
                Chunk(
                    text=row["text"],
                    source=row["source"],
                    file_type=row["file_type"],
                    page=row["page"],
                    chunk_index=row["chunk_index"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    metadata={"indexed_at": row["indexed_at"]},
                )
            )

        return result

    def get_all(self) -> list[Chunk]:
        """Return every stored chunk ordered by chunk_id."""
        rows = self._conn.execute(
            "SELECT * FROM chunks ORDER BY chunk_id"
        ).fetchall()
        return [
            Chunk(
                text=row["text"],
                source=row["source"],
                file_type=row["file_type"],
                page=row["page"],
                chunk_index=row["chunk_index"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                metadata={"indexed_at": row["indexed_at"]},
            )
            for row in rows
        ]

    def count(self) -> int:
        """Return the total number of stored chunks."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ChunkStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
