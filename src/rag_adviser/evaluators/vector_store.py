"""Vector store abstraction for evaluation — supports ChromaDB, pgvector, FAISS, sqlite-vec."""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag_adviser.evaluators.chunking_strategies import Chunk
from rag_adviser.models import RagAdvisorError

logger = logging.getLogger(__name__)


class VectorStoreError(RagAdvisorError):
    """Failed to interact with vector store."""


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    text: str = ""
    source_file: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract vector store interface for evaluation."""

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create or reset a collection."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings to the store."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search for similar chunks."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Clean up the collection."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of items in the collection."""


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store for evaluation."""

    def __init__(self) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as e:
            raise VectorStoreError(
                "chromadb is required for evaluation. "
                "Install it: pip install ragadvisor[eval]"
            ) from e
        self._client = chromadb.Client()  # In-memory, ephemeral
        self._collection = None
        self._collection_name = ""

    def create_collection(self, name: str, dimension: int) -> None:
        """Create or reset an in-memory collection."""

        # Delete if exists
        with contextlib.suppress(Exception):
            self._client.delete_collection(name)

        # ChromaDB needs an embedding function, but we provide pre-computed embeddings.
        # Use a dummy function that returns the embeddings we've already computed.
        self._collection = self._client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = name

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with pre-computed embeddings."""
        if not chunks or not self._collection:
            return

        # ChromaDB has a batch size limit, process in batches
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            self._collection.add(
                ids=[f"chunk_{i + j}" for j in range(len(batch_chunks))],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch_chunks],
                metadatas=[
                    {
                        "source": c.source_file,
                        "chunk_index": c.chunk_index,
                        "strategy": c.strategy,
                    }
                    for c in batch_chunks
                ],
            )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search using cosine similarity."""
        if not self._collection:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0], strict=False,
            ):
                # ChromaDB returns distances (lower = more similar for cosine)
                # Convert to similarity score
                score = 1.0 - dist
                search_results.append(SearchResult(
                    text=doc,
                    source_file=meta.get("source", ""),
                    score=score,
                    metadata=meta,
                ))

        return search_results

    def delete_collection(self) -> None:
        """Clean up the collection."""
        if self._collection_name:
            with contextlib.suppress(Exception):
                self._client.delete_collection(self._collection_name)
        self._collection = None

    def count(self) -> int:
        """Return item count."""
        return self._collection.count() if self._collection else 0


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector-backed vector store for evaluation."""

    def __init__(
        self,
        connection_string: str = "postgresql://localhost:5432/ragadvisor_eval",
    ) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as e:
            raise VectorStoreError(
                "psycopg[binary] and pgvector are required for pgvector evaluation. "
                "Install: pip install ragadvisor[eval-pgvector]"
            ) from e

        self._connection_string = connection_string
        self._conn = None
        self._table_name = ""
        self._dimension = 0

    def _connect(self) -> None:
        """Establish database connection."""
        import psycopg

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._connection_string, autocommit=True)

            # Ensure pgvector extension is available
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def create_collection(self, name: str, dimension: int) -> None:
        """Create or reset a pgvector table."""
        self._connect()
        # Sanitize name for SQL (alphanumeric + underscore only)
        safe_name = "eval_" + "".join(c if c.isalnum() else "_" for c in name)
        self._table_name = safe_name
        self._dimension = dimension

        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {safe_name}")
            cur.execute(f"""
                CREATE TABLE {safe_name} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_file TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    strategy TEXT NOT NULL DEFAULT '',
                    embedding vector({dimension}) NOT NULL
                )
            """)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks with embeddings using COPY for performance."""
        if not chunks or not self._table_name:
            return

        self._connect()
        with self._conn.cursor() as cur:
            # Use executemany with parameterized query
            batch_size = 1000
            for i in range(0, len(chunks), batch_size):
                batch = list(
                    zip(chunks[i : i + batch_size], embeddings[i : i + batch_size], strict=False)
                )
                cur.executemany(
                    f"""
                    INSERT INTO {self._table_name}
                        (text, source_file, chunk_index, strategy, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    """,
                    [
                        (c.text, c.source_file, c.chunk_index, c.strategy, str(emb))
                        for c, emb in batch
                    ],
                )

            # Create index after bulk insert (faster than maintaining during insert)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
                ON {self._table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {max(1, len(chunks) // 100)})
            """)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search using cosine distance."""
        if not self._table_name:
            return []

        self._connect()
        emb_str = str(query_embedding)

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT text, source_file, chunk_index, strategy,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self._table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (emb_str, emb_str, top_k),
            )
            rows = cur.fetchall()

        return [
            SearchResult(
                text=row[0],
                source_file=row[1],
                score=float(row[4]),
                metadata={
                    "source": row[1],
                    "chunk_index": row[2],
                    "strategy": row[3],
                },
            )
            for row in rows
        ]

    def delete_collection(self) -> None:
        """Drop the evaluation table."""
        if self._table_name and self._conn and not self._conn.closed:
            with self._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self._table_name}")
        self._table_name = ""

    def count(self) -> int:
        """Return item count."""
        if not self._table_name:
            return 0
        self._connect()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            return cur.fetchone()[0]

    def __del__(self) -> None:
        """Close connection on cleanup."""
        if self._conn and not self._conn.closed:
            self._conn.close()


class FaissVectorStore(BaseVectorStore):
    """FAISS-backed vector store for evaluation — fastest similarity search."""

    def __init__(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise VectorStoreError(
                "faiss-cpu is required for FAISS evaluation. "
                "Install: pip install ragadvisor[eval-faiss]"
            ) from e
        import numpy as np

        self._faiss = faiss
        self._np = np
        self._index = None
        self._chunks: list[Chunk] = []
        self._dimension = 0

    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new FAISS index (IndexFlatIP for cosine similarity on normalized vectors)."""
        self._dimension = dimension
        self._index = self._faiss.IndexFlatIP(dimension)
        self._chunks = []

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with pre-computed embeddings."""
        if not chunks or self._index is None:
            return

        vectors = self._np.array(embeddings, dtype=self._np.float32)
        # Normalize for cosine similarity via inner product
        self._faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search using inner product (cosine similarity on normalized vectors)."""
        if self._index is None or self._index.ntotal == 0:
            return []

        query = self._np.array([query_embedding], dtype=self._np.float32)
        self._faiss.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            results.append(SearchResult(
                text=chunk.text,
                source_file=chunk.source_file,
                score=float(score),
                metadata={
                    "source": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                },
            ))

        return results

    def delete_collection(self) -> None:
        """Reset the index."""
        self._index = None
        self._chunks = []

    def count(self) -> int:
        """Return item count."""
        return self._index.ntotal if self._index else 0


class InMemoryVectorStore(BaseVectorStore):
    """Dependency-free brute-force store (numpy only).

    Exact cosine search over a normalised matrix. Fine for evaluation-sized
    corpora (up to a few hundred thousand chunks); it exists so that
    ``ragadvisor run --validate`` works with nothing but sentence-transformers
    installed. numpy is already a dependency of sentence-transformers.
    """

    def __init__(self) -> None:
        try:
            import numpy as np
        except ImportError as e:  # pragma: no cover - numpy ships with sentence-transformers
            raise VectorStoreError(
                "numpy is required for the in-memory vector store. "
                "Install: pip install ragadvisor[eval]"
            ) from e
        self._np = np
        self._matrix = None  # (n, dim) float32, L2-normalised rows
        self._chunks: list[Chunk] = []
        self._dimension = 0

    def create_collection(self, name: str, dimension: int) -> None:
        self._dimension = dimension
        self._matrix = self._np.empty((0, dimension), dtype=self._np.float32)
        self._chunks = []

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks or self._matrix is None:
            return
        vectors = self._np.asarray(embeddings, dtype=self._np.float32)
        norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = self._np.vstack([self._matrix, vectors / norms])
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        if self._matrix is None or len(self._chunks) == 0:
            return []
        q = self._np.asarray(query_embedding, dtype=self._np.float32)
        norm = self._np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self._matrix @ q
        k = min(top_k, len(self._chunks))
        top = self._np.argpartition(-scores, k - 1)[:k]
        top = top[self._np.argsort(-scores[top])]
        return [
            SearchResult(
                text=self._chunks[i].text,
                source_file=self._chunks[i].source_file,
                score=float(scores[i]),
                metadata={
                    "source": self._chunks[i].source_file,
                    "chunk_index": self._chunks[i].chunk_index,
                    "strategy": self._chunks[i].strategy,
                },
            )
            for i in top
        ]

    def delete_collection(self) -> None:
        self._matrix = None
        self._chunks = []

    def count(self) -> int:
        return len(self._chunks)


class SqliteVecStore(BaseVectorStore):
    """SQLite + sqlite-vec backed vector store for evaluation.

    Single-file, serverless, surprisingly capable for small-to-medium corpora.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        try:
            import sqlite3  # noqa: F401

            import sqlite_vec  # noqa: F401
        except ImportError as e:
            raise VectorStoreError(
                "sqlite-vec is required for SQLite vector evaluation. "
                "Install: pip install ragadvisor[eval-sqlite]"
            ) from e

        import sqlite3 as _sqlite3
        import struct

        import sqlite_vec as _sqlite_vec

        self._sqlite3 = _sqlite3
        self._sqlite_vec = _sqlite_vec
        self._struct = struct
        self._db_path = db_path
        self._conn = None
        self._table_name = ""
        self._dimension = 0

    def _connect(self) -> None:
        """Establish database connection and load sqlite-vec extension."""
        if self._conn is None:
            self._conn = self._sqlite3.connect(self._db_path)
            self._conn.enable_load_extension(True)
            self._sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

    def _serialize_float32(self, vec: list[float]) -> bytes:
        """Serialize a float vector to bytes for sqlite-vec."""
        return self._struct.pack(f"{len(vec)}f", *vec)

    def create_collection(self, name: str, dimension: int) -> None:
        """Create a virtual table for vector search."""
        self._connect()
        safe_name = "eval_" + "".join(c if c.isalnum() else "_" for c in name)
        self._table_name = safe_name
        self._dimension = dimension

        cur = self._conn.cursor()
        # Drop existing tables
        cur.execute(f"DROP TABLE IF EXISTS {safe_name}_vec")
        cur.execute(f"DROP TABLE IF EXISTS {safe_name}_meta")

        # Metadata table (text, source, etc.)
        cur.execute(f"""
            CREATE TABLE {safe_name}_meta (
                rowid INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                source_file TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL DEFAULT 0,
                strategy TEXT NOT NULL DEFAULT ''
            )
        """)

        # Virtual table for vector search
        cur.execute(f"""
            CREATE VIRTUAL TABLE {safe_name}_vec USING vec0(
                embedding float[{dimension}]
            )
        """)
        self._conn.commit()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks with embeddings."""
        if not chunks or not self._table_name:
            return

        self._connect()
        cur = self._conn.cursor()

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=False)):
            rowid = i + 1
            cur.execute(
                f"INSERT INTO {self._table_name}_meta "
                "(rowid, text, source_file, chunk_index, strategy) "
                "VALUES (?, ?, ?, ?, ?)",
                (rowid, chunk.text, chunk.source_file, chunk.chunk_index, chunk.strategy),
            )
            cur.execute(
                f"INSERT INTO {self._table_name}_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, self._serialize_float32(emb)),
            )

        self._conn.commit()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Search using vec0 virtual table."""
        if not self._table_name:
            return []

        self._connect()
        cur = self._conn.cursor()

        cur.execute(
            f"""
            SELECT m.text, m.source_file, m.chunk_index, m.strategy, v.distance
            FROM {self._table_name}_vec v
            JOIN {self._table_name}_meta m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ?
                AND k = ?
            ORDER BY v.distance
            """,
            (self._serialize_float32(query_embedding), top_k),
        )
        rows = cur.fetchall()

        return [
            SearchResult(
                text=row[0],
                source_file=row[1],
                score=1.0 - float(row[4]),  # Convert distance to similarity
                metadata={
                    "source": row[1],
                    "chunk_index": row[2],
                    "strategy": row[3],
                },
            )
            for row in rows
        ]

    def delete_collection(self) -> None:
        """Drop the tables."""
        if self._table_name and self._conn:
            cur = self._conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {self._table_name}_vec")
            cur.execute(f"DROP TABLE IF EXISTS {self._table_name}_meta")
            self._conn.commit()
        self._table_name = ""

    def count(self) -> int:
        """Return item count."""
        if not self._table_name or not self._conn:
            return 0
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self._table_name}_meta")
        return cur.fetchone()[0]

    def __del__(self) -> None:
        """Close connection on cleanup."""
        if self._conn:
            self._conn.close()


def create_vector_store(
    backend: str = "chroma",
    connection_string: str | None = None,
) -> BaseVectorStore:
    """Factory function to create the appropriate vector store.

    Args:
        backend: "chroma", "pgvector", "faiss", "sqlite", or "memory"
        connection_string: PostgreSQL connection string (pgvector) or
                          SQLite DB path (sqlite). Defaults vary by backend.

    Returns:
        A BaseVectorStore instance.
    """
    if backend == "pgvector":
        conn_str = connection_string or "postgresql://localhost:5432/ragadvisor_eval"
        return PgVectorStore(connection_string=conn_str)
    elif backend == "chroma":
        return ChromaVectorStore()
    elif backend == "faiss":
        return FaissVectorStore()
    elif backend == "memory":
        return InMemoryVectorStore()
    elif backend == "sqlite":
        db_path = connection_string or ":memory:"
        return SqliteVecStore(db_path=db_path)
    else:
        raise VectorStoreError(
            f"Unknown vector store backend: '{backend}'. "
            f"Use 'chroma', 'faiss', 'pgvector', 'sqlite', or 'memory'."
        )
