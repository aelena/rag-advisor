"""Knowledge base builder — chunks and indexes the research papers for the mini-RAG."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from rag_adviser.evaluators.chunking_strategies import (
    Chunk,
    chunk_recursive,
    load_documents,
)
from rag_adviser.models import RagAdvisorError

logger = logging.getLogger(__name__)

# Default path to research papers (shipped with the package)
_PAPERS_DIR = Path(__file__).parent / "papers"

# Cache directory for pre-built index
_CACHE_DIR = Path(__file__).parent / ".cache"


class KnowledgeBaseError(RagAdvisorError):
    """Failed to build or query the knowledge base."""


class KnowledgeBase:
    """Manages the research knowledge base — loading, chunking, embedding, indexing.

    Uses FAISS if available (fastest), falls back to a simple in-memory brute-force
    search. No external database required.
    """

    def __init__(self, papers_dir: Path | None = None) -> None:
        self._papers_dir = Path(papers_dir) if papers_dir else _PAPERS_DIR
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
        self._model = None
        self._dimension = 0
        self._index = None  # FAISS index or None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def build(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Build the knowledge base: load papers, chunk, embed, index.

        Caches embeddings to disk so subsequent runs are instant.
        """
        # Load papers
        if not self._papers_dir.exists():
            raise KnowledgeBaseError(
                f"Research papers directory not found: {self._papers_dir}"
            )

        documents = load_documents(self._papers_dir)
        if not documents:
            raise KnowledgeBaseError(
                f"No documents found in {self._papers_dir}"
            )

        # Chunk with markdown-aware splitting
        chunked = chunk_recursive(
            documents, chunk_size=400, chunk_overlap=50
        )
        self._chunks = chunked.chunks

        # Check cache
        cache_key = self._compute_cache_key(documents, model_id)
        cached = self._load_cache(cache_key)

        if cached is not None:
            self._embeddings = cached
            logger.info("Loaded cached embeddings (%d chunks)", len(self._embeddings))
        else:
            # Load embedding model and embed
            self._load_model(model_id)
            self._embeddings = self._embed_all()
            self._save_cache(cache_key, self._embeddings)

        # Build search index
        self._build_index()
        self._ready = True

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search the knowledge base for chunks relevant to the query.

        Returns list of (chunk, score) tuples, sorted by relevance.
        """
        if not self._ready:
            raise KnowledgeBaseError("Knowledge base not built. Call build() first.")

        # Embed query
        if self._model is None:
            self._load_model()
        query_emb = self._model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

        # Search
        if self._index is not None:
            return self._search_faiss(query_emb, top_k)
        else:
            return self._search_brute_force(query_emb, top_k)

    def _load_model(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise KnowledgeBaseError(
                "sentence-transformers is required for the research assistant. "
                "Install: pip install ragadvisor[eval]"
            ) from e
        self._model = SentenceTransformer(model_id)
        test_emb = self._model.encode(["test"], show_progress_bar=False)
        self._dimension = len(test_emb[0])

    def _embed_all(self) -> list[list[float]]:
        """Embed all chunks."""
        texts = [c.text for c in self._chunks]
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]

    def _build_index(self) -> None:
        """Build a FAISS index, or fall back to storing embeddings for brute-force."""
        try:
            import faiss
            import numpy as np

            vectors = np.array(self._embeddings, dtype=np.float32)
            self._dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)
        except ImportError:
            # No FAISS available — will use brute-force search
            self._index = None

    def _search_faiss(self, query_emb: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Search using FAISS index."""
        import numpy as np

        query = np.array([query_emb], dtype=np.float32)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if 0 <= idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))
        return results

    def _search_brute_force(self, query_emb: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Brute-force cosine similarity search (fallback when FAISS unavailable)."""
        scores = []
        for i, emb in enumerate(self._embeddings):
            dot = sum(a * b for a, b in zip(query_emb, emb, strict=False))
            scores.append((i, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            (self._chunks[idx], score)
            for idx, score in scores[:top_k]
        ]

    def _compute_cache_key(self, documents: list[tuple[str, str]], model_id: str) -> str:
        """Compute a hash of documents + model to use as cache key."""
        h = hashlib.sha256()
        h.update(model_id.encode())
        for name, text in sorted(documents):
            h.update(name.encode())
            h.update(text.encode())
        return h.hexdigest()[:16]

    def _load_cache(self, key: str) -> list[list[float]] | None:
        """Try to load cached embeddings."""
        cache_file = _CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) == len(self._chunks):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _save_cache(self, key: str, embeddings: list[list[float]]) -> None:
        """Save embeddings to cache."""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file = _CACHE_DIR / f"{key}.json"
            cache_file.write_text(
                json.dumps(embeddings), encoding="utf-8"
            )
        except (OSError, PermissionError):
            pass  # Cache is best-effort
