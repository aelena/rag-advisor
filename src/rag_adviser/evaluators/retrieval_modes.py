"""Retrieval modes for the evaluation pipeline: BM25 hybrid fusion and reranking.

The advisor can recommend hybrid (sparse + dense) retrieval and a cross-encoder
reranker. To *measure* those recommendations instead of trusting them, the
evaluation pipeline needs the same stages. Everything here is dependency-free
except the optional cross-encoder, which uses sentence-transformers.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag_adviser.evaluators.chunking_strategies import Chunk
from rag_adviser.evaluators.vector_store import SearchResult

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens. Good enough for BM25 on alphabetic languages."""
    return _TOKEN_RE.findall(text.lower())


def mode_label(hybrid: bool, rerank: bool) -> str:
    """Human-readable retrieval mode name."""
    base = "hybrid" if hybrid else "dense"
    return f"{base}+rerank" if rerank else base


class BM25Index:
    """Minimal Okapi BM25 over chunk texts (no external dependency)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._doc_len.append(len(tokens))
            df.update(counts.keys())
        n = max(len(chunks), 1)
        self._avgdl = (sum(self._doc_len) / n) if chunks else 0.0
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def scores(self, query: str) -> list[float]:
        """BM25 score of every chunk for the query."""
        q_terms = [t for t in tokenize(query) if t in self._idf]
        out: list[float] = []
        for tf, dl in zip(self._tf, self._doc_len, strict=True):
            score = 0.0
            if q_terms and self._avgdl > 0:
                norm = self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
                for term in q_terms:
                    f = tf.get(term, 0)
                    if f:
                        score += self._idf[term] * (f * (self._k1 + 1.0)) / (f + norm)
            out.append(score)
        return out

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Top-k chunks with a positive BM25 score, best first."""
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        results: list[SearchResult] = []
        for i in order[:top_k]:
            if scores[i] <= 0.0:
                break
            c = self._chunks[i]
            results.append(SearchResult(
                text=c.text,
                source_file=c.source_file,
                score=scores[i],
                metadata={"source": c.source_file, "chunk_index": c.chunk_index,
                          "strategy": c.strategy},
            ))
        return results


def _result_key(r: SearchResult) -> tuple:
    return (r.source_file, r.metadata.get("chunk_index"), r.text[:64])


def rrf_fuse(
    ranked_lists: list[list[SearchResult]], top_k: int, rrf_k: int = 60
) -> list[SearchResult]:
    """Reciprocal rank fusion of several ranked result lists."""
    fused: dict[tuple, float] = {}
    keep: dict[tuple, SearchResult] = {}
    for results in ranked_lists:
        for rank, r in enumerate(results):
            key = _result_key(r)
            keep.setdefault(key, r)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    ordered = sorted(fused.items(), key=lambda item: -item[1])
    out: list[SearchResult] = []
    for key, score in ordered[:top_k]:
        r = keep[key]
        out.append(SearchResult(text=r.text, source_file=r.source_file, score=score,
                                metadata=dict(r.metadata)))
    return out


class CrossEncoderReranker:
    """Rerank (query, passage) pairs with a sentence-transformers CrossEncoder."""

    def __init__(self, model_id: str, trust_remote_code: bool = False, max_length: int = 512):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install: pip install ragadvisor[eval]"
            ) from e
        self.model_id = model_id
        self._model = CrossEncoder(model_id, max_length=max_length,
                                   trust_remote_code=trust_remote_code)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        scores = self._model.predict([(query, r.text) for r in results], show_progress_bar=False)
        ranked = sorted(zip(results, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [
            SearchResult(text=r.text, source_file=r.source_file, score=float(s),
                         metadata=dict(r.metadata))
            for r, s in ranked[:top_k]
        ]
