"""Hybrid (sparse + dense) retrieval recommender.

Dense embeddings capture meaning but blur exact terms: product codes, function
names, statute numbers, rare proper nouns. A BM25 index run alongside the
vector search and merged with reciprocal rank fusion recovers those cases at
little cost. This module decides when that is worth recommending and how to
implement it for the chosen vector database.
"""

from __future__ import annotations

from rag_adviser.models import (
    AnswerType,
    ContentType,
    LatencyBudget,
    QueryComplexity,
    QueryType,
    RetrievalRecommendation,
    UseCase,
    UserAnswers,
    VectorDBRecommendation,
)

_CJK = {"zh", "ja", "ko"}

# Evidence weights; hybrid is recommended at >= _THRESHOLD.
_STRONG = 2
_MEDIUM = 1
_THRESHOLD = 2

_LEXICAL_CONTENT = {
    ContentType.CODE: "identifiers, function names and error strings",
    ContentType.LEGAL: "statute numbers, clause references and defined terms",
    ContentType.TABULAR: "codes, IDs and column values",
    ContentType.SCIENTIFIC: "citations, gene/compound names and notation",
}

# How each supported vector DB exposes hybrid search natively.
_NATIVE_HINTS = {
    "qdrant": (
        "Qdrant: add a sparse vector (models.SparseVectorParams) to the collection, "
        "index BM25/SPLADE weights, and use query_points with prefetch on both "
        "vectors and models.FusionQuery(fusion=models.Fusion.RRF)."
    ),
    "weaviate": (
        "Weaviate: collection.query.hybrid(query, alpha=0.5, limit=k) blends BM25 "
        "and vector scores; alpha=0 is pure BM25, alpha=1 pure vector."
    ),
    "pgvector": (
        "pgvector: add a tsvector column with a GIN index, rank with ts_rank_cd, "
        "and fuse with the vector <=> ranking via RRF in a single SQL CTE."
    ),
    "lancedb": (
        "LanceDB: table.create_fts_index('text') then "
        "table.search(query, query_type='hybrid').limit(k)."
    ),
    "pinecone": (
        "Pinecone: use a dotproduct index and upsert sparse_values alongside dense "
        "vectors (sparse-dense vectors), or a separate sparse index with RRF."
    ),
    "milvus": (
        "Milvus 2.5+: declare a BM25 Function on a SPARSE_FLOAT_VECTOR field and "
        "call hybrid_search with RRFRanker()."
    ),
}


class HybridRecommender:
    """Decide whether to recommend hybrid retrieval and how to implement it."""

    def assess(
        self,
        answers: UserAnswers,
        content_type: ContentType | None,
        languages: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Return ``(recommend_hybrid, reasons)``."""
        score = 0
        reasons: list[str] = []
        langs = languages or ["en"]

        if answers.query_type == QueryType.SHORT_KEYWORDS:
            score += _STRONG
            reasons.append(
                "Short keyword queries carry little semantic signal; BM25 matches them directly"
            )

        if content_type in _LEXICAL_CONTENT:
            score += _STRONG
            reasons.append(
                f"{content_type.value.title()} content is full of "
                f"{_LEXICAL_CONTENT[content_type]} that dense embeddings blur"
            )

        if answers.use_case in (UseCase.CODE, UseCase.LEGAL, UseCase.SEARCH):
            score += _MEDIUM
            reasons.append(
                f"{answers.use_case.value.replace('_', ' ').title()} rewards exact-term recall"
            )

        if answers.expected_answer_type == AnswerType.EXACT_PASSAGE:
            score += _MEDIUM
            reasons.append("Exact-passage answers benefit from lexical precision")

        if answers.query_complexity == QueryComplexity.AGGREGATIVE:
            score += _MEDIUM
            reasons.append("Aggregative queries need broad recall; two retrievers widen the net")

        recommend = score >= _THRESHOLD
        if not recommend:
            return False, reasons

        if any(lang.lower().split("-")[0] in _CJK for lang in langs):
            reasons.append(
                "CJK text must be word-segmented (jieba / sudachi / mecab) before BM25 indexing"
            )
        if answers.constraints.latency_budget == LatencyBudget.FAST:
            reasons.append(
                "BM25 adds ~10-30ms in-process; run both retrievers in parallel to stay <500ms"
            )
        return True, reasons

    def apply(
        self,
        rec: RetrievalRecommendation,
        recommend: bool,
        reasons: list[str],
        vector_db: VectorDBRecommendation | None,
    ) -> None:
        """Fill the hybrid fields of a retrieval recommendation in place."""
        rec.hybrid_search = recommend
        rec.hybrid_reasons = list(reasons)

        if not recommend:
            rec.notes.append(
                "Dense-only retrieval is sufficient for this profile; add a BM25 hybrid "
                "later if keyword-style queries underperform"
            )
            return

        provider = (vector_db.provider if vector_db else "").lower()
        rec.hybrid_native = bool(vector_db and vector_db.supports_hybrid_search)
        rec.notes.append(
            "Hybrid retrieval: run BM25 and dense search in parallel, merge with "
            "reciprocal rank fusion (RRF), then apply reranking/threshold"
        )
        if rec.hybrid_native:
            rec.notes.append(
                f"{vector_db.provider} supports hybrid search natively; prefer its built-in "
                f"sparse index over a separate BM25 service"
            )
        elif vector_db:
            rec.notes.append(
                f"{vector_db.provider} has no native sparse index: use rank-bm25 in-process "
                f"for small corpora, or Elasticsearch/OpenSearch/Typesense at scale"
            )
        rec.hybrid_code_snippet = self._snippet(rec.top_k, provider if rec.hybrid_native else "")

    @staticmethod
    def _snippet(top_k: int, native_provider: str) -> str:
        lines = [
            "# Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion (RRF)",
            "# pip install rank-bm25",
            "import re",
            "",
            "from rank_bm25 import BM25Okapi",
            "",
            "",
            "def tokenize(text: str) -> list[str]:",
            '    return re.findall(r"\\w+", text.lower())',
            "",
            "",
            "# chunks: list of your chunk objects with .id and .text; vectorstore: dense index",
            "bm25 = BM25Okapi([tokenize(c.text) for c in chunks])",
            "",
            "",
            f"def hybrid_search(query: str, k: int = {top_k}, fetch_k: int = 20, rrf_k: int = 60):",
            "    dense = vectorstore.similarity_search(query, k=fetch_k)",
            "    sparse_scores = bm25.get_scores(tokenize(query))",
            "    order = sorted(range(len(chunks)), key=lambda i: -sparse_scores[i])[:fetch_k]",
            "    sparse = [chunks[i] for i in order]",
            "",
            "    fused: dict[str, float] = {}",
            "    by_id = {}",
            "    for results in (dense, sparse):",
            "        for rank, chunk in enumerate(results):",
            "            by_id[chunk.id] = chunk",
            "            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank + 1)",
            "    ranked = sorted(fused.items(), key=lambda item: -item[1])",
            "    return [by_id[chunk_id] for chunk_id, _ in ranked[:k]]",
        ]
        hint = _NATIVE_HINTS.get(native_provider)
        if hint:
            lines.extend(["", f"# Native alternative -- {hint}"])
        return "\n".join(lines)
