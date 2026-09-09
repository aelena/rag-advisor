"""Retrieval evaluation metrics — Recall@k, MRR, NDCG, Precision@k."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Result of a single query evaluation."""

    query: str = ""
    retrieved_texts: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_passages: list[str] = field(default_factory=list)
    hit: bool = False
    reciprocal_rank: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0


@dataclass
class EvalMetrics:
    """Aggregated metrics across all queries for a strategy."""

    strategy_name: str = ""
    num_queries: int = 0
    num_chunks: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0
    mean_ndcg_at_k: float = 0.0
    per_query: list[QueryResult] = field(default_factory=list)


def compute_metrics(
    query_results: list[QueryResult],
    strategy_name: str = "",
    num_chunks: int = 0,
) -> EvalMetrics:
    """Compute aggregated metrics from per-query results."""
    n = len(query_results)
    if n == 0:
        return EvalMetrics(strategy_name=strategy_name)

    return EvalMetrics(
        strategy_name=strategy_name,
        num_queries=n,
        num_chunks=num_chunks,
        hit_rate=sum(1 for q in query_results if q.hit) / n,
        mrr=sum(q.reciprocal_rank for q in query_results) / n,
        mean_precision_at_k=sum(q.precision_at_k for q in query_results) / n,
        mean_recall_at_k=sum(q.recall_at_k for q in query_results) / n,
        mean_ndcg_at_k=sum(q.ndcg_at_k for q in query_results) / n,
        per_query=query_results,
    )


def evaluate_single_query(
    retrieved_texts: list[str],
    retrieved_doc_ids: list[str],
    retrieved_scores: list[float],
    relevant_doc_ids: list[str],
    relevant_passages: list[str],
    query: str = "",
    k: int = 5,
) -> QueryResult:
    """Evaluate a single query's retrieval results.

    Matching logic:
    - If relevant_doc_ids are provided, match by doc_id (source filename in metadata)
    - If relevant_passages are provided, match by substring containment
    - Both can be used simultaneously
    """
    result = QueryResult(
        query=query,
        retrieved_texts=retrieved_texts[:k],
        retrieved_doc_ids=retrieved_doc_ids[:k],
        retrieved_scores=retrieved_scores[:k],
        relevant_doc_ids=relevant_doc_ids,
        relevant_passages=relevant_passages,
    )

    # Build relevance vector: 1 if retrieved item is relevant, 0 otherwise
    relevance = []
    for i in range(min(k, len(retrieved_texts))):
        is_relevant = False

        # Check doc_id match
        if (
            relevant_doc_ids
            and i < len(retrieved_doc_ids)
            and retrieved_doc_ids[i] in relevant_doc_ids
        ):
            is_relevant = True

        # Check passage containment match
        if relevant_passages and i < len(retrieved_texts):
            chunk_text = retrieved_texts[i].lower()
            for passage in relevant_passages:
                if passage.lower() in chunk_text or chunk_text in passage.lower():
                    is_relevant = True
                    break

        relevance.append(1 if is_relevant else 0)

    # Pad if fewer results than k
    while len(relevance) < k:
        relevance.append(0)

    # Hit rate: at least one relevant result in top-k
    result.hit = any(r == 1 for r in relevance)

    # MRR: reciprocal of first relevant rank
    result.reciprocal_rank = 0.0
    for i, r in enumerate(relevance):
        if r == 1:
            result.reciprocal_rank = 1.0 / (i + 1)
            break

    # Precision@k
    result.precision_at_k = sum(relevance[:k]) / k if k > 0 else 0.0

    # Recall@k
    total_relevant = max(len(relevant_doc_ids), len(relevant_passages), 1)
    result.recall_at_k = sum(relevance[:k]) / total_relevant

    # NDCG@k
    result.ndcg_at_k = _ndcg(relevance, k)

    return result


def _ndcg(relevance: list[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    dcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k])
    )
    # Ideal DCG: all relevant items at top
    ideal = sorted(relevance[:k], reverse=True)
    idcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0
