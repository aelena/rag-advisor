"""Cost, footprint and latency estimates for a recommended RAG configuration.

Everything here is an order-of-magnitude estimate assembled from the corpus
statistics, the chosen models and public price/throughput figures. The
numbers are meant to be *compared* (does this fit the latency budget? is the
index 200 MB or 20 GB? is the API bill $3 or $3,000 a month?), not billed.
"""

from __future__ import annotations

from rag_adviser.models import (
    CostEstimate,
    HardwareProfile,
    LatencyBudget,
    Recommendations,
    UpdateFrequency,
    UserAnswers,
)

# ── Throughput / latency priors ────────────────────────────────────────────
# Local embedding throughput (tokens/second) by fp32 checkpoint size on a
# modern multi-core CPU with batching. GPU multiplies this.
_CPU_EMBED_TOKENS_PER_S = [(0.2, 4000.0), (0.7, 1500.0), (1.5, 600.0), (99.0, 300.0)]
_GPU_SPEEDUP_EMBED = 15.0

# Single-query embedding latency (ms) incl. tokenisation, by model size.
_CPU_QUERY_EMBED_MS = [(0.2, 15), (0.7, 40), (1.5, 120), (99.0, 250)]
_GPU_QUERY_EMBED_MS = [(0.2, 8), (0.7, 12), (1.5, 20), (99.0, 30)]
_API_EMBED_MS = 120
_API_EMBED_QUERY_TOKENS = 30  # average tokens per user query

# Vector search latency (ms) by index size (chunks) for embedded / in-memory
# stores with an ANN index; client-server adds a network hop.
_SEARCH_MS = [(50_000, 5), (500_000, 15), (5_000_000, 40), (10**12, 120)]
_CLIENT_SERVER_HOP_MS = 10
_MANAGED_HOP_MS = 40
_BM25_MS = [(50_000, 5), (500_000, 20), (10**12, 60)]

# Vector storage: fp32 dimension x 4 bytes, plus ~50% for HNSW graph and
# metadata/text payload (~1 KB per chunk).
_VECTOR_OVERHEAD = 1.5
_PAYLOAD_BYTES_PER_CHUNK = 1024

_UPDATES_PER_MONTH = {
    UpdateFrequency.NEVER: 0.0,
    UpdateFrequency.WEEKLY: 4.3,
    UpdateFrequency.DAILY: 30.0,
    UpdateFrequency.REALTIME: 30.0,  # continuous; treat as daily re-embedding of changes
}
_UPDATE_CHANGE_FRACTION = 0.05  # assume 5% of the corpus changes per update

_BUDGET_MS = {
    LatencyBudget.FAST: 500,
    LatencyBudget.MODERATE: 2000,
    LatencyBudget.BATCH: 10**9,
}


def _lookup(table: list[tuple[float, float]], key: float) -> float:
    for limit, value in table:
        if key <= limit:
            return value
    return table[-1][1]


class CostEstimator:
    """Turn recommendations plus corpus statistics into rough numbers."""

    def estimate(self, answers: UserAnswers, recs: Recommendations) -> CostEstimate:
        est = CostEstimate()
        stats = answers.document_stats
        hw = answers.constraints.hardware
        gpu = hw == HardwareProfile.GPU_AVAILABLE
        est.queries_per_day = max(int(answers.expected_queries_per_day), 0)

        # ── Corpus and chunks ──────────────────────────────────────────────
        chunk_tokens = recs.chunking.chunk_size if recs.chunking else 512
        overlap = recs.chunking.chunk_overlap if recs.chunking else 50
        stride = max(chunk_tokens - overlap, 1)
        if stats and stats.total_tokens > 0:
            est.corpus_tokens = stats.total_tokens
            est.corpus_tokens_estimated = 0 < stats.sampled_files < stats.total_files
        else:
            est.corpus_tokens = 0
        est.chunk_count = max(1, int(est.corpus_tokens / stride)) if est.corpus_tokens else 0
        est.tokens_to_embed = est.chunk_count * chunk_tokens

        top = recs.embedding_models[0] if recs.embedding_models else None
        dim = top.dimension if top and top.dimension else 768
        est.embedding_model = top.model_id if top else ""
        est.embedding_is_api = bool(top and top.provider != "huggingface")

        # ── Index footprint ────────────────────────────────────────────────
        vector_bytes = est.chunk_count * dim * 4 * _VECTOR_OVERHEAD
        payload_bytes = est.chunk_count * _PAYLOAD_BYTES_PER_CHUNK
        est.index_size_mb = round((vector_bytes + payload_bytes) / 1_048_576, 1)
        est.index_memory_mb = round(vector_bytes / 1_048_576, 1)
        if recs.retrieval and recs.retrieval.hybrid_search:
            est.index_size_mb = round(est.index_size_mb + est.tokens_to_embed * 8 / 1_048_576, 1)
            est.assumptions.append("BM25 index adds ~8 bytes per token of postings")

        # ── One-off indexing cost / time ───────────────────────────────────
        price = float(top.price_per_million_tokens) if top and top.price_per_million_tokens else 0.0
        if est.embedding_is_api:
            est.indexing_cost_usd = round(est.tokens_to_embed / 1e6 * price, 2)
            # ~2 minutes per 1M tokens with batched API calls
            est.indexing_time_min = round(est.tokens_to_embed / 1e6 * 2.0, 1)
            est.assumptions.append(
                f"API embedding at ${price}/1M tokens, ~2 min per 1M tokens with batching"
            )
        else:
            size = top.estimated_size_gb if top else 0.5
            tps = _lookup(_CPU_EMBED_TOKENS_PER_S, size)
            if gpu:
                tps *= _GPU_SPEEDUP_EMBED
            est.indexing_cost_usd = 0.0
            est.indexing_time_min = round(est.tokens_to_embed / tps / 60, 1) if tps else 0.0
            est.assumptions.append(
                f"Local embedding at ~{tps:,.0f} tokens/s on "
                f"{'GPU' if gpu else 'CPU'} for a {size}GB model"
            )

        # ── Recurring re-indexing ──────────────────────────────────────────
        updates = _UPDATES_PER_MONTH.get(answers.update_frequency, 0.0)
        changed_tokens = est.tokens_to_embed * _UPDATE_CHANGE_FRACTION * updates
        est.monthly_reindex_cost_usd = (
            round(changed_tokens / 1e6 * price, 2) if est.embedding_is_api else 0.0
        )
        if updates:
            est.assumptions.append(
                f"{answers.update_frequency.value} updates re-embed ~"
                f"{_UPDATE_CHANGE_FRACTION:.0%} of the corpus each time"
            )

        # ── Per-query latency ──────────────────────────────────────────────
        breakdown: dict[str, int] = {}
        if est.embedding_is_api:
            breakdown["query_embedding"] = _API_EMBED_MS
        else:
            table = _GPU_QUERY_EMBED_MS if gpu else _CPU_QUERY_EMBED_MS
            size_gb = top.estimated_size_gb if top else 0.5
            breakdown["query_embedding"] = int(_lookup(table, size_gb))

        search_ms = _lookup(_SEARCH_MS, est.chunk_count)
        category = recs.vector_db.category if recs.vector_db else "embedded"
        if category == "client-server":
            search_ms += _CLIENT_SERVER_HOP_MS
        elif category == "managed":
            search_ms += _MANAGED_HOP_MS
        breakdown["vector_search"] = int(search_ms)

        if recs.retrieval and recs.retrieval.hybrid_search:
            breakdown["bm25_and_fusion"] = int(_lookup(_BM25_MS, est.chunk_count))
        if recs.reranker and recs.reranker.enabled:
            breakdown["rerank"] = int(recs.reranker.estimated_latency_ms)
        if recs.query_transformation and recs.query_transformation.latency_impact_ms:
            breakdown["query_transformation_llm"] = int(recs.query_transformation.latency_impact_ms)

        est.query_latency_breakdown_ms = breakdown
        est.query_latency_ms = sum(breakdown.values())
        est.latency_budget_ms = _BUDGET_MS[answers.constraints.latency_budget]
        est.fits_latency_budget = est.query_latency_ms <= est.latency_budget_ms
        est.assumptions.append(
            "Per-query latency covers retrieval only; LLM answer generation "
            "(typically 1-10 s) is not included"
        )

        # ── Monthly query-side API cost ────────────────────────────────────
        monthly_queries = est.queries_per_day * 30
        cost = 0.0
        if est.embedding_is_api and price:
            cost += monthly_queries * _API_EMBED_QUERY_TOKENS / 1e6 * price
        rr = recs.reranker
        if rr and rr.enabled and rr.provider != "huggingface":
            # Hosted rerankers bill per search or per token; ~$2 per 1k searches
            # is the common order of magnitude.
            cost += monthly_queries / 1000 * 2.0
            est.assumptions.append("Hosted reranker at ~$2 per 1,000 searches")
        est.monthly_query_cost_usd = round(cost, 2)
        if est.queries_per_day:
            est.assumptions.append(
                f"{est.queries_per_day:,} queries/day (set --queries-per-day to change)"
            )

        # ── Notes / warnings ───────────────────────────────────────────────
        if recs.vector_db and recs.vector_db.category == "managed":
            est.notes.append(
                "Managed vector DB fees (storage + read/write units) are not estimated; "
                f"budget for ~{max(est.index_size_mb / 1024, 0.01):.2f} GB of stored vectors"
            )
        if est.corpus_tokens == 0:
            est.notes.append(
                "No corpus analyzed: chunk count, footprint and indexing cost are unknown"
            )
        if not est.fits_latency_budget:
            over = est.query_latency_ms - est.latency_budget_ms
            heaviest = max(breakdown, key=breakdown.get) if breakdown else ""
            est.warnings.append(
                f"Estimated retrieval latency ~{est.query_latency_ms} ms exceeds the "
                f"{answers.constraints.latency_budget.value} budget by ~{over} ms; "
                f"largest stage: {heaviest.replace('_', ' ')}"
            )
        elif breakdown and est.latency_budget_ms < 10**9 and (
            est.query_latency_ms > 0.7 * est.latency_budget_ms
        ):
            est.notes.append(
                f"Retrieval uses ~{est.query_latency_ms / est.latency_budget_ms:.0%} of the "
                f"latency budget before generation; keep the LLM call fast or stream it"
            )
        return est
