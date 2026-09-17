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
    SizingProfile,
    UpdateFrequency,
    UserAnswers,
    dtype_bytes,
    hnsw_overhead_multiplier,
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

# Vector storage: raw vectors + HNSW graph overhead + payload. The
# overhead used to be a fixed ~50% (fine for fp32 + M=16 in-memory),
# but real deployments vary the datatype, HNSW graph degree, and
# whether vectors live in RAM or on disk. Since 0.5.0 the estimator
# uses the user's ``SizingProfile`` (default ``cpu-balanced``: fp32,
# M=16, in-memory) to compute a workload-specific footprint.
_PAYLOAD_BYTES_PER_CHUNK = 1024

# Text-payload chunk metadata sits on disk regardless of vector mode
# but is only paged into RAM on hits, so we don't count it against
# ``index_memory_mb`` in mmap-on-disk configurations either.

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
        # Use the sizing profile if provided; otherwise conservative
        # defaults (fp32, M=16, in-memory) matching pre-0.5.0 behaviour.
        sizing = answers.sizing_profile or SizingProfile()
        # Effective bytes per stored vector scalar. Scalar/product
        # quantization compresses the raw vectors to ~1 byte per
        # dimension regardless of the source dtype.
        if sizing.quantization in ("scalar", "product"):
            effective_bytes = 1.0
        else:
            effective_bytes = dtype_bytes(sizing.vector_dtype)
        raw_vector_bytes = est.chunk_count * dim * effective_bytes
        hnsw_bytes = raw_vector_bytes * hnsw_overhead_multiplier(sizing.hnsw_m)
        payload_bytes = est.chunk_count * _PAYLOAD_BYTES_PER_CHUNK

        est.index_size_mb = round(
            (raw_vector_bytes + hnsw_bytes + payload_bytes) / 1_048_576, 1
        )
        if sizing.on_disk_vectors:
            # Only the HNSW graph resides in RAM when mmap is on; the
            # raw vectors are paged from disk.
            est.index_memory_mb = round(hnsw_bytes / 1_048_576, 1)
            est.assumptions.append(
                f"Sizing profile '{sizing.name}': mmap on-disk vectors, "
                f"only HNSW graph in RAM"
            )
        else:
            est.index_memory_mb = round(
                (raw_vector_bytes + hnsw_bytes) / 1_048_576, 1
            )
        est.assumptions.append(
            f"Sizing profile '{sizing.name}': {sizing.vector_dtype} vectors, "
            f"HNSW M={sizing.hnsw_m}"
            + (
                f", {sizing.quantization} quantization"
                if sizing.quantization != "none" else ""
            )
        )
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

        # Required query transformations (e.g. multi-turn conversation
        # condensation) must be in the baseline — the pipeline literally
        # can't work without them. Optional / recommended techniques
        # (HyDE for exact-passage retrieval, step-back prompting, …)
        # are reported as separate scenarios so a reader who elects not
        # to enable them isn't quoted a latency figure that assumed they
        # did.
        required_techniques: list[dict] = []
        optional_techniques: list[dict] = []
        if recs.query_transformation and recs.query_transformation.techniques:
            for t in recs.query_transformation.techniques:
                if t.get("priority") == "required":
                    required_techniques.append(t)
                else:
                    optional_techniques.append(t)
        if required_techniques:
            breakdown["query_transformation_required"] = sum(
                int(t.get("latency_ms", 0)) for t in required_techniques
            )
        elif (
            recs.query_transformation
            and recs.query_transformation.latency_impact_ms
            and not recs.query_transformation.techniques
        ):
            # No per-technique metadata to split on — treat the reported
            # latency as opaque and keep it in the baseline (this is the
            # legacy behaviour, preserved so callers that construct a
            # QueryTransformationRecommendation directly still get a
            # populated breakdown).
            breakdown["query_transformation_llm"] = int(
                recs.query_transformation.latency_impact_ms
            )

        # Baseline scenario: what the user runs by default.
        est.query_latency_breakdown_ms = dict(breakdown)
        est.query_latency_ms = sum(breakdown.values())
        baseline_desc_parts = ["Dense retrieval"]
        if recs.retrieval and recs.retrieval.hybrid_search:
            baseline_desc_parts.append("hybrid fusion")
        if recs.reranker and recs.reranker.enabled:
            baseline_desc_parts.append("rerank")
        if required_techniques:
            baseline_desc_parts.append(
                "required transforms ("
                + ", ".join(t.get("name", "unnamed") for t in required_techniques)
                + ")"
            )
        est.query_latency_scenarios = [
            {
                "name": "baseline",
                "description": " + ".join(baseline_desc_parts),
                "total_ms": est.query_latency_ms,
                "breakdown_ms": dict(breakdown),
                "includes_optional": [],
            }
        ]

        if optional_techniques:
            extended = dict(breakdown)
            extended["query_transformation_optional"] = sum(
                int(t.get("latency_ms", 0)) for t in optional_techniques
            )
            opt_names = [t.get("name", "unnamed") for t in optional_techniques]
            est.query_latency_scenarios.append(
                {
                    "name": "with optional query transforms",
                    "description": "Baseline + " + ", ".join(opt_names),
                    "total_ms": sum(extended.values()),
                    "breakdown_ms": extended,
                    "includes_optional": opt_names,
                }
            )
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
        # RAM feasibility: raise a CRITICAL warning when the in-memory
        # index alone consumes most of the host's RAM, before the
        # embedding model, reranker, application and OS. Silent
        # under-provisioning was exactly what the 2026-09-17 review
        # caught on the Books scan (15.4 GB index in 16 GB host, zero
        # warnings).
        ram_mb = float(answers.constraints.ram_gb) * 1024.0
        if ram_mb > 0 and est.index_memory_mb > 0 and not sizing.on_disk_vectors:
            share = est.index_memory_mb / ram_mb
            if share >= 0.7:
                overhead_mb = 2048  # rough allowance for OS + model + app
                headroom_mb = ram_mb - est.index_memory_mb - overhead_mb
                remedy = (
                    "swap to `--sizing-preset gpu-fp16-quantized` or "
                    "`--sizing-preset on-disk-mmap`, or add RAM"
                )
                est.warnings.append(
                    f"CRITICAL: index alone consumes ~{share:.0%} of the "
                    f"{answers.constraints.ram_gb:.0f} GB host RAM "
                    f"({est.index_memory_mb:,.0f} MB of {int(ram_mb):,} MB), "
                    f"leaving ~{max(headroom_mb, 0):,.0f} MB for the "
                    f"embedding model, reranker, application, OS and file "
                    f"cache. This configuration is not operationally viable; "
                    f"{remedy}."
                )
            elif share >= 0.5:
                est.notes.append(
                    f"Index consumes ~{share:.0%} of host RAM "
                    f"({est.index_memory_mb:,.0f} MB of {int(ram_mb):,} MB); "
                    "leave headroom for the embedding model, reranker and OS."
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
