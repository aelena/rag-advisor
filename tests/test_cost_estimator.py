"""Tests for cost, footprint and latency estimates."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    BudgetTier,
    ChunkingRecommendation,
    DocumentStats,
    EmbeddingModelRecommendation,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    QueryTransformationRecommendation,
    Recommendations,
    ReportFormat,
    RerankerRecommendation,
    RetrievalRecommendation,
    UpdateFrequency,
    UseCase,
    UserAnswers,
    VectorDBRecommendation,
)
from rag_adviser.recommenders.cost_estimator import CostEstimator


def _stats(tokens: int = 1_000_000, files: int = 1000, sampled: int = 50) -> DocumentStats:
    return DocumentStats(total_files=files, total_tokens=tokens, sampled_files=sampled)


def _recs(api: bool = False, hybrid: bool = False, rerank_ms: int = 0, qt_ms: int = 0,
          db_category: str = "embedded") -> Recommendations:
    model = EmbeddingModelRecommendation(
        model_id="openai/text-embedding-3-small" if api else "BAAI/bge-base-en-v1.5",
        provider="openai" if api else "huggingface",
        dimension=1536 if api else 768,
        estimated_size_gb=0 if api else 0.44,
        price_per_million_tokens=0.02 if api else None,
    )
    return Recommendations(
        embedding_models=[model],
        chunking=ChunkingRecommendation(chunk_size=512, chunk_overlap=50),
        vector_db=VectorDBRecommendation(category=db_category),
        retrieval=RetrievalRecommendation(top_k=5, hybrid_search=hybrid),
        reranker=RerankerRecommendation(enabled=rerank_ms > 0, estimated_latency_ms=rerank_ms),
        query_transformation=QueryTransformationRecommendation(latency_impact_ms=qt_ms),
    )


class TestEstimates:
    def test_chunk_count_and_footprint(self) -> None:
        answers = UserAnswers(document_stats=_stats())
        e = CostEstimator().estimate(answers, _recs())
        # 1M tokens / (512 - 50) stride
        assert e.chunk_count == 1_000_000 // 462
        assert e.tokens_to_embed == e.chunk_count * 512
        assert e.corpus_tokens_estimated is True
        # 768 dims x 4 bytes x 1.5 overhead ~= 4.6 KB per chunk -> ~10 MB
        assert 8 < e.index_memory_mb < 12
        assert e.index_size_mb > e.index_memory_mb
        assert e.indexing_cost_usd == 0.0
        assert e.indexing_time_min > 0

    def test_api_model_costs_money(self) -> None:
        answers = UserAnswers(
            document_stats=_stats(),
            update_frequency=UpdateFrequency.DAILY,
            expected_queries_per_day=10_000,
        )
        e = CostEstimator().estimate(answers, _recs(api=True))
        assert e.embedding_is_api is True
        # ~1.1M tokens at $0.02/1M
        assert 0.02 <= e.indexing_cost_usd <= 0.03
        assert e.monthly_reindex_cost_usd > 0
        assert e.monthly_query_cost_usd > 0
        assert e.query_latency_breakdown_ms["query_embedding"] == 120

    def test_latency_budget_warning(self) -> None:
        answers = UserAnswers(
            document_stats=_stats(),
            constraints=HardwareConstraints(latency_budget=LatencyBudget.FAST),
        )
        e = CostEstimator().estimate(
            answers, _recs(hybrid=True, rerank_ms=900, qt_ms=600, db_category="client-server")
        )
        assert e.fits_latency_budget is False
        assert e.warnings and "rerank" in e.warnings[0]
        assert set(e.query_latency_breakdown_ms) == {
            "query_embedding", "vector_search", "bm25_and_fusion", "rerank",
            "query_transformation_llm",
        }

    def test_gpu_is_faster_than_cpu(self) -> None:
        cpu = CostEstimator().estimate(UserAnswers(document_stats=_stats()), _recs())
        gpu = CostEstimator().estimate(
            UserAnswers(document_stats=_stats(),
                        constraints=HardwareConstraints(hardware=HardwareProfile.GPU_AVAILABLE)),
            _recs(),
        )
        assert gpu.indexing_time_min < cpu.indexing_time_min
        assert gpu.query_latency_ms < cpu.query_latency_ms

    def test_no_corpus_gives_note(self) -> None:
        e = CostEstimator().estimate(UserAnswers(), _recs())
        assert e.chunk_count == 0
        assert any("No corpus" in n for n in e.notes)
        assert e.query_latency_ms > 0  # latency still estimable


class TestEndToEnd:
    def test_estimates_in_reports(self, tmp_path: Path) -> None:
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Retrieval augmented generation. " * 200, "utf-8")
        answers = UserAnswers(
            document_path=corpus,
            use_case=UseCase.QA,
            expected_queries_per_day=500,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT, budget=BudgetTier.FREE),
        )
        out = tmp_path / "out"
        recs = RAGAdviser().run(answers, out, [ReportFormat.ALL])

        assert recs.estimates is not None and recs.estimates.chunk_count >= 1
        config = yaml.safe_load((out / "rag_config.yaml").read_text("utf-8"))
        assert config["estimates"]["queries_per_day"] == 500
        assert "query_latency_ms" in config["estimates"]
        md = (out / "rag_report.md").read_text("utf-8")
        assert "## Cost, Footprint & Latency Estimates" in md
        assert "Latency Estimates" in (out / "rag_report.html").read_text("utf-8")
