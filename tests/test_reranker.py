"""Tests for the reranker recommender."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    AnswerType,
    BudgetTier,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    QueryComplexity,
    ReportFormat,
    RetrievalRecommendation,
    UseCase,
    UserAnswers,
)
from rag_adviser.recommenders.reranker_recommender import RerankerRecommender


def _answers(**kw) -> UserAnswers:
    constraints = kw.pop("constraints", HardwareConstraints())
    return UserAnswers(constraints=constraints, **kw)


class TestDecision:
    def test_simple_factual_no_rerank_but_suggestion(self) -> None:
        answers = _answers(
            query_complexity=QueryComplexity.SIMPLE_FACTUAL,
            expected_answer_type=AnswerType.SYNTHESIZED,
        )
        rec = RerankerRecommender().recommend(answers, RetrievalRecommendation(top_k=5))
        assert rec.enabled is False
        assert rec.model_id  # still tells the user what to reach for
        assert any("later" in n for n in rec.notes)
        assert rec.code_snippet == ""

    def test_batch_budget_picks_quality(self) -> None:
        answers = _answers(constraints=HardwareConstraints(latency_budget=LatencyBudget.BATCH))
        rec = RerankerRecommender().recommend(answers, RetrievalRecommendation(top_k=5))
        assert rec.enabled is True
        assert rec.quality_score >= 64
        assert rec.provider == "huggingface"
        compile(rec.code_snippet, "rerank", "exec")

    def test_fast_cpu_picks_minilm(self) -> None:
        answers = _answers(
            expected_answer_type=AnswerType.EXACT_PASSAGE,
            constraints=HardwareConstraints(
                latency_budget=LatencyBudget.FAST, hardware=HardwareProfile.CPU_ONLY
            ),
        )
        rec = RerankerRecommender().recommend(answers, RetrievalRecommendation(top_k=5))
        assert rec.enabled is True
        assert rec.model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert rec.estimated_latency_ms <= 150
        assert rec.fetch_k <= 12  # short candidate list under a <500ms budget

    def test_fast_multilingual_cpu_cannot_fit(self) -> None:
        answers = _answers(
            expected_answer_type=AnswerType.EXACT_PASSAGE,
            constraints=HardwareConstraints(
                latency_budget=LatencyBudget.FAST,
                hardware=HardwareProfile.CPU_ONLY,
                privacy=PrivacyLevel.STRICT,
            ),
        )
        rec = RerankerRecommender().recommend(
            answers, RetrievalRecommendation(top_k=5), multilingual=True
        )
        # The only fast-enough models are English-only; they are heavily
        # penalised but still "fit" the latency allowance, so reranking is
        # enabled with a loud warning about language coverage.
        assert rec.enabled is True
        assert any("multilingual" in w.lower() for w in rec.warnings)

    def test_multilingual_moderate_picks_multilingual_model(self) -> None:
        answers = _answers(
            query_complexity=QueryComplexity.MULTI_HOP,
            constraints=HardwareConstraints(
                latency_budget=LatencyBudget.MODERATE, hardware=HardwareProfile.GPU_AVAILABLE,
                vram_gb=16, ram_gb=32,
            ),
        )
        rec = RerankerRecommender().recommend(
            answers, RetrievalRecommendation(top_k=8), multilingual=True
        )
        assert rec.enabled is True
        assert rec.multilingual is True

    def test_api_reranker_when_paid_and_privacy_allows(self) -> None:
        answers = _answers(
            query_complexity=QueryComplexity.MULTI_HOP,
            constraints=HardwareConstraints(
                budget=BudgetTier.PAID_API, privacy=PrivacyLevel.NONE,
                latency_budget=LatencyBudget.MODERATE,
            ),
        )
        rec = RerankerRecommender().recommend(
            answers, RetrievalRecommendation(top_k=8), multilingual=True
        )
        providers = {rec.provider, *(a["provider"] for a in rec.alternatives)}
        assert providers & {"cohere", "voyage"}

    def test_api_excluded_for_strict_privacy(self) -> None:
        answers = _answers(
            query_complexity=QueryComplexity.MULTI_HOP,
            constraints=HardwareConstraints(
                budget=BudgetTier.PAID_API, privacy=PrivacyLevel.STRICT
            ),
        )
        rec = RerankerRecommender().recommend(answers, RetrievalRecommendation(top_k=8))
        assert rec.provider == "huggingface"
        assert all(a["provider"] == "huggingface" for a in rec.alternatives)

    def test_long_chunks_warn_on_short_window(self) -> None:
        answers = _answers(constraints=HardwareConstraints(latency_budget=LatencyBudget.FAST))
        rec = RerankerRecommender().recommend(
            answers, RetrievalRecommendation(top_k=5), chunk_tokens=1024
        )
        # MiniLM (512 tokens) is the only fast pick; it must warn about truncation.
        assert any("truncated" in w for w in rec.warnings)

    def test_hybrid_is_a_signal(self) -> None:
        answers = _answers(expected_answer_type=AnswerType.SYNTHESIZED)
        rec = RerankerRecommender().recommend(
            answers, RetrievalRecommendation(top_k=5, hybrid_search=True)
        )
        assert rec.enabled is True
        assert any("fusion" in r.lower() for r in rec.reasons)


class TestEndToEnd:
    def test_pipeline_syncs_retrieval_and_reports(self, tmp_path: Path) -> None:
        answers = UserAnswers(
            use_case=UseCase.LEGAL,
            query_complexity=QueryComplexity.MULTI_HOP,
            constraints=HardwareConstraints(
                privacy=PrivacyLevel.STRICT, latency_budget=LatencyBudget.MODERATE
            ),
        )
        recs = RAGAdviser().run(answers, tmp_path, [ReportFormat.ALL])

        assert recs.reranker is not None and recs.reranker.enabled
        assert recs.retrieval.rerank is True
        assert recs.retrieval.rerank_model == recs.reranker.model_id

        config = yaml.safe_load((tmp_path / "rag_config.yaml").read_text("utf-8"))
        assert config["reranker"]["enabled"] is True
        assert config["reranker"]["model"] == recs.reranker.model_id
        assert "## Reranking" in (tmp_path / "rag_report.md").read_text("utf-8")
        assert "Reranking" in (tmp_path / "rag_report.html").read_text("utf-8")
        assert any("rerank" in s.lower() for s in recs.implementation_steps)
