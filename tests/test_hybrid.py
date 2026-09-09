"""Tests for hybrid (BM25 + dense) retrieval recommendations."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    AnswerType,
    ContentType,
    HardwareConstraints,
    PrivacyLevel,
    QueryType,
    ReportFormat,
    RetrievalRecommendation,
    UseCase,
    UserAnswers,
    VectorDBRecommendation,
)
from rag_adviser.recommenders.hybrid_recommender import HybridRecommender
from rag_adviser.recommenders.vector_db_recommender import VectorDBRecommender


class TestAssess:
    def test_short_keywords_alone_trigger_hybrid(self) -> None:
        answers = UserAnswers(query_type=QueryType.SHORT_KEYWORDS)
        want, reasons = HybridRecommender().assess(answers, ContentType.PROSE)
        assert want is True
        assert any("keyword" in r.lower() for r in reasons)

    def test_natural_questions_over_prose_stay_dense(self) -> None:
        answers = UserAnswers(
            query_type=QueryType.NATURAL_QUESTIONS,
            use_case=UseCase.QA,
            expected_answer_type=AnswerType.SYNTHESIZED,
        )
        want, _ = HybridRecommender().assess(answers, ContentType.PROSE)
        assert want is False

    def test_code_content_triggers_hybrid(self) -> None:
        answers = UserAnswers(use_case=UseCase.QA, expected_answer_type=AnswerType.SYNTHESIZED)
        want, reasons = HybridRecommender().assess(answers, ContentType.CODE)
        assert want is True
        assert any("identifiers" in r for r in reasons)

    def test_two_medium_signals_add_up(self) -> None:
        answers = UserAnswers(use_case=UseCase.LEGAL, expected_answer_type=AnswerType.EXACT_PASSAGE)
        want, _ = HybridRecommender().assess(answers, ContentType.PROSE)
        assert want is True

    def test_cjk_adds_segmentation_note(self) -> None:
        answers = UserAnswers(query_type=QueryType.SHORT_KEYWORDS)
        _, reasons = HybridRecommender().assess(answers, ContentType.PROSE, languages=["zh", "en"])
        assert any("segment" in r.lower() for r in reasons)


class TestApply:
    def test_native_backend_flag_and_hint(self) -> None:
        rec = RetrievalRecommendation(top_k=7)
        db = VectorDBRecommendation(provider="Qdrant", supports_hybrid_search=True)
        HybridRecommender().apply(rec, True, ["r"], db)
        assert rec.hybrid_search is True
        assert rec.hybrid_native is True
        assert "Qdrant" in rec.hybrid_code_snippet
        assert "k: int = 7" in rec.hybrid_code_snippet
        compile(rec.hybrid_code_snippet, "hybrid", "exec")

    def test_non_native_backend_suggests_rank_bm25(self) -> None:
        rec = RetrievalRecommendation()
        db = VectorDBRecommendation(provider="FAISS", supports_hybrid_search=False)
        HybridRecommender().apply(rec, True, [], db)
        assert rec.hybrid_native is False
        assert any("rank-bm25" in n for n in rec.notes)

    def test_not_recommended_leaves_note(self) -> None:
        rec = RetrievalRecommendation()
        HybridRecommender().apply(rec, False, [], None)
        assert rec.hybrid_search is False
        assert rec.hybrid_code_snippet == ""
        assert any("Dense-only" in n for n in rec.notes)


class TestVectorDbPreference:
    def test_prefer_hybrid_picks_hybrid_capable_backend(self) -> None:
        rec = VectorDBRecommender().recommend(
            doc_count=100,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
            prefer_hybrid=True,
        )
        assert rec.supports_hybrid_search is True


class TestEndToEnd:
    def test_pipeline_emits_hybrid_everywhere(self, tmp_path: Path) -> None:
        answers = UserAnswers(
            query_type=QueryType.SHORT_KEYWORDS,
            use_case=UseCase.CODE,
            content_type_override=ContentType.CODE,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
        )
        recs = RAGAdviser().run(answers, tmp_path, [ReportFormat.ALL])

        assert recs.retrieval.hybrid_search is True
        config = yaml.safe_load((tmp_path / "rag_config.yaml").read_text("utf-8"))
        assert config["retrieval"]["hybrid"]["enabled"] is True
        md = (tmp_path / "rag_report.md").read_text("utf-8")
        assert "### Hybrid Retrieval" in md
        html = (tmp_path / "rag_report.html").read_text("utf-8")
        assert "Hybrid Retrieval" in html
        if not recs.retrieval.hybrid_native:
            assert any("rank-bm25" in s for s in recs.implementation_steps)
