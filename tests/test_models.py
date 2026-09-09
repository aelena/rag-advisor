"""Tests for data models."""

from rag_adviser.models import (
    ContentType,
    DocumentStats,
    HardwareConstraints,
    Recommendations,
    UseCase,
    UserAnswers,
)


class TestEnums:
    def test_content_type_values(self):
        assert ContentType.PROSE.value == "prose"
        assert ContentType.CODE.value == "code"
        assert ContentType.LEGAL.value == "legal"

    def test_use_case_values(self):
        assert UseCase.QA.value == "question_answering"
        assert UseCase.SEARCH.value == "semantic_search"

    def test_content_type_from_string(self):
        assert ContentType("prose") == ContentType.PROSE
        assert ContentType("code") == ContentType.CODE


class TestDataclasses:
    def test_document_stats_defaults(self):
        stats = DocumentStats()
        assert stats.total_files == 0
        assert stats.primary_language == "en"
        assert stats.has_cjk is False
        assert stats.detected_content_type == ContentType.PROSE

    def test_user_answers_defaults(self):
        answers = UserAnswers()
        assert answers.use_case == UseCase.QA
        assert answers.future_languages is False
        assert answers.has_ground_truth is False

    def test_hardware_constraints_defaults(self):
        hw = HardwareConstraints()
        assert hw.ram_gb == 16.0
        assert hw.vram_gb == 0.0

    def test_recommendations_empty(self):
        recs = Recommendations()
        assert recs.embedding_models == []
        assert recs.chunking is None
        assert recs.vector_db is None
        assert recs.warnings == []

    def test_embedding_model_fields(self, sample_embedding_model):
        m = sample_embedding_model
        assert m.model_id == "BAAI/bge-small-en-v1.5"
        assert m.dimension == 384
        assert m.score == 0.85

    def test_recommendations_with_data(self, sample_recommendations):
        recs = sample_recommendations
        assert len(recs.embedding_models) == 2
        assert recs.chunking is not None
        assert recs.chunking.strategy == "recursive"
        assert recs.vector_db is not None
        assert recs.vector_db.provider == "ChromaDB"
