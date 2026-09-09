"""Tests for the approach analyzer."""

from rag_adviser.analyzers.approach_analyzer import ApproachAnalyzer
from rag_adviser.models import (
    ContentType,
    DocumentStats,
    QueryType,
    RecommendedApproach,
    UseCase,
    UserAnswers,
)


class TestApproachAnalyzer:
    """Test the approach analyzer's decision logic."""

    def setup_method(self) -> None:
        self.analyzer = ApproachAnalyzer()

    def test_default_rag_recommendation(self) -> None:
        """Normal scenario should recommend RAG."""
        answers = UserAnswers()
        answers.document_stats = DocumentStats(
            total_files=100,
            total_tokens=500_000,
            detected_content_type=ContentType.PROSE,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.RAG
        assert result.proceed_with_rag is True

    def test_tiny_corpus_direct_context(self) -> None:
        """Very small corpus should recommend direct context stuffing."""
        answers = UserAnswers()
        answers.document_stats = DocumentStats(
            total_files=5,
            total_tokens=10_000,
            detected_content_type=ContentType.PROSE,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.DIRECT_CONTEXT
        assert result.proceed_with_rag is False
        assert result.confidence >= 0.8

    def test_tabular_data_text_to_sql(self) -> None:
        """Tabular data with Q&A should recommend Text-to-SQL."""
        answers = UserAnswers()
        answers.use_case = UseCase.QA
        answers.content_type_override = ContentType.TABULAR
        answers.document_stats = DocumentStats(
            total_files=50,
            total_tokens=200_000,
            detected_content_type=ContentType.TABULAR,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.TEXT_TO_SQL
        assert result.proceed_with_rag is False

    def test_summarization_small_corpus_long_context(self) -> None:
        """Summarization with small corpus should recommend long-context LLM."""
        answers = UserAnswers()
        answers.use_case = UseCase.SUMMARIZATION
        answers.document_stats = DocumentStats(
            total_files=20,
            total_tokens=80_000,
            detected_content_type=ContentType.PROSE,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.LONG_CONTEXT_LLM
        assert result.proceed_with_rag is False

    def test_summarization_large_corpus_still_rag(self) -> None:
        """Large corpus summarization should still recommend RAG."""
        answers = UserAnswers()
        answers.use_case = UseCase.SUMMARIZATION
        answers.document_stats = DocumentStats(
            total_files=500,
            total_tokens=500_000,
            detected_content_type=ContentType.PROSE,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.RAG
        assert result.proceed_with_rag is True

    def test_single_legal_doc_structured_extraction(self) -> None:
        """Single legal document (above tiny threshold) should recommend structured extraction."""
        answers = UserAnswers()
        answers.document_stats = DocumentStats(
            total_files=1,
            total_tokens=60_000,
            detected_content_type=ContentType.LEGAL,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.STRUCTURED_EXTRACTION
        assert result.proceed_with_rag is False

    def test_keyword_search_full_text(self) -> None:
        """Pure keyword search should recommend full-text search."""
        answers = UserAnswers()
        answers.use_case = UseCase.SEARCH
        answers.query_type = QueryType.SHORT_KEYWORDS
        answers.document_stats = DocumentStats(
            total_files=100,
            total_tokens=500_000,
            detected_content_type=ContentType.PROSE,
        )
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.FULL_TEXT_SEARCH
        assert result.proceed_with_rag is False

    def test_no_document_stats_defaults_to_rag(self) -> None:
        """Without document stats, default to RAG."""
        answers = UserAnswers()
        result = self.analyzer.assess(answers)
        assert result.recommended_approach == RecommendedApproach.RAG
        assert result.proceed_with_rag is True

    def test_content_type_override_respected(self) -> None:
        """Content type override should take precedence over detection."""
        answers = UserAnswers()
        answers.content_type_override = ContentType.TABULAR
        answers.use_case = UseCase.QA
        answers.document_stats = DocumentStats(
            total_files=50,
            total_tokens=200_000,
            detected_content_type=ContentType.PROSE,  # detection says prose
        )
        result = self.analyzer.assess(answers)
        # Override says tabular, so Text-to-SQL
        assert result.recommended_approach == RecommendedApproach.TEXT_TO_SQL

    def test_tabular_summarization_still_rag(self) -> None:
        """Tabular data with summarization should still recommend RAG."""
        answers = UserAnswers()
        answers.use_case = UseCase.SUMMARIZATION
        answers.content_type_override = ContentType.TABULAR
        answers.document_stats = DocumentStats(
            total_files=50,
            total_tokens=200_000,
            detected_content_type=ContentType.TABULAR,
        )
        result = self.analyzer.assess(answers)
        # Tabular + summarization doesn't trigger text-to-sql
        assert result.recommended_approach != RecommendedApproach.TEXT_TO_SQL
