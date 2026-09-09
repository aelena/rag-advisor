"""Tests for the query transformation recommender."""

from rag_adviser.models import (
    AnswerType,
    HardwareConstraints,
    LatencyBudget,
    QueryComplexity,
    QueryType,
    UserAnswers,
)
from rag_adviser.recommenders.query_recommender import QueryRecommender


class TestQueryRecommender:
    """Test query transformation recommendations."""

    def setup_method(self) -> None:
        self.recommender = QueryRecommender()

    def test_simple_factual_no_transforms(self) -> None:
        """Simple factual queries with natural language need minimal transforms."""
        answers = UserAnswers()
        answers.query_type = QueryType.NATURAL_QUESTIONS
        answers.query_complexity = QueryComplexity.SIMPLE_FACTUAL
        answers.expected_answer_type = AnswerType.SYNTHESIZED
        rec = self.recommender.recommend(answers)
        # Should have no required techniques
        required = [t for t in rec.techniques if t["priority"] == "required"]
        assert len(required) == 0

    def test_multi_turn_requires_condensation(self) -> None:
        """Multi-turn queries must have conversation condensation."""
        answers = UserAnswers()
        answers.query_type = QueryType.MULTI_TURN
        answers.query_complexity = QueryComplexity.SIMPLE_FACTUAL
        rec = self.recommender.recommend(answers)
        names = [t["name"] for t in rec.techniques]
        assert "Conversation Condensation" in names
        # It should be required
        condensation = next(t for t in rec.techniques if t["name"] == "Conversation Condensation")
        assert condensation["priority"] == "required"
        assert rec.requires_llm is True

    def test_short_keywords_query_expansion(self) -> None:
        """Short keyword queries should recommend query expansion."""
        answers = UserAnswers()
        answers.query_type = QueryType.SHORT_KEYWORDS
        answers.query_complexity = QueryComplexity.SIMPLE_FACTUAL
        rec = self.recommender.recommend(answers)
        names = [t["name"] for t in rec.techniques]
        assert "Query Expansion" in names

    def test_multi_hop_hyde_and_decomposition(self) -> None:
        """Multi-hop queries should get HyDE and multi-query decomposition."""
        answers = UserAnswers()
        answers.query_type = QueryType.NATURAL_QUESTIONS
        answers.query_complexity = QueryComplexity.MULTI_HOP
        rec = self.recommender.recommend(answers)
        names = [t["name"] for t in rec.techniques]
        assert "HyDE (Hypothetical Document Embeddings)" in names
        assert "Multi-Query Decomposition" in names
        assert rec.requires_llm is True
        assert rec.latency_impact_ms > 0

    def test_comparative_multi_query(self) -> None:
        """Comparative queries should get multi-query retrieval."""
        answers = UserAnswers()
        answers.query_type = QueryType.NATURAL_QUESTIONS
        answers.query_complexity = QueryComplexity.COMPARATIVE
        rec = self.recommender.recommend(answers)
        names = [t["name"] for t in rec.techniques]
        assert "Multi-Query Retrieval" in names

    def test_aggregative_broad_retrieval(self) -> None:
        """Aggregative queries should get broad retrieval with reranking."""
        answers = UserAnswers()
        answers.query_type = QueryType.NATURAL_QUESTIONS
        answers.query_complexity = QueryComplexity.AGGREGATIVE
        rec = self.recommender.recommend(answers)
        names = [t["name"] for t in rec.techniques]
        assert "Broad Retrieval with Reranking" in names

    def test_fast_latency_warning(self) -> None:
        """High-latency techniques with fast budget should generate warning."""
        answers = UserAnswers()
        answers.query_type = QueryType.MULTI_TURN
        answers.query_complexity = QueryComplexity.MULTI_HOP
        answers.constraints = HardwareConstraints(latency_budget=LatencyBudget.FAST)
        rec = self.recommender.recommend(answers)
        assert any("latency budget" in n.lower() or "WARNING" in n for n in rec.notes)

    def test_code_snippet_generated(self) -> None:
        """Recommendations with techniques should include code snippets."""
        answers = UserAnswers()
        answers.query_type = QueryType.MULTI_TURN
        answers.query_complexity = QueryComplexity.SIMPLE_FACTUAL
        rec = self.recommender.recommend(answers)
        assert rec.code_snippet != ""
        assert "langchain" in rec.code_snippet.lower()

    def test_exact_passage_optional_hyde(self) -> None:
        """Exact passage with natural questions should get optional HyDE."""
        answers = UserAnswers()
        answers.query_type = QueryType.NATURAL_QUESTIONS
        answers.query_complexity = QueryComplexity.SIMPLE_FACTUAL
        answers.expected_answer_type = AnswerType.EXACT_PASSAGE
        rec = self.recommender.recommend(answers)
        names = [t["name"] for t in rec.techniques]
        assert "HyDE (Hypothetical Document Embeddings)" in names
        hyde = next(t for t in rec.techniques if t["name"].startswith("HyDE"))
        assert hyde["priority"] == "optional"

    def test_no_techniques_no_snippet(self) -> None:
        """When no techniques are recommended, no code snippet."""
        answers = UserAnswers()
        answers.query_type = QueryType.NATURAL_QUESTIONS
        answers.query_complexity = QueryComplexity.SIMPLE_FACTUAL
        answers.expected_answer_type = AnswerType.SYNTHESIZED
        rec = self.recommender.recommend(answers)
        # Synthesized + simple + natural = no techniques
        assert rec.code_snippet == ""
