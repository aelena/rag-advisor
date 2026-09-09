"""Tests for ChunkingRecommender."""

from rag_adviser.models import ContentType, UseCase
from rag_adviser.recommenders.chunking_recommender import ChunkingRecommender


class TestChunkingRecommender:
    def test_default_prose(self):
        rec = ChunkingRecommender()
        result = rec.recommend(content_type=ContentType.PROSE, languages=["en"])
        assert result.strategy == "recursive"
        assert result.chunk_size == 512
        assert result.chunk_overlap == 50
        assert result.count_by == "tokens"

    def test_code_content_type(self):
        rec = ChunkingRecommender()
        result = rec.recommend(content_type=ContentType.CODE, languages=["en"])
        assert result.strategy == "language_aware"
        assert result.chunk_size == 256

    def test_legal_content_type(self):
        rec = ChunkingRecommender()
        result = rec.recommend(
            content_type=ContentType.LEGAL,
            languages=["en"],
            embedding_max_tokens=2048,  # Legal needs a large context window
        )
        assert result.strategy == "hierarchical"
        assert result.chunk_size == 1024
        assert result.chunk_overlap == 100

    def test_cjk_language_override(self):
        rec = ChunkingRecommender()
        result = rec.recommend(content_type=ContentType.PROSE, languages=["en", "zh"])
        assert "zh" in result.language_overrides
        override = result.language_overrides["zh"]
        assert override["count_by"] == "characters"
        assert override["tokenizer"] == "jieba"
        assert override["chunk_size"] < 512  # Reduced for CJK

    def test_japanese_override(self):
        rec = ChunkingRecommender()
        result = rec.recommend(content_type=ContentType.PROSE, languages=["ja"])
        assert "ja" in result.language_overrides
        assert result.language_overrides["ja"]["tokenizer"] == "sudachi"

    def test_embedding_limit_adjustment(self):
        rec = ChunkingRecommender()
        # If embedding model only supports 256 tokens
        result = rec.recommend(
            content_type=ContentType.PROSE,
            languages=["en"],
            embedding_max_tokens=256,
        )
        assert result.chunk_size <= 256

    def test_qa_use_case_adjustment(self):
        rec = ChunkingRecommender()
        result = rec.recommend(
            content_type=ContentType.PROSE,
            languages=["en"],
            use_case=UseCase.QA,
        )
        assert result.chunk_size <= 512

    def test_summarization_use_case(self):
        rec = ChunkingRecommender()
        result = rec.recommend(
            content_type=ContentType.PROSE,
            languages=["en"],
            use_case=UseCase.SUMMARIZATION,
        )
        assert result.chunk_size > 512  # Should be larger

    def test_code_snippet_generated(self):
        rec = ChunkingRecommender()
        result = rec.recommend(content_type=ContentType.PROSE, languages=["en"])
        assert result.code_snippet
        assert "RecursiveCharacterTextSplitter" in result.code_snippet

    def test_spacy_override_for_spanish(self):
        rec = ChunkingRecommender()
        result = rec.recommend(content_type=ContentType.PROSE, languages=["es"])
        assert "es" in result.language_overrides
        assert result.language_overrides["es"]["sentence_model"] == "es_core_news_sm"
