"""Tests for DocumentAnalyzer."""

from pathlib import Path

import pytest

from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.models import ContentType, DocumentAnalysisError

FIXTURES = Path(__file__).parent / "fixtures" / "sample_docs"


class TestDocumentAnalyzer:
    def test_analyze_sample_dir(self):
        analyzer = DocumentAnalyzer()
        stats = analyzer.analyze(FIXTURES)

        assert stats.total_files >= 1
        assert stats.total_size_bytes > 0
        assert ".txt" in stats.file_types
        assert stats.primary_language == "en"
        assert stats.avg_tokens_per_doc > 0

    def test_analyze_nonexistent_path(self):
        analyzer = DocumentAnalyzer()
        with pytest.raises(DocumentAnalysisError, match="does not exist"):
            analyzer.analyze(Path("/nonexistent/path"))

    def test_analyze_empty_dir(self, tmp_path):
        analyzer = DocumentAnalyzer()
        with pytest.raises(DocumentAnalysisError, match="No supported documents"):
            analyzer.analyze(tmp_path)

    def test_analyze_single_file(self):
        analyzer = DocumentAnalyzer()
        sample_file = FIXTURES / "sample_en.txt"
        if sample_file.exists():
            stats = analyzer.analyze(sample_file)
            assert stats.total_files == 1
            assert stats.primary_language == "en"

    def test_content_type_detection_prose(self):
        analyzer = DocumentAnalyzer()
        stats = analyzer.analyze(FIXTURES)
        # Our sample is prose about AI
        assert stats.detected_content_type in (ContentType.PROSE, ContentType.MIXED)

    def test_cjk_detection_english(self):
        analyzer = DocumentAnalyzer()
        assert analyzer._check_cjk("Hello world") is False

    def test_cjk_detection_chinese(self):
        analyzer = DocumentAnalyzer()
        assert analyzer._check_cjk("Hello world. This is a test.") is False

    def test_language_detection(self):
        analyzer = DocumentAnalyzer()
        texts = ["This is a sample English text about technology and science."]
        langs = analyzer._detect_languages(texts)
        assert "en" in langs
