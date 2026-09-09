"""Tests for XML input parser."""

from pathlib import Path

import pytest

from rag_adviser.input_modes.xml_input import XmlInputParser
from rag_adviser.models import (
    InvalidInputError,
    LatencyBudget,
    PrivacyLevel,
    UseCase,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestXmlInputParser:
    def test_parse_sample_xml(self):
        parser = XmlInputParser(FIXTURES / "sample_input.xml")
        answers = parser.parse()

        assert answers.use_case == UseCase.QA
        assert answers.constraints.privacy == PrivacyLevel.STRICT
        assert answers.constraints.latency_budget == LatencyBudget.MODERATE
        assert answers.constraints.ram_gb == 16.0
        assert answers.future_languages is True
        assert answers.embedding_provider == "huggingface"
        assert answers.llm_provider == "none"
        assert "langchain" in answers.preferred_libraries
        assert "chromadb" in answers.preferred_libraries

    def test_nonexistent_file(self):
        with pytest.raises(InvalidInputError, match="not found"):
            XmlInputParser(Path("/nonexistent.xml"))

    def test_invalid_xml(self, tmp_path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("not valid xml <<>>", encoding="utf-8")
        parser = XmlInputParser(bad_xml)
        with pytest.raises(InvalidInputError, match="Failed to parse"):
            parser.parse()

    def test_wrong_root_element(self, tmp_path):
        bad_root = tmp_path / "bad_root.xml"
        bad_root.write_text("<wrongroot></wrongroot>", encoding="utf-8")
        parser = XmlInputParser(bad_root)
        with pytest.raises(InvalidInputError, match="Expected root element"):
            parser.parse()

    def test_invalid_enum_value(self, tmp_path):
        bad_enum = tmp_path / "bad_enum.xml"
        bad_enum.write_text(
            '<?xml version="1.0"?><ragadvisor>'
            "<use_case_constraints><use_case>invalid_value</use_case></use_case_constraints>"
            "</ragadvisor>",
            encoding="utf-8",
        )
        parser = XmlInputParser(bad_enum)
        with pytest.raises(InvalidInputError, match="Invalid use_case"):
            parser.parse()

    def test_minimal_xml(self, tmp_path):
        minimal = tmp_path / "minimal.xml"
        minimal.write_text(
            '<?xml version="1.0"?><ragadvisor></ragadvisor>',
            encoding="utf-8",
        )
        parser = XmlInputParser(minimal)
        answers = parser.parse()
        # Should use defaults
        assert answers.use_case == UseCase.QA

    def test_partial_xml(self, tmp_path):
        partial = tmp_path / "partial.xml"
        partial.write_text(
            '<?xml version="1.0"?><ragadvisor>'
            "<use_case_constraints>"
            "<use_case>semantic_search</use_case>"
            "<privacy>strict</privacy>"
            "</use_case_constraints>"
            "</ragadvisor>",
            encoding="utf-8",
        )
        parser = XmlInputParser(partial)
        answers = parser.parse()
        assert answers.use_case == UseCase.SEARCH
        assert answers.constraints.privacy == PrivacyLevel.STRICT
