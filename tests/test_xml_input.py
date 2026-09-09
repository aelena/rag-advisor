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

    def test_operational_fields(self, tmp_path):
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "a.txt").write_text("hello world " * 20, encoding="utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text('{"query": "q", "relevant_docs": ["a.txt"]}\n', encoding="utf-8")
        xml = tmp_path / "in.xml"
        xml.write_text(f"""<?xml version="1.0"?>
<ragadvisor>
  <document_discovery><document_path>{corpus}</document_path></document_discovery>
  <query_patterns>
    <query_complexity>multi_hop</query_complexity>
    <expected_answer_type>synthesized</expected_answer_type>
    <sample_queries><query>one</query><query> two </query></sample_queries>
    <expected_queries_per_day>2,500</expected_queries_per_day>
    <ground_truth_path>{gt}</ground_truth_path>
    <validate>yes</validate>
    <validate_models>3</validate_models>
  </query_patterns>
</ragadvisor>""", encoding="utf-8")

        answers = XmlInputParser(xml).parse()
        assert answers.query_complexity.value == "multi_hop"
        assert answers.expected_answer_type.value == "synthesized"
        assert answers.sample_queries == ["one", "two"]
        assert answers.expected_queries_per_day == 2500
        assert answers.has_ground_truth is True
        assert answers.ground_truth_path == gt
        assert answers.run_validation is True
        assert answers.validate_models == 3

    def test_validate_requires_paths(self, tmp_path):
        xml = tmp_path / "in.xml"
        xml.write_text("""<?xml version="1.0"?>
<ragadvisor>
  <query_patterns><validate>true</validate></query_patterns>
</ragadvisor>""", encoding="utf-8")
        with pytest.raises(InvalidInputError, match="ground_truth_path"):
            XmlInputParser(xml).parse()

    def test_negative_queries_per_day_rejected(self, tmp_path):
        xml = tmp_path / "in.xml"
        xml.write_text("""<?xml version="1.0"?>
<ragadvisor>
  <query_patterns><expected_queries_per_day>-5</expected_queries_per_day></query_patterns>
</ragadvisor>""", encoding="utf-8")
        with pytest.raises(InvalidInputError, match="expected_queries_per_day"):
            XmlInputParser(xml).parse()

    def test_shipped_example_parses(self):
        example = Path(__file__).parent.parent / "examples" / "answers.example.xml"
        answers = XmlInputParser(example).parse()
        assert answers.use_case.value == "question_answering"
        assert answers.expected_queries_per_day == 2000
        assert len(answers.sample_queries) == 2
        assert answers.run_validation is False

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
