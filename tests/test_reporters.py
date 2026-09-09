"""Tests for report generators."""


from rag_adviser.models import ReportFormat
from rag_adviser.reporters.html_renderer import HtmlRenderer
from rag_adviser.reporters.markdown_renderer import MarkdownRenderer
from rag_adviser.reporters.report_generator import ReportGenerator
from rag_adviser.reporters.yaml_renderer import YamlRenderer


class TestMarkdownRenderer:
    def test_generates_file(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = MarkdownRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        assert path.exists()
        assert path.suffix == ".md"

    def test_contains_sections(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = MarkdownRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "# RAG Configuration Audit Report" in content
        assert "## Embedding Model" in content
        assert "## Chunking Strategy" in content
        assert "## Vector Database" in content
        assert "## Retrieval Settings" in content
        assert "## Re-Indexing Warning" in content

    def test_contains_model_info(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = MarkdownRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "BAAI/bge-small-en-v1.5" in content

    def test_contains_user_input(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = MarkdownRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "question_answering" in content


class TestYamlRenderer:
    def test_generates_file(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = YamlRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        assert path.exists()
        assert path.name == "rag_config.yaml"

    def test_valid_yaml(self, tmp_path, sample_user_answers, sample_recommendations):
        import yaml
        renderer = YamlRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "embedding" in data
        assert "chunking" in data
        assert "vector_db" in data

    def test_embedding_model_in_yaml(self, tmp_path, sample_user_answers, sample_recommendations):
        import yaml
        renderer = YamlRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["embedding"]["model"] == "BAAI/bge-small-en-v1.5"


class TestHtmlRenderer:
    def test_generates_file(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = HtmlRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        assert path.exists()
        assert path.suffix == ".html"

    def test_contains_html_structure(self, tmp_path, sample_user_answers, sample_recommendations):
        renderer = HtmlRenderer()
        path = renderer.render(sample_user_answers, sample_recommendations, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<h1>" in content
        assert "ragadvisor" in content.lower()


class TestReportGenerator:
    def test_generate_markdown(self, tmp_path, sample_user_answers, sample_recommendations):
        gen = ReportGenerator()
        paths = gen.generate(
            answers=sample_user_answers,
            recommendations=sample_recommendations,
            output_dir=tmp_path,
            formats=[ReportFormat.MARKDOWN],
        )
        assert len(paths) == 1
        assert paths[0].suffix == ".md"

    def test_generate_multiple_formats(self, tmp_path, sample_user_answers, sample_recommendations):
        gen = ReportGenerator()
        paths = gen.generate(
            answers=sample_user_answers,
            recommendations=sample_recommendations,
            output_dir=tmp_path,
            formats=[ReportFormat.MARKDOWN, ReportFormat.HTML],
        )
        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert ".md" in suffixes
        assert ".html" in suffixes

    def test_creates_output_dir(self, tmp_path, sample_user_answers, sample_recommendations):
        output = tmp_path / "new_dir" / "reports"
        gen = ReportGenerator()
        paths = gen.generate(
            answers=sample_user_answers,
            recommendations=sample_recommendations,
            output_dir=output,
            formats=[ReportFormat.YAML],
        )
        assert output.exists()
        assert len(paths) == 1
