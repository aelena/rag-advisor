"""Tests for CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from rag_adviser.cli import app

# Plain, wide output so Rich does not wrap or colour option names in CI terminals.
runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})


class TestCli:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "ragadvisor" in result.stdout
        assert "0.3.1" in result.stdout

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "analyze" in result.stdout
        assert "version" in result.stdout

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--from-xml" in result.stdout
        assert "--use-case" in result.stdout
        assert "--format" in result.stdout

    def test_run_non_interactive_basic(self, tmp_path):
        result = runner.invoke(app, [
            "run",
            "--no-interactive",
            "--use-case", "question_answering",
            "--privacy", "strict",
            "--format", "yaml",
            "--output", str(tmp_path / "output"),
            "--no-llm",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "output" / "rag_config.yaml").exists()

    def test_run_from_xml(self, tmp_path):
        fixtures = Path(__file__).parent / "fixtures"
        result = runner.invoke(app, [
            "run",
            "--from-xml", str(fixtures / "sample_input.xml"),
            "--format", "markdown",
            "--output", str(tmp_path / "output"),
        ])
        assert result.exit_code == 0
        assert (tmp_path / "output" / "rag_report.md").exists()

    def test_run_invalid_use_case(self, tmp_path):
        result = runner.invoke(app, [
            "run",
            "--no-interactive",
            "--use-case", "invalid_use_case",
            "--format", "yaml",
            "--output", str(tmp_path / "output"),
            "--no-llm",
        ])
        assert result.exit_code == 1

    def test_analyze_nonexistent_path(self):
        result = runner.invoke(app, ["analyze", "/nonexistent/path"])
        assert result.exit_code == 1
