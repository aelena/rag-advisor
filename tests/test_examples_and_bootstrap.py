"""Tests for the bundled example corpus, synthetic query bootstrap and JSON report."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

import rag_adviser.research as research_pkg
from rag_adviser.cli import app
from rag_adviser.evaluators.ground_truth_loader import GroundTruthLoader
from rag_adviser.evaluators.query_bootstrap import (
    SyntheticQuery,
    _first_sentence,
    _parse_llm_json,
    generate_queries,
    sample_chunks,
    write_jsonl,
)
from rag_adviser.models import ReportFormat
from rag_adviser.reporters.report_generator import ReportGenerator

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})
RESEARCH = Path(research_pkg.__file__).parent


class TestBundledGroundTruth:
    def test_every_query_points_at_a_real_passage(self) -> None:
        gt = GroundTruthLoader().load(RESEARCH / "ground_truth.jsonl")
        assert gt.query_count >= 40
        assert gt.synthetic_count == 0
        papers = {
            p.name: p.read_text(encoding="utf-8").lower()
            for p in (RESEARCH / "papers").glob("*.md")
        }
        for entry in gt.entries:
            assert entry.relevant_doc_ids, entry.query
            for doc in entry.relevant_doc_ids:
                assert doc in papers, f"{entry.query}: unknown doc {doc}"
            for passage in entry.relevant_passages:
                assert any(passage.lower() in papers[d] for d in entry.relevant_doc_ids), (
                    f"passage not found for: {entry.query}"
                )
        # Coverage: every paper has at least one query.
        covered = {d for e in gt.entries for d in e.relevant_doc_ids}
        assert covered == set(papers)

    def test_example_corpus_command(self, tmp_path: Path) -> None:
        dest = tmp_path / "ex"
        result = runner.invoke(app, ["example-corpus", str(dest)])
        assert result.exit_code == 0, result.stdout
        assert len(list((dest / "corpus").glob("*.md"))) == 14
        assert (dest / "queries.jsonl").exists()
        assert "--validate" in result.stdout


class TestBootstrap:
    def _corpus(self, tmp_path: Path) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir()
        for i in range(4):
            (docs / f"doc{i}.txt").write_text(
                f"Section {i}. " + f"Fact number {i} about topic {i} is important. " * 40, "utf-8"
            )
        return docs

    def test_sampling_spreads_across_files(self, tmp_path: Path) -> None:
        chunks = sample_chunks(self._corpus(tmp_path), n=6, chunk_chars=600)
        assert len(chunks) == 6
        assert len({c.source_file for c in chunks}) >= 3

    def test_generate_writes_synthetic_jsonl_and_skips_failures(self, tmp_path: Path) -> None:
        chunks = sample_chunks(self._corpus(tmp_path), n=4, chunk_chars=600)
        client = MagicMock()
        client.complete.side_effect = [
            '{"question": "What is fact number 0 about?", "answer": "Topic 0."}',
            '```json\n{"question": "Which topic does fact 1 concern?", '
            '"answer": "Topic 1."}\n```',
            "not json at all",
            '{"question": "What is fact number 0 about?", "answer": "dup"}',  # duplicate
        ]
        result = generate_queries(client, chunks)
        assert len(result.queries) == 2
        assert result.failures == 2
        path = write_jsonl(result.queries, tmp_path / "out" / "q.jsonl")

        gt = GroundTruthLoader().load(path)
        assert gt.query_count == 2
        assert gt.synthetic_count == 2
        assert all(e.relevant_doc_ids for e in gt.entries)
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert first["synthetic"] is True and first["passages"]

    def test_helpers(self) -> None:
        assert _parse_llm_json(' {"question": "Q?", "answer": "A"} ') == ("Q?", "A")
        assert _first_sentence("Hello there world. Second sentence.") == "Hello there world."
        q = SyntheticQuery(query="q", relevant_docs=["a"], passages=["p"], answer="a")
        assert json.loads(q.to_json())["synthetic"] is True

    def test_dry_run_needs_no_key(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = runner.invoke(
            app, ["bootstrap-queries", str(self._corpus(tmp_path)), "--n", "3", "--dry-run"]
        )
        assert result.exit_code == 0, result.stdout
        assert "Dry run" in result.stdout

    def test_missing_key_is_a_clear_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = runner.invoke(app, ["bootstrap-queries", str(self._corpus(tmp_path)), "--n", "2"])
        assert result.exit_code == 1
        assert "No LLM configured" in result.stdout


class TestJsonReport:
    def test_json_format(self, tmp_path: Path, sample_user_answers, sample_recommendations) -> None:
        paths = ReportGenerator().generate(
            answers=sample_user_answers, recommendations=sample_recommendations,
            output_dir=tmp_path, formats=[ReportFormat.JSON],
        )
        assert [p.name for p in paths] == ["rag_config.json"]
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        assert data["embedding"]["model"] == "BAAI/bge-small-en-v1.5"

    def test_all_includes_json(
        self, tmp_path: Path, sample_user_answers, sample_recommendations
    ) -> None:
        paths = ReportGenerator().generate(
            answers=sample_user_answers, recommendations=sample_recommendations,
            output_dir=tmp_path, formats=[ReportFormat.ALL],
        )
        assert {p.suffix for p in paths} == {".md", ".html", ".yaml", ".json"}
