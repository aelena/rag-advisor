"""Tests for the chunk-size sweep in evaluate and --validate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_adviser.evaluators.metrics import EvalMetrics
from rag_adviser.evaluators.pipeline_runner import (
    EvalConfig,
    EvalPipelineRunner,
    EvalReport,
    generate_eval_report_markdown,
)
from rag_adviser.evaluators.retrieval_modes import tokenize
from rag_adviser.evaluators.validator import RecommendationValidator
from rag_adviser.models import (
    ChunkingRecommendation,
    EmbeddingModelRecommendation,
    Recommendations,
    RetrievalRecommendation,
    UserAnswers,
)


class _FakeModel:
    VOCAB = ["pump", "overheated", "e4021", "invoice", "cooling", "error", "payment", "fan"]

    def encode(self, texts, **_):
        import numpy as np

        out = []
        for t in texts:
            toks = tokenize(t)
            vec = np.array([float(toks.count(w)) for w in self.VOCAB]) + 1e-3
            out.append(vec / np.linalg.norm(vec))
        return np.array(out)


def _config(tmp_path: Path) -> EvalConfig:
    pytest.importorskip("numpy")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "Error code E4021 means the pump overheated. " * 6 + "The cooling fan failed. " * 6, "utf-8"
    )
    (docs / "b.txt").write_text("Invoice payment terms are thirty days. " * 12, "utf-8")
    gt = tmp_path / "gt.jsonl"
    gt.write_text(json.dumps({"query": "pump overheated E4021", "relevant_docs": ["a.txt"]}) + "\n")
    return EvalConfig(
        corpus_path=docs, ground_truth_path=gt, strategies=["recursive"],
        top_k=2, vector_backend="memory",
    )


def _run(config: EvalConfig) -> EvalReport:
    def fake_load(self, model_id, trust_remote_code=False):
        self._model = _FakeModel()
        self._dimension = len(_FakeModel.VOCAB)

    with patch.object(EvalPipelineRunner, "_load_embedding_model", fake_load):
        return EvalPipelineRunner().run(config)


class TestPipelineSweep:
    def test_sweep_labels_sizes_and_scales_overlap(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.chunk_sizes = [100, 300]
        config.overlap_ratio = 0.1
        report = _run(config)

        labels = [m.strategy_name for m in report.strategy_results]
        assert labels == ["recursive @100", "recursive @300"]
        assert [m.chunk_size for m in report.strategy_results] == [100, 300]
        assert [m.chunk_overlap for m in report.strategy_results] == [10, 30]
        # Smaller chunks -> more chunks
        assert report.strategy_results[0].num_chunks > report.strategy_results[1].num_chunks
        assert report.best_chunk_size in (100, 300)

        md = generate_eval_report_markdown(report, tmp_path / "out").read_text("utf-8")
        assert "Chunk-size sweep" in md
        assert "| recursive | " in md
        assert "**Best chunk size:**" in md

    def test_single_size_keeps_plain_labels_and_fixed_overlap(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.chunk_size = 200
        config.chunk_overlap = 25
        report = _run(config)
        assert report.strategy_results[0].strategy_name == "recursive"
        assert report.strategy_results[0].chunk_size == 200
        assert report.strategy_results[0].chunk_overlap == 25
        assert report.best_chunk_size == 200


class TestValidatorSweep:
    def _answers(self, tmp_path: Path, sizes: list[int]) -> UserAnswers:
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "x.txt").write_text("text", "utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text(json.dumps({"query": "q", "relevant_docs": ["x.txt"]}) + "\n")
        return UserAnswers(document_path=docs, ground_truth_path=gt, validate_chunk_sizes=sizes)

    def _recs(self) -> Recommendations:
        return Recommendations(
            chunking=ChunkingRecommendation(chunk_size=512, chunk_overlap=50, count_by="tokens"),
            embedding_models=[EmbeddingModelRecommendation(model_id="org/a")],
            retrieval=RetrievalRecommendation(top_k=5),
        )

    def test_config_gets_sizes_in_chars_and_winner_is_reported(self, tmp_path: Path) -> None:
        answers = self._answers(tmp_path, [256, 1024])
        captured = {}

        def fake_run(self, config):
            captured["config"] = config
            return EvalReport(strategy_results=[
                EvalMetrics(strategy_name="recursive @1024", embedding_model="org/a",
                            chunk_size=1024, num_queries=8, hit_rate=0.5, mrr=0.4),
                EvalMetrics(strategy_name="recursive @2048", embedding_model="org/a",
                            chunk_size=2048, num_queries=8, hit_rate=0.75, mrr=0.6),
                EvalMetrics(strategy_name="recursive @4096", embedding_model="org/a",
                            chunk_size=4096, num_queries=8, hit_rate=0.9, mrr=0.85),
            ])

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(answers, self._recs())

        cfg = captured["config"]
        assert cfg.chunk_sizes == [1024, 2048, 4096]     # 256/512/1024 tokens x 4
        assert cfg.overlap_ratio == pytest.approx(50 / 512)
        assert result.ran is True
        assert result.recommended_chunk_size_tokens == 512
        assert result.best_chunk_size_tokens == 1024
        assert result.hit_rate == 0.9                     # metrics of the winning size
        assert [c["chunk_size_tokens"] for c in result.chunk_size_comparison] == [1024, 512, 256]
        assert any("1024 tokens beat the recommended 512" in s for s in result.suggestions)

    def test_recommended_size_winning_is_a_note(self, tmp_path: Path) -> None:
        answers = self._answers(tmp_path, [256])

        def fake_run(self, config):
            return EvalReport(strategy_results=[
                EvalMetrics(strategy_name="recursive @1024", embedding_model="org/a",
                            chunk_size=1024, num_queries=8, hit_rate=0.6, mrr=0.5),
                EvalMetrics(strategy_name="recursive @2048", embedding_model="org/a",
                            chunk_size=2048, num_queries=8, hit_rate=0.9, mrr=0.9),
            ])

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(answers, self._recs())

        assert result.best_chunk_size_tokens == 512
        assert any("was the best of 2 sizes" in n for n in result.notes)
        assert not any("beat the recommended" in s for s in result.suggestions)

    def test_no_sweep_when_no_extra_sizes(self, tmp_path: Path) -> None:
        answers = self._answers(tmp_path, [])
        captured = {}

        def fake_run(self, config):
            captured["config"] = config
            return EvalReport(strategy_results=[
                EvalMetrics(strategy_name="recursive", embedding_model="org/a",
                            chunk_size=2048, hit_rate=1.0, mrr=1.0),
            ])

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(answers, self._recs())
        assert captured["config"].chunk_sizes == [2048]
        assert result.chunk_size_comparison == []
