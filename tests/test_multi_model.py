"""Tests for evaluating several embedding models in one run."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_adviser.evaluators.metrics import EvalMetrics
from rag_adviser.evaluators.pipeline_runner import EvalConfig, EvalPipelineRunner, EvalReport
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

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def encode(self, texts, **_):
        import numpy as np

        out = []
        for t in texts:
            toks = tokenize(t)
            vec = np.array([float(toks.count(w)) for w in self.VOCAB]) * self.scale + 1e-3
            out.append(vec / np.linalg.norm(vec))
        return np.array(out)


def _corpus(tmp_path: Path) -> EvalConfig:
    pytest.importorskip("numpy")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Error code E4021 means the pump overheated.", "utf-8")
    (docs / "b.txt").write_text("Invoice payment terms are thirty days.", "utf-8")
    gt = tmp_path / "gt.jsonl"
    gt.write_text(json.dumps({"query": "pump overheated E4021", "relevant_docs": ["a.txt"]}) + "\n")
    return EvalConfig(
        corpus_path=docs, ground_truth_path=gt, strategies=["recursive"],
        top_k=1, chunk_size=200, chunk_overlap=0, vector_backend="memory",
    )


def _run_with_fakes(config: EvalConfig, failing: set[str] = frozenset()) -> EvalReport:
    def fake_load(self, model_id, trust_remote_code=False):
        if model_id in failing:
            raise OSError(f"{model_id} is gated")
        self._model = _FakeModel()
        self._dimension = len(_FakeModel.VOCAB)

    with patch.object(EvalPipelineRunner, "_load_embedding_model", fake_load):
        return EvalPipelineRunner().run(config)


class TestPipelineMultiModel:
    def test_two_models_labelled_and_best_picked(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path)
        config.embedding_models = ["org/model-a", "org/model-b"]
        report = _run_with_fakes(config)

        assert [m.strategy_name for m in report.strategy_results] == [
            "recursive [model-a]", "recursive [model-b]",
        ]
        assert [m.embedding_model for m in report.strategy_results] == [
            "org/model-a", "org/model-b",
        ]
        assert report.best_model in {"org/model-a", "org/model-b"}
        assert report.embedding_models == ["org/model-a", "org/model-b"]

    def test_failing_model_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path)
        config.embedding_models = ["org/broken", "org/good"]
        report = _run_with_fakes(config, failing={"org/broken"})

        assert list(report.model_errors) == ["org/broken"]
        assert "gated" in report.model_errors["org/broken"]
        assert [m.embedding_model for m in report.strategy_results] == ["org/good"]

    def test_all_models_failing_raises(self, tmp_path: Path) -> None:
        from rag_adviser.evaluators.pipeline_runner import EvalPipelineError

        config = _corpus(tmp_path)
        config.embedding_models = ["org/x", "org/y"]
        with pytest.raises(EvalPipelineError, match="No embedding model could be evaluated"):
            _run_with_fakes(config, failing={"org/x", "org/y"})

    def test_max_successful_models_stops_early(self, tmp_path: Path) -> None:
        config = _corpus(tmp_path)
        config.embedding_models = ["org/broken", "org/first", "org/second"]
        config.max_successful_models = 1
        report = _run_with_fakes(config, failing={"org/broken"})
        # Fallback-chain behaviour: one failure skipped, one success, then stop.
        assert [m.embedding_model for m in report.strategy_results] == ["org/first"]
        # Single model evaluated -> plain labels
        assert report.strategy_results[0].strategy_name == "recursive [first]"


class TestValidatorComparison:
    def test_compares_models_and_picks_winner(self, tmp_path: Path) -> None:
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "x.txt").write_text("text", "utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text(json.dumps({"query": "q", "relevant_docs": ["x.txt"]}) + "\n")
        answers = UserAnswers(document_path=docs, ground_truth_path=gt, validate_models=2)
        recs = Recommendations(
            chunking=ChunkingRecommendation(),
            embedding_models=[
                EmbeddingModelRecommendation(model_id="org/a"),
                EmbeddingModelRecommendation(model_id="org/b"),
            ],
            retrieval=RetrievalRecommendation(top_k=5),
        )
        captured = {}

        def fake_run(self, config):
            captured["config"] = config
            return EvalReport(strategy_results=[
                EvalMetrics(strategy_name="recursive [a]", embedding_model="org/a",
                            num_queries=5, hit_rate=0.6, mrr=0.5),
                EvalMetrics(strategy_name="recursive [b]", embedding_model="org/b",
                            num_queries=5, hit_rate=0.8, mrr=0.75),
            ])

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(answers, recs)

        cfg = captured["config"]
        assert cfg.embedding_models[:2] == ["org/a", "org/b"]
        assert cfg.max_successful_models == 2
        assert result.ran is True
        assert result.embedding_model == "org/b"          # best by MRR
        assert result.hit_rate == 0.8
        assert [c["model"] for c in result.model_comparison] == ["org/b", "org/a"]
        assert any("won" in n for n in result.notes)
