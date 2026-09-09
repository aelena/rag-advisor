"""Tests for the recommendation validator (ragadvisor run --validate)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from rag_adviser.evaluators.metrics import EvalMetrics
from rag_adviser.evaluators.pipeline_runner import EvalConfig, EvalReport
from rag_adviser.evaluators.validator import RecommendationValidator, map_strategy
from rag_adviser.models import (
    ChunkingRecommendation,
    EmbeddingModelRecommendation,
    Recommendations,
    RetrievalRecommendation,
    UserAnswers,
)


def _corpus_and_gt(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "rag.txt").write_text("Retrieval augmented generation grounds answers.", "utf-8")
    gt = tmp_path / "gt.jsonl"
    gt.write_text(json.dumps({"query": "What is RAG?", "relevant_docs": ["rag.txt"]}) + "\n")
    return corpus, gt


def _recs(
    strategy: str = "hierarchical",
    top_k: int = 8,
    rerank: bool = False,
    hybrid: bool = False,
    api_first: bool = True,
    trc: bool = False,
) -> Recommendations:
    models = []
    if api_first:
        models.append(
            EmbeddingModelRecommendation(model_id="voyage/voyage-3-large", provider="voyage")
        )
    models.append(
        EmbeddingModelRecommendation(model_id="BAAI/bge-base-en-v1.5", trust_remote_code=trc)
    )
    return Recommendations(
        chunking=ChunkingRecommendation(
            strategy=strategy, chunk_size=1024, chunk_overlap=100, count_by="tokens"
        ),
        embedding_models=models,
        retrieval=RetrievalRecommendation(top_k=top_k, rerank=rerank, hybrid_search=hybrid),
    )


def _fake_report(hit_rate: float, mrr: float, strategy: str = "hierarchical") -> EvalReport:
    return EvalReport(
        strategy_results=[
            EvalMetrics(
                strategy_name=strategy,
                num_queries=3,
                num_chunks=12,
                hit_rate=hit_rate,
                mrr=mrr,
                mean_precision_at_k=0.3,
                mean_recall_at_k=hit_rate,
                mean_ndcg_at_k=mrr,
            )
        ]
    )


class TestStrategyMapping:
    def test_known_and_unknown(self) -> None:
        assert map_strategy("recursive") == "recursive"
        assert map_strategy("hierarchical") == "hierarchical"
        assert map_strategy("language_aware") == "adaptive"
        assert map_strategy("speaker_split") == "recursive"
        assert map_strategy("something_new") == "recursive"


class TestValidator:
    def test_missing_paths_do_not_run(self) -> None:
        result = RecommendationValidator().validate(UserAnswers(), _recs())
        assert result.ran is False
        assert "document path" in result.error

    def test_missing_ground_truth_file(self, tmp_path: Path) -> None:
        corpus, _ = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=tmp_path / "nope.jsonl")
        result = RecommendationValidator().validate(answers, _recs())
        assert result.ran is False
        assert "not found" in result.error

    def test_config_translation_and_strong_verdict(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)
        captured: dict[str, EvalConfig] = {}

        def fake_run(self, config):  # noqa: ANN001
            captured["config"] = config
            return _fake_report(hit_rate=0.9, mrr=0.8)

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(answers, _recs())

        cfg = captured["config"]
        assert cfg.embedding_models[0] == "BAAI/bge-base-en-v1.5"
        assert cfg.max_successful_models == 1
        assert cfg.strategies == ["hierarchical"]
        assert cfg.chunk_size == 1024 * 4          # tokens -> characters
        assert cfg.chunk_overlap == 100 * 4
        assert cfg.top_k == 8
        # Hosted API model skipped in favour of the first local one
        assert cfg.embedding_models[0] == "BAAI/bge-base-en-v1.5"
        assert any("hosted API" in n for n in result.notes)

        assert result.ran is True
        assert result.verdict == "strong"
        assert result.hit_rate == 0.9
        assert result.num_queries == 3

    def test_weak_verdict_produces_suggestions(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)

        with patch.object(
            RecommendationValidator, "_run", lambda self, cfg: _fake_report(0.3, 0.2)
        ):
            result = RecommendationValidator().validate(
                answers, _recs(top_k=5, rerank=False, hybrid=False)
            )

        assert result.verdict == "weak"
        joined = " ".join(result.suggestions).lower()
        assert "top_k" in joined
        assert "rerank" in joined
        assert "hybrid" in joined
        assert "ragadvisor evaluate" in joined

    def test_runner_exception_becomes_error(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)
        attempts: list[str] = []
        captured_models: list[str] = []

        def boom(self, cfg):  # noqa: ANN001
            attempts.append(cfg.embedding_models[0])
            captured_models.extend(cfg.embedding_models)
            raise RuntimeError("model download failed")

        with patch.object(RecommendationValidator, "_run", boom):
            result = RecommendationValidator().validate(answers, _recs())

        assert result.ran is False
        assert "RuntimeError" in result.error
        # One pipeline run; the fallback chain is handled inside the pipeline.
        assert attempts == ["BAAI/bge-base-en-v1.5"]
        assert captured_models == [
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

    def test_falls_back_to_next_model_on_load_failure(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)

        def flaky(self, cfg):  # noqa: ANN001
            # The pipeline skips a model that fails to load and records why.
            report = _fake_report(0.85, 0.7)
            report.model_errors["BAAI/bge-base-en-v1.5"] = "ImportError: No module named einops"
            for m in report.strategy_results:
                m.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            return report

        with patch.object(RecommendationValidator, "_run", flaky):
            result = RecommendationValidator().validate(answers, _recs())

        assert result.ran is True
        assert result.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert any("einops" in n for n in result.notes)

    def test_import_error_gives_install_hint(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)

        def missing(self, cfg):  # noqa: ANN001
            raise ImportError("No module named sentence_transformers")

        with patch.object(RecommendationValidator, "_run", missing):
            result = RecommendationValidator().validate(answers, _recs())

        assert "ragadvisor[eval]" in result.error

    def test_all_api_models_fall_back_to_default(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)
        recs = _recs()
        recs.embedding_models = [
            EmbeddingModelRecommendation(
                model_id="openai/text-embedding-3-small", provider="openai"
            )
        ]
        captured = {}

        def fake_run(self, config):  # noqa: ANN001
            captured["model"] = config.embedding_models[0]
            return _fake_report(0.9, 0.9)

        with patch.object(RecommendationValidator, "_run", fake_run):
            RecommendationValidator().validate(answers, recs)

        assert captured["model"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_trust_remote_code_propagates(self, tmp_path: Path) -> None:
        corpus, gt = _corpus_and_gt(tmp_path)
        answers = UserAnswers(document_path=corpus, ground_truth_path=gt)
        captured = {}

        def fake_run(self, config):  # noqa: ANN001
            captured["trc"] = "BAAI/bge-base-en-v1.5" in config.trust_remote_code_models
            return _fake_report(0.9, 0.9)

        with patch.object(RecommendationValidator, "_run", fake_run):
            RecommendationValidator().validate(answers, _recs(api_first=False, trc=True))

        assert captured["trc"] is True
