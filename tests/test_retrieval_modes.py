"""Tests for hybrid (BM25 + RRF) and rerank stages in the evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_adviser.evaluators.chunking_strategies import Chunk
from rag_adviser.evaluators.metrics import EvalMetrics
from rag_adviser.evaluators.pipeline_runner import EvalConfig, EvalReport
from rag_adviser.evaluators.retrieval_modes import BM25Index, mode_label, rrf_fuse, tokenize
from rag_adviser.evaluators.validator import RecommendationValidator
from rag_adviser.evaluators.vector_store import SearchResult
from rag_adviser.models import (
    ChunkingRecommendation,
    EmbeddingModelRecommendation,
    Recommendations,
    RerankerRecommendation,
    RetrievalRecommendation,
    UserAnswers,
)


def _chunk(text: str, src: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, source_file=src, chunk_index=idx, strategy="recursive")


class TestBM25:
    def test_exact_term_wins(self) -> None:
        index = BM25Index([
            _chunk("error code E4021 means the pump overheated", "a.txt"),
            _chunk("general guidance about pumps and cooling", "b.txt"),
            _chunk("unrelated text about invoices", "c.txt"),
        ])
        results = index.search("E4021", top_k=3)
        assert results[0].source_file == "a.txt"
        assert len(results) == 1  # only chunks containing the term score > 0

    def test_empty_and_unknown_query(self) -> None:
        index = BM25Index([_chunk("hello world", "a.txt")])
        assert index.search("zzz", top_k=5) == []
        assert BM25Index([]).search("hello", top_k=5) == []
        assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


class TestRRF:
    def test_fusion_prefers_items_in_both_lists(self) -> None:
        a = SearchResult(text="a", source_file="a.txt", metadata={"chunk_index": 0})
        b = SearchResult(text="b", source_file="b.txt", metadata={"chunk_index": 0})
        c = SearchResult(text="c", source_file="c.txt", metadata={"chunk_index": 0})
        fused = rrf_fuse([[a, b], [b, c]], top_k=3)
        assert [r.source_file for r in fused] == ["b.txt", "a.txt", "c.txt"]
        assert fused[0].score > fused[1].score

    def test_mode_labels(self) -> None:
        assert mode_label(False, False) == "dense"
        assert mode_label(True, False) == "hybrid"
        assert mode_label(False, True) == "dense+rerank"
        assert mode_label(True, True) == "hybrid+rerank"


class _FakeModel:
    """Bag-of-words embedder so the pipeline runs without sentence-transformers."""

    VOCAB = ["pump", "overheated", "e4021", "invoice", "cooling", "error", "payment", "fan"]

    def encode(self, texts, **_):
        import numpy as np

        out = []
        for t in texts:
            toks = tokenize(t)
            vec = np.array([float(toks.count(w)) for w in self.VOCAB]) + 1e-3
            out.append(vec / np.linalg.norm(vec))
        return np.array(out)


class TestPipelineModes:
    def _setup(self, tmp_path: Path) -> EvalConfig:
        pytest.importorskip("numpy")
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("Error code E4021 means the pump overheated.", "utf-8")
        (docs / "b.txt").write_text("The cooling fan keeps the pump from overheating.", "utf-8")
        (docs / "c.txt").write_text("Invoice payment terms are thirty days.", "utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text(
            json.dumps({"query": "what does E4021 mean", "relevant_docs": ["a.txt"]}) + "\n"
        )
        return EvalConfig(
            corpus_path=docs, ground_truth_path=gt, strategies=["recursive"],
            top_k=1, chunk_size=200, chunk_overlap=0, vector_backend="memory",
        )

    def _run(self, config: EvalConfig):
        from rag_adviser.evaluators.pipeline_runner import EvalPipelineRunner

        runner = EvalPipelineRunner()

        def fake_load(self, model_id, trust_remote_code=False):
            self._model = _FakeModel()
            self._dimension = len(_FakeModel.VOCAB)

        with patch.object(EvalPipelineRunner, "_load_embedding_model", fake_load):
            return runner.run(config)

    def test_hybrid_with_dense_baseline(self, tmp_path: Path) -> None:
        config = self._setup(tmp_path)
        config.hybrid = True
        config.dense_baseline = True
        report = self._run(config)

        names = [m.strategy_name for m in report.strategy_results]
        assert names == ["recursive", "recursive (dense baseline)"]
        assert report.strategy_results[0].retrieval_mode == "hybrid"
        assert report.strategy_results[1].retrieval_mode == "dense"
        # The exact error code is only in a.txt; BM25 guarantees the hit.
        assert report.strategy_results[0].hit_rate == 1.0

    def test_dense_only_has_no_baseline_run(self, tmp_path: Path) -> None:
        config = self._setup(tmp_path)
        config.dense_baseline = True  # ignored without hybrid/rerank
        report = self._run(config)
        assert len(report.strategy_results) == 1
        assert report.strategy_results[0].retrieval_mode == "dense"


class TestValidatorModes:
    def _recs(self, hybrid: bool, rerank: bool) -> Recommendations:
        rr = RerankerRecommendation(
            enabled=rerank, model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            provider="huggingface", fetch_k=24,
        )
        return Recommendations(
            chunking=ChunkingRecommendation(strategy="recursive", chunk_size=256, chunk_overlap=20),
            embedding_models=[EmbeddingModelRecommendation(model_id="BAAI/bge-small-en-v1.5")],
            retrieval=RetrievalRecommendation(top_k=5, hybrid_search=hybrid, rerank=rerank),
            reranker=rr,
        )

    def test_config_carries_modes_and_baseline_is_compared(self, tmp_path: Path) -> None:
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "x.txt").write_text("text", "utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text(json.dumps({"query": "q", "relevant_docs": ["x.txt"]}) + "\n")
        answers = UserAnswers(document_path=docs, ground_truth_path=gt)
        captured = {}

        def fake_run(self, config):
            captured["config"] = config
            return EvalReport(strategy_results=[
                EvalMetrics(strategy_name="recursive", retrieval_mode="hybrid+rerank",
                            num_queries=10, hit_rate=0.9, mrr=0.8),
                EvalMetrics(strategy_name="recursive (dense baseline)", retrieval_mode="dense",
                            num_queries=10, hit_rate=0.6, mrr=0.5),
            ])

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(answers, self._recs(True, True))

        cfg = captured["config"]
        assert cfg.hybrid is True
        assert cfg.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert cfg.fetch_k == 24
        assert cfg.dense_baseline is True
        assert result.retrieval_mode == "hybrid+rerank"
        assert result.has_baseline and result.baseline_hit_rate == 0.6
        assert any("improved hit rate" in n for n in result.notes)

    def test_hosted_reranker_is_noted_not_run(self, tmp_path: Path) -> None:
        docs = tmp_path / "d"
        docs.mkdir()
        (docs / "x.txt").write_text("text", "utf-8")
        gt = tmp_path / "gt.jsonl"
        gt.write_text(json.dumps({"query": "q", "relevant_docs": ["x.txt"]}) + "\n")
        recs = self._recs(False, True)
        recs.reranker.provider = "cohere"
        recs.reranker.model_id = "cohere/rerank-v3.5"
        captured = {}

        def fake_run(self, config):
            captured["config"] = config
            return EvalReport(
                strategy_results=[EvalMetrics(strategy_name="recursive", hit_rate=1.0)]
            )

        with patch.object(RecommendationValidator, "_run", fake_run):
            result = RecommendationValidator().validate(
                UserAnswers(document_path=docs, ground_truth_path=gt), recs
            )
        assert captured["config"].rerank_model is None
        assert any("not evaluated locally" in n for n in result.notes)
