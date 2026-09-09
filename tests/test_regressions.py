"""Regression tests for defects found in the 0.2.0 review.

Each test pins a behaviour that was previously wrong:
- corpus token totals were computed from a 50-file sample and never scaled up
- prose with commas was classified as tabular (and then routed to Text-to-SQL)
- a --content-type override was dropped when no corpus was analyzed
- the use case never reached the chunking recommender
- --preset with --no-interactive silently ignored every other flag
- hosted embedding APIs were never recommended, even for paid_api budgets
- the Anthropic client sent `temperature`, which current models reject
"""

from __future__ import annotations

from pathlib import Path

from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.input_modes.cli_params import CliParamsCollector
from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    BudgetTier,
    ContentType,
    HardwareConstraints,
    PrivacyLevel,
    ReportFormat,
    UpdateFrequency,
    UseCase,
    UserAnswers,
)
from rag_adviser.presets.manager import PresetManager
from rag_adviser.recommenders.model_finder import HFModelFinder
from rag_adviser.recommenders.vector_db_recommender import VectorDBRecommender

PROSE = (
    "Retrieval-augmented generation combines a retriever, a reader, and a generator. "
    "It was proposed to ground answers in evidence, reduce hallucination, and allow "
    "knowledge updates without retraining. The approach has been adopted widely.\n"
) * 8


class TestDocumentAnalyzerRegressions:
    def test_total_tokens_extrapolated_to_whole_corpus(self, tmp_path: Path) -> None:
        # 120 identical files, but only 50 are ever opened.
        for i in range(120):
            (tmp_path / f"doc_{i:03d}.txt").write_text(PROSE, encoding="utf-8")

        stats = DocumentAnalyzer().analyze(tmp_path)

        assert stats.total_files == 120
        assert stats.sampled_files == 50
        # One file is ~250 tokens; the total must reflect all 120 files, not 50.
        assert stats.total_tokens > 120 * 150
        assert abs(stats.total_tokens - stats.avg_tokens_per_doc * 120) < 1

    def test_truncated_long_file_scaled(self, tmp_path: Path) -> None:
        # Far longer than the 5,000-char sampling cap.
        (tmp_path / "long.txt").write_text(PROSE * 40, encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        # ~8,000+ tokens in the file; the old code capped this near ~1,100.
        assert stats.total_tokens > 5_000

    def test_prose_with_commas_is_not_tabular(self, tmp_path: Path) -> None:
        (tmp_path / "essay.txt").write_text(PROSE, encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.detected_content_type == ContentType.PROSE

    def test_csv_is_tabular_by_extension(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text(
            "id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30\n", encoding="utf-8"
        )
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.detected_content_type == ContentType.TABULAR

    def test_code_by_extension(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("import os\nprint(os.name)\n", encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.detected_content_type == ContentType.CODE

    def test_mixed_corpus_reported_as_mixed(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
        (tmp_path / "b.csv").write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text(PROSE, encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.detected_content_type == ContentType.MIXED
        assert stats.content_type_confidence < 0.5


class TestPipelineRegressions:
    def test_content_type_override_without_corpus(self, tmp_path: Path) -> None:
        answers = UserAnswers(
            content_type_override=ContentType.CODE,
            use_case=UseCase.CODE,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
        )
        recs = RAGAdviser().run(answers, tmp_path, [ReportFormat.YAML])
        assert recs.chunking is not None
        assert recs.chunking.content_type == "code"
        assert recs.chunking.strategy == "language_aware"

    def test_use_case_reaches_chunking(self, tmp_path: Path) -> None:
        answers = UserAnswers(
            use_case=UseCase.SUMMARIZATION,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
        )
        recs = RAGAdviser().run(answers, tmp_path, [ReportFormat.YAML])
        assert any("summarization" in n.lower() for n in recs.chunking.notes)

    def test_vector_db_snippet_uses_model_dimension(self) -> None:
        rec = VectorDBRecommender().recommend(
            doc_count=10,
            constraints=HardwareConstraints(),
            update_frequency=UpdateFrequency.NEVER,
            estimated_chunks=100,
            embedding_dimension=1024,
        )
        assert rec.provider  # something was chosen
        assert "384" not in rec.code_snippet or "1024" in rec.code_snippet


class TestPresetFlagMerge:
    def test_cli_flags_override_preset(self) -> None:
        mgr = PresetManager()
        base = mgr.apply_preset(mgr.load_preset("legal-discovery"))
        assert base.constraints.ram_gb == 32

        merged = CliParamsCollector.from_options(base=base, ram_gb=8, privacy="air_gapped")

        assert merged.constraints.ram_gb == 8
        assert merged.constraints.privacy == PrivacyLevel.AIR_GAPPED
        # Untouched preset values survive.
        assert merged.use_case == UseCase.LEGAL
        assert merged.content_type_override == ContentType.LEGAL


class TestModelFinderRegressions:
    def test_api_models_offered_for_paid_budget(self) -> None:
        finder = HFModelFinder(offline=True)
        hw = HardwareConstraints(budget=BudgetTier.PAID_API, privacy=PrivacyLevel.NONE)
        models = finder.find_models(doc_stats=None, constraints=hw, use_case=UseCase.QA)
        assert any(m.provider != "huggingface" for m in models)

    def test_api_models_excluded_for_strict_privacy(self) -> None:
        finder = HFModelFinder(offline=True)
        hw = HardwareConstraints(budget=BudgetTier.PAID_API, privacy=PrivacyLevel.STRICT)
        models = finder.find_models(doc_stats=None, constraints=hw, use_case=UseCase.QA)
        assert all(m.provider == "huggingface" for m in models)

    def test_quality_outranks_tiny_model(self) -> None:
        # With no latency pressure and 16GB RAM, the tiny MiniLM model should
        # not beat higher-quality BGE models on an English corpus.
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=None, constraints=HardwareConstraints(), use_case=UseCase.QA
        )
        assert models[0].model_id != "sentence-transformers/all-MiniLM-L6-v2"
        assert models[0].quality_score >= 50

    def test_trust_remote_code_flagged(self) -> None:
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=None, constraints=HardwareConstraints(), use_case=UseCase.QA
        )
        nomic = next((m for m in models if m.model_id.startswith("nomic-ai/")), None)
        if nomic is not None:
            assert nomic.trust_remote_code is True
            assert any("trust_remote_code" in w for w in nomic.warnings)

    def test_non_commercial_license_flagged(self) -> None:
        finder = HFModelFinder(offline=True)
        hw = HardwareConstraints(ram_gb=64)
        rec = finder._score_model(
            {
                "model_id": "x/y",
                "license": "cc-by-nc-4.0",
                "estimated_size_gb": 0.5,
                "quality_score": 55,
            },
            ["en"], False, 3.0, hw, UseCase.QA,
        )
        assert any("commercial" in w for w in rec.warnings)
