"""Regression tests for defects found in the 0.2.0 review.

Each test pins a behaviour that was previously wrong:
- corpus token totals were computed from a 50-file sample and never scaled up
- prose with commas was classified as tabular (and then routed to Text-to-SQL)
- a --content-type override was dropped when no corpus was analyzed
- the use case never reached the chunking recommender
- --preset with --no-interactive silently ignored every other flag
- hosted embedding APIs were never recommended, even for paid_api budgets
- the Anthropic client sent `temperature`, which current models reject

0.3.3 review adds:
- legacy .ppt routed to python-pptx (which can't open it)
- silently defaulted query preprocessing to lowercase + strip punctuation
"""

from __future__ import annotations

from pathlib import Path

from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.input_modes.cli_params import CliParamsCollector
from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    BudgetTier,
    ContentType,
    DocumentStats,
    HardwareConstraints,
    PrivacyLevel,
    ReportFormat,
    RetrievalRecommendation,
    UpdateFrequency,
    UseCase,
    UserAnswers,
)
from rag_adviser.presets.manager import PresetManager
from rag_adviser.recommenders.modality_recommender import ModalityRecommender
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


class TestPresentationModalityRoutingRegressions:
    """0.3.3: python-pptx only opens .pptx; legacy .ppt / .odp / .key need
    LibreOffice conversion first. The report used to recommend python-pptx
    for every presentation extension."""

    @staticmethod
    def _stats(extensions: dict[str, int]) -> DocumentStats:
        return DocumentStats(
            total_files=0,
            total_files_all=sum(extensions.values()),
            modalities={"presentation": sum(extensions.values())},
            file_types=extensions,
        )

    def test_legacy_ppt_triggers_conversion_and_warning(self) -> None:
        stats = self._stats({".ppt": 3})
        rec = ModalityRecommender().recommend(stats, UserAnswers())[0]
        assert rec.modality == "presentation"
        assert any("LibreOffice" in step for step in rec.ingestion), rec.ingestion
        assert any(".ppt" in w and "python-pptx" in w for w in rec.warnings)

    def test_pptx_only_does_not_warn_about_conversion(self) -> None:
        stats = self._stats({".pptx": 3})
        rec = ModalityRecommender().recommend(stats, UserAnswers())[0]
        assert any("python-pptx" in step for step in rec.ingestion), rec.ingestion
        assert not any(
            "must be converted" in w or "cannot open" in w for w in rec.warnings
        ), rec.warnings

    def test_mixed_pptx_and_legacy_covers_both_paths(self) -> None:
        stats = self._stats({".pptx": 1, ".ppt": 1, ".odp": 1})
        rec = ModalityRecommender().recommend(stats, UserAnswers())[0]
        assert any("python-pptx" in step for step in rec.ingestion)
        assert any("LibreOffice" in step for step in rec.ingestion)


class TestFormatTaxonomyRegressions:
    """0.5.0 Phase 6: reclassify formats.

    - ``.epub`` / ``.docx`` / ``.doc`` / ``.mobi`` / ``.rtf`` / ``.html`` /
      ``.htm`` / ``.djvu`` / ``.chm`` / ``.opf`` are first-class text
      documents, not "other".
    - ``.crdownload``, ``.lnk``, ``.msi``, ``.db``, ``.bin``, ``.dat``,
      ML checkpoint / font / disc-image extensions are excluded entirely.
    - ``.gp`` / ``.ptb`` land in the new ``unsupported`` modality.
    - Filename fragments (extension contains spaces or is >6 chars) are
      excluded and reported as a separate warning.
    """

    def test_epub_counted_as_text_document(self, tmp_path: Path) -> None:
        # A file with a .epub suffix should be a document, even when
        # ebooklib is not available to extract text (the count still
        # matters, only the token estimate degrades).
        (tmp_path / "book.epub").write_bytes(b"PK\x03\x04")   # bare zip header
        (tmp_path / "notes.txt").write_text(PROSE, encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.total_files == 2
        assert stats.modalities["document"] == 2
        assert "other" not in stats.modalities

    def test_ml_checkpoint_extensions_are_excluded(self, tmp_path: Path) -> None:
        (tmp_path / "prose.txt").write_text(PROSE, encoding="utf-8")
        for junk in ("model.safetensors", "model.pt", "weights.pkl",
                     "cache.bin", "weights.ckpt", "installer.msi"):
            (tmp_path / junk).write_bytes(b"\x00" * 4)
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.total_files_all == 1  # only the .txt counts
        assert ".safetensors" not in stats.file_types
        assert ".pt" not in stats.file_types
        assert ".msi" not in stats.file_types

    def test_guitar_tab_is_unsupported_not_other(self, tmp_path: Path) -> None:
        (tmp_path / "prose.txt").write_text(PROSE, encoding="utf-8")
        (tmp_path / "song.gp").write_bytes(b"\x00" * 4)
        (tmp_path / "riff.ptb").write_bytes(b"\x00" * 4)
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.modalities.get("unsupported") == 2
        assert stats.modalities.get("other", 0) == 0

    def test_filename_fragment_excluded_and_reported(self, tmp_path: Path) -> None:
        (tmp_path / "prose.txt").write_text(PROSE, encoding="utf-8")
        # Files whose "suffix" is a spurious fragment because the
        # filename contains a dot in the middle (this is what happens on
        # a real Books/Papers folder: "Jain. Machine Learning" reports
        # ``. Machine Learning`` as the suffix).
        (tmp_path / "Book.machine learning paradigms").write_text(
            "junk", encoding="utf-8"
        )
        (tmp_path / "Author.chapter 2 draft").write_text(
            "junk", encoding="utf-8"
        )
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.total_files == 1
        assert len(stats.filename_fragment_paths) == 2
        # And the outer pipeline turns that into a warning line.
        answers = UserAnswers(
            document_path=tmp_path,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
        )
        out = tmp_path / "out"
        recs = RAGAdviser().run(answers, out, [ReportFormat.MARKDOWN])
        assert any("FILENAME FRAGMENTS" in w for w in recs.warnings)


class TestErrorCostRegressions:
    """0.6.0 Phase 8: ``error_cost`` on UserAnswers flips retrieval's
    precision / recall balance so a "wrong answer is worse than no
    answer" scenario doesn't silently share defaults with "always
    answer" scenarios (review §12)."""

    def test_wrong_worse_tightens_threshold(self, tmp_path: Path) -> None:
        from rag_adviser.models import ErrorCost
        answers = UserAnswers(
            use_case=UseCase.QA,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
            error_cost=ErrorCost.WRONG_WORSE,
        )
        recs = RAGAdviser().run(answers, tmp_path, [ReportFormat.YAML])
        assert recs.retrieval.similarity_threshold >= 0.80
        assert any("wrong_worse" in n for n in recs.retrieval.notes)

    def test_no_answer_worse_drops_threshold_and_widens_topk(
        self, tmp_path: Path
    ) -> None:
        from rag_adviser.models import ErrorCost
        answers = UserAnswers(
            use_case=UseCase.QA,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
            error_cost=ErrorCost.NO_ANSWER_WORSE,
        )
        recs = RAGAdviser().run(answers, tmp_path, [ReportFormat.YAML])
        assert recs.retrieval.similarity_threshold == 0.0
        assert recs.retrieval.top_k >= 8
        assert any("no_answer_worse" in n for n in recs.retrieval.notes)


class TestPhysicalSizingRegressions:
    """0.5.0 Phase 4: index footprint now uses a SizingProfile bundle
    (fp16 halves memory, quantization further compresses, on-disk mmap
    keeps only HNSW in RAM). Pre-0.5.0 the ``index_memory_mb`` was
    fp32 with a fixed ~50% overhead regardless of the deployment."""

    @staticmethod
    def _base_recs():
        # Minimal recs sufficient to run the estimator.
        from rag_adviser.models import (
            ChunkingRecommendation,
            EmbeddingModelRecommendation,
            Recommendations,
            RetrievalRecommendation,
            VectorDBRecommendation,
        )
        return Recommendations(
            embedding_models=[
                EmbeddingModelRecommendation(
                    model_id="BAAI/bge-m3", dimension=1024,
                    estimated_size_gb=2.27,
                )
            ],
            chunking=ChunkingRecommendation(chunk_size=512, chunk_overlap=50),
            vector_db=VectorDBRecommendation(category="client-server"),
            retrieval=RetrievalRecommendation(top_k=5),
        )

    def test_fp16_halves_index_memory_vs_fp32(self) -> None:
        from rag_adviser.models import DocumentStats, SizingProfile
        from rag_adviser.recommenders.cost_estimator import CostEstimator

        stats = DocumentStats(total_files=1000, total_tokens=10_000_000, sampled_files=50)
        base = UserAnswers(document_stats=stats)
        base.sizing_profile = SizingProfile(name="cpu-balanced", vector_dtype="fp32", hnsw_m=16)
        half = UserAnswers(document_stats=stats)
        half.sizing_profile = SizingProfile(name="gpu-fp16", vector_dtype="fp16", hnsw_m=16)
        cost = CostEstimator()
        fp32 = cost.estimate(base, self._base_recs())
        fp16 = cost.estimate(half, self._base_recs())
        assert fp16.index_memory_mb < fp32.index_memory_mb
        assert 0.45 < fp16.index_memory_mb / fp32.index_memory_mb < 0.55

    def test_on_disk_mmap_keeps_only_hnsw_graph_in_ram(self) -> None:
        from rag_adviser.models import DocumentStats, SizingProfile
        from rag_adviser.recommenders.cost_estimator import CostEstimator

        stats = DocumentStats(total_files=1000, total_tokens=10_000_000, sampled_files=50)
        base = UserAnswers(document_stats=stats)
        base.sizing_profile = SizingProfile(name="cpu-balanced", vector_dtype="fp32", hnsw_m=16)
        mmap = UserAnswers(document_stats=stats)
        mmap.sizing_profile = SizingProfile(
            name="on-disk-mmap", vector_dtype="fp32", hnsw_m=16, on_disk_vectors=True
        )
        cost = CostEstimator()
        in_memory = cost.estimate(base, self._base_recs())
        on_disk = cost.estimate(mmap, self._base_recs())
        # HNSW graph is ~1.26x raw vectors at M=16, so on-disk mmap
        # saves ~44% of RAM (raw vectors gone, graph stays resident) —
        # not a full 50% but the meaningful direction is unambiguous.
        assert on_disk.index_memory_mb < in_memory.index_memory_mb * 0.6
        assert on_disk.index_memory_mb > 0

    def test_sizing_preset_loads_from_yaml(self) -> None:
        from rag_adviser.presets.manager import (
            list_sizing_preset_names,
            load_sizing_preset,
        )

        names = list_sizing_preset_names()
        for expected in ("cpu-balanced", "gpu-fp16", "gpu-fp16-quantized",
                         "on-disk-mmap"):
            assert expected in names, f"missing sizing preset {expected}"
        fp16 = load_sizing_preset("gpu-fp16")
        assert fp16.vector_dtype == "fp16"
        assert fp16.hnsw_m == 32
        assert fp16.quantization == "none"
        quantized = load_sizing_preset("gpu-fp16-quantized")
        assert quantized.quantization == "scalar"


class TestSamplingRegressions:
    """0.5.0 Phase 5: sampling scales with corpus, stratifies by
    extension, exposes percentile stats and a CI on the scanned-PDF rate."""

    def test_sample_scales_with_corpus_size(self, tmp_path: Path) -> None:
        # 400 text files → target 5% = 20 falls under the floor, so the
        # sample size clamps to 50. 2000 files → 100; 20000 → capped 500.
        from rag_adviser.analyzers.document_analyzer import _target_sample_size
        assert _target_sample_size(10) == 10
        assert _target_sample_size(400) == 50
        assert _target_sample_size(2000) == 100
        assert _target_sample_size(20000) == 500

    def test_stratified_sample_covers_every_extension(self, tmp_path: Path) -> None:
        # 200 .txt + 40 .md → target sample = max(50, 5% of 240) = 50.
        # Both extensions must contribute (previously a sorted-order
        # fixed cap of 50 could miss the smaller cohort entirely).
        for i in range(200):
            (tmp_path / f"doc_{i:03d}.txt").write_text(PROSE, encoding="utf-8")
        for i in range(40):
            (tmp_path / f"note_{i:03d}.md").write_text(PROSE, encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.total_files == 240
        assert 40 <= stats.sampled_files <= 60

    def test_token_percentiles_populated(self, tmp_path: Path) -> None:
        # Heavy-tailed distribution: 60 short docs, 4 long ones.
        for i in range(60):
            (tmp_path / f"short_{i:02d}.txt").write_text(PROSE, encoding="utf-8")
        for i in range(4):
            (tmp_path / f"long_{i}.txt").write_text(PROSE * 20, encoding="utf-8")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.tokens_p50 > 0
        assert stats.tokens_p50 <= stats.tokens_p75 <= stats.tokens_p90
        assert stats.tokens_p90 <= stats.tokens_p95 <= stats.tokens_p99
        # Long docs pull p99 well above p50 (heavy tail visible).
        assert stats.tokens_p99 > stats.tokens_p50 * 3

    def test_scanned_pdf_rate_has_wilson_ci(self, tmp_path: Path) -> None:
        from pypdf import PdfWriter

        # 3 blank-page (scanned-looking) PDFs and 1 text-bearing one.
        for i in range(3):
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with open(tmp_path / f"scan_{i}.pdf", "wb") as f:
                writer.write(f)
        (tmp_path / "prose.txt").write_text(PROSE, encoding="utf-8")

        stats = DocumentAnalyzer().analyze(tmp_path)
        low, high = stats.scanned_pdf_rate_ci
        # 3 of 3 scanned → point estimate 100%, but the interval must
        # be wider than the point estimate given the tiny sample.
        assert 0.0 <= low <= high <= 1.0
        assert low < 1.0  # tiny sample must reflect uncertainty
        assert stats.scanned_pdfs == 3


class TestRetrievalDefaultRegressions:
    """0.3.3: RetrievalRecommendation used to lowercase queries and strip
    punctuation by default. That is wrong for transformer bi-encoders and
    was invisible in the human-readable report."""

    def test_query_preprocessing_default_is_empty(self) -> None:
        assert RetrievalRecommendation().query_preprocessing == {}

    def test_preprocessing_when_set_surfaces_in_markdown(self, tmp_path: Path) -> None:
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "a.txt").write_text(PROSE, encoding="utf-8")
        answers = UserAnswers(
            document_path=corpus,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
        )
        recs = RAGAdviser().run(answers, tmp_path / "out", [ReportFormat.MARKDOWN])
        recs.retrieval.query_preprocessing = {"lowercase": True, "remove_punctuation": False}
        # Re-render to exercise the renderer directly.
        from rag_adviser.reporters.markdown_renderer import MarkdownRenderer
        out = tmp_path / "out2"
        out.mkdir()
        MarkdownRenderer().render(answers, recs, out)
        md = (out / "rag_report.md").read_text("utf-8")
        assert "Query preprocessing:" in md
        assert "lowercase" in md
        assert "remove_punctuation" not in md  # only truthy entries listed
