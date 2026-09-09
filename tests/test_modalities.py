"""Tests for multimodal corpus inventory and per-modality recommendations."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    BudgetTier,
    HardwareConstraints,
    PrivacyLevel,
    ReportFormat,
    UserAnswers,
)
from rag_adviser.recommenders.modality_recommender import ModalityRecommender

PROSE = "Retrieval augmented generation grounds answers in retrieved evidence. " * 10


def _mixed_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "img").mkdir(parents=True)
    (corpus / "notes.txt").write_text(PROSE, "utf-8")
    (corpus / "readme.md").write_text(PROSE, "utf-8")
    for i in range(4):
        (corpus / "img" / f"photo_{i}.jpg").write_bytes(b"\xff\xd8\xff")
    (corpus / "plan.dwg").write_bytes(b"AC1027")
    (corpus / "budget.xlsx").write_bytes(b"PK")
    (corpus / "talk.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    (corpus / "deck.pptx").write_bytes(b"PK")
    (corpus / "build.pyc").write_bytes(b"junk")   # ignored
    (corpus / "weird.xyz").write_bytes(b"?")      # other
    return corpus


class TestInventory:
    def test_counts_modalities_and_ignores_noise(self, tmp_path: Path) -> None:
        stats = DocumentAnalyzer().analyze(_mixed_corpus(tmp_path))

        assert stats.total_files == 2                     # text documents only
        assert stats.modalities["document"] == 2
        assert stats.modalities["image"] == 4
        assert stats.modalities["cad"] == 1
        assert stats.modalities["spreadsheet"] == 1
        assert stats.modalities["video"] == 1
        assert stats.modalities["presentation"] == 1
        assert stats.modalities["other"] == 1
        assert ".pyc" not in stats.file_types
        assert stats.total_files_all == 11
        # Token statistics still come from the text documents only.
        assert stats.total_tokens > 0

    def test_images_only_corpus_does_not_raise(self, tmp_path: Path) -> None:
        (tmp_path / "a.png").write_bytes(b"\x89PNG")
        (tmp_path / "b.png").write_bytes(b"\x89PNG")
        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.total_files == 0
        assert stats.modalities == {"image": 2}

    def test_scanned_pdf_detected(self, tmp_path: Path) -> None:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(tmp_path / "scan.pdf", "wb") as f:
            writer.write(f)
        (tmp_path / "text.txt").write_text(PROSE, "utf-8")

        stats = DocumentAnalyzer().analyze(tmp_path)
        assert stats.sampled_pdfs == 1
        assert stats.scanned_pdfs == 1


class TestRecommender:
    def test_one_recommendation_per_modality_in_order(self, tmp_path: Path) -> None:
        stats = DocumentAnalyzer().analyze(_mixed_corpus(tmp_path))
        recs = ModalityRecommender().recommend(stats, UserAnswers())
        kinds = [r.modality for r in recs]
        assert kinds == ["image", "spreadsheet", "presentation", "video", "cad", "other"]
        image = recs[0]
        assert image.file_count == 4
        assert ".jpg" in image.extensions
        assert image.tools_local and image.ingestion and image.strategy
        compile(image.code_snippet, "img", "exec")

    def test_strict_privacy_marks_hosted_tools_reference_only(self, tmp_path: Path) -> None:
        stats = DocumentAnalyzer().analyze(_mixed_corpus(tmp_path))
        answers = UserAnswers(constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT))
        recs = ModalityRecommender().recommend(stats, answers)
        video = next(r for r in recs if r.modality == "video")
        assert any("reference only" in n for n in video.notes)
        assert any("CPU" in w for w in video.warnings)  # no GPU declared

    def test_paid_api_uses_multimodal_embeddings_for_images(self, tmp_path: Path) -> None:
        stats = DocumentAnalyzer().analyze(_mixed_corpus(tmp_path))
        answers = UserAnswers(
            constraints=HardwareConstraints(budget=BudgetTier.PAID_API, privacy=PrivacyLevel.NONE)
        )
        image = ModalityRecommender().recommend(stats, answers)[0]
        assert "Multimodal" in image.embedding

    def test_scanned_pdf_recommendation_extrapolates(self) -> None:
        from rag_adviser.models import DocumentStats

        stats = DocumentStats(
            total_files=100, total_files_all=100,
            modalities={"document": 100}, file_types={".pdf": 100},
            sampled_pdfs=50, scanned_pdfs=10,
        )
        recs = ModalityRecommender().recommend(stats, UserAnswers())
        assert recs[0].modality == "scanned_pdf"
        assert recs[0].file_count == 20  # 10/50 of 100 PDFs
        assert "docling" in recs[0].pip_packages


class TestEndToEnd:
    def test_pipeline_reports_modalities(self, tmp_path: Path) -> None:
        corpus = _mixed_corpus(tmp_path)
        out = tmp_path / "out"
        answers = UserAnswers(
            document_path=corpus,
            constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
        )
        recs = RAGAdviser().run(answers, out, [ReportFormat.ALL])

        assert len(recs.modalities) == 6
        assert any(w.startswith("MULTIMODAL CORPUS") for w in recs.warnings)
        assert any("pip install" in s and "ezdxf" in s for s in recs.implementation_steps)

        config = yaml.safe_load((out / "rag_config.yaml").read_text("utf-8"))
        assert {m["modality"] for m in config["modalities"]} >= {"image", "cad", "video"}
        assert config["corpus"]["modalities"]["image"] == 4
        assert "## Non-Text Modalities" in (out / "rag_report.md").read_text("utf-8")
        assert "Non-Text Modalities" in (out / "rag_report.html").read_text("utf-8")

    def test_text_only_corpus_has_no_modality_section(self, tmp_path: Path) -> None:
        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "a.txt").write_text(PROSE, "utf-8")
        out = tmp_path / "out"
        recs = RAGAdviser().run(
            UserAnswers(document_path=corpus,
                        constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT)),
            out, [ReportFormat.MARKDOWN],
        )
        assert recs.modalities == []
        assert "Non-Text Modalities" not in (out / "rag_report.md").read_text("utf-8")
