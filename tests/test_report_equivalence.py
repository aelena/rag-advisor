"""Cross-format semantic equivalence for the four report renderers.

The critical review of 2026-09-16 pointed out that Markdown, HTML, YAML
and JSON silently disagreed about what the recommendation actually was
— the approach decision was missing from the human reports, hardware
constraints were missing from the machine reports, alternatives were
name-only in YAML while richly annotated in Markdown, and so on.

This test asserts that a handful of load-bearing decisions and inputs
appear in every rendered output. It intentionally checks strings rather
than semantic structure: the goal is to catch drift, not to over-fit
to any particular schema. When adding a new "canonical" field to the
report, add it here so every renderer stays honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rag_adviser.models import (
    ApproachAssessment,
    BudgetTier,
    ChunkingRecommendation,
    ContentType,
    CostEstimate,
    DeploymentTarget,
    DocumentStats,
    EmbeddingModelRecommendation,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    QueryType,
    Recommendations,
    RecommendedApproach,
    RetrievalRecommendation,
    UpdateFrequency,
    UseCase,
    UserAnswers,
    VectorDBRecommendation,
)
from rag_adviser.reporters.html_renderer import HtmlRenderer
from rag_adviser.reporters.json_renderer import JsonRenderer
from rag_adviser.reporters.markdown_renderer import MarkdownRenderer
from rag_adviser.reporters.yaml_renderer import YamlRenderer


def _rich_answers() -> UserAnswers:
    return UserAnswers(
        document_stats=DocumentStats(
            total_files=42,
            total_size_bytes=10_000_000,
            file_types={".pdf": 30, ".txt": 12},
            languages_detected={"en": 0.9, "es": 0.1},
            primary_language="en",
            has_cjk=False,
            avg_tokens_per_doc=1_500.0,
            total_tokens=63_000,
            detected_content_type=ContentType.PROSE,
        ),
        use_case=UseCase.QA,
        constraints=HardwareConstraints(
            environment=DeploymentTarget.LOCAL,
            latency_budget=LatencyBudget.MODERATE,
            hardware=HardwareProfile.CPU_ONLY,
            ram_gb=16.0,
            vram_gb=0.0,
            budget=BudgetTier.FREE,
            privacy=PrivacyLevel.NONE,
        ),
        query_type=QueryType.NATURAL_QUESTIONS,
        update_frequency=UpdateFrequency.NEVER,
        expected_queries_per_day=1000,
    )


def _rich_recs() -> Recommendations:
    top = EmbeddingModelRecommendation(
        model_id="BAAI/bge-m3",
        provider="huggingface",
        dimension=1024,
        max_tokens=8192,
        estimated_size_gb=2.27,
        license="mit",
        multilingual=True,
        score=0.96,
        quality_score=54,
        reasons=["Multilingual", "Long context"],
    )
    alt = EmbeddingModelRecommendation(
        model_id="intfloat/multilingual-e5-large",
        provider="huggingface",
        dimension=1024,
        max_tokens=512,
        estimated_size_gb=2.24,
        license="mit",
        multilingual=True,
        score=0.87,
        quality_score=51,
        reasons=["Broadly benchmarked"],
    )
    return Recommendations(
        approach=ApproachAssessment(
            recommended_approach=RecommendedApproach.RAG,
            confidence=0.9,
            reasoning="Natural-language questions over a document corpus.",
            proceed_with_rag=True,
        ),
        embedding_models=[top, alt],
        chunking=ChunkingRecommendation(
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=50,
            count_by="tokens",
            content_type="prose",
        ),
        vector_db=VectorDBRecommendation(
            provider="Qdrant",
            reason="Handles the estimated chunk count and supports hybrid search",
            category="client-server",
            library="qdrant-client",
            estimated_capacity="~100,000,000 documents",
            supports_hybrid_search=True,
        ),
        retrieval=RetrievalRecommendation(
            top_k=5,
            similarity_threshold=0.75,
            rerank=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        ),
        estimates=CostEstimate(
            corpus_tokens=63_000,
            chunk_count=137,
            tokens_to_embed=70_144,
            index_size_mb=1.5,
            index_memory_mb=1.2,
            indexing_time_min=4.0,
            query_latency_ms=850,
            fits_latency_budget=True,
            latency_budget_ms=2000,
        ),
    )


def _render_all(tmp: Path) -> dict[str, str]:
    tmp.mkdir(exist_ok=True)
    answers, recs = _rich_answers(), _rich_recs()
    MarkdownRenderer().render(answers, recs, tmp)
    HtmlRenderer().render(answers, recs, tmp)
    YamlRenderer().render(answers, recs, tmp)
    JsonRenderer().render(answers, recs, tmp)
    return {
        "md": (tmp / "rag_report.md").read_text("utf-8"),
        "html": (tmp / "rag_report.html").read_text("utf-8"),
        "yaml": (tmp / "rag_config.yaml").read_text("utf-8"),
        "json": (tmp / "rag_config.json").read_text("utf-8"),
    }


class TestRendererEquivalence:
    """Every load-bearing decision must appear in every format."""

    def test_approach_decision_in_every_format(self, tmp_path: Path) -> None:
        out = _render_all(tmp_path)
        # Recommended approach + confidence figure (as 0.9 or 90%)
        for fmt in ("md", "html", "yaml", "json"):
            assert "rag" in out[fmt].lower(), f"{fmt} is missing the approach"
        assert "90%" in out["md"] and "90%" in out["html"]
        assert "0.9" in out["yaml"] and "0.9" in out["json"]

    def test_top_embedding_model_in_every_format(self, tmp_path: Path) -> None:
        out = _render_all(tmp_path)
        for fmt, text in out.items():
            assert "bge-m3" in text, f"{fmt} does not name the embedding model"

    def test_embedding_alternative_carries_score_in_machine_output(
        self, tmp_path: Path
    ) -> None:
        out = _render_all(tmp_path)
        # Alternatives table lists the second model in every format.
        for fmt, text in out.items():
            assert "multilingual-e5-large" in text, fmt
        # Machine formats now include the score/quality (used to be name-only).
        yaml_config = yaml.safe_load(out["yaml"])
        alt = yaml_config["embedding"]["alternatives"][0]
        assert alt["model_id"] == "intfloat/multilingual-e5-large"
        assert alt["score"] == 0.87
        assert alt["quality_score"] == 51
        assert alt["reasons"]
        # JSON is a subclass of YAML so must agree.
        json_config = json.loads(out["json"])
        assert (
            json_config["embedding"]["alternatives"][0]["score"] == 0.87
        )

    def test_chunk_size_and_top_k_in_every_format(self, tmp_path: Path) -> None:
        out = _render_all(tmp_path)
        for fmt, text in out.items():
            assert "512" in text, f"{fmt} is missing the chunk size"
        # top_k=5 keyed on the label so we do not false-positive on stray 5s
        assert "Top-K:** 5" in out["md"]
        assert "Top-K:</strong> 5" in out["html"]
        assert "top_k: 5" in out["yaml"]
        assert '"top_k": 5' in out["json"]

    def test_vector_db_reason_now_in_machine_output(self, tmp_path: Path) -> None:
        out = _render_all(tmp_path)
        needle = "Handles the estimated chunk count"
        for fmt, text in out.items():
            assert needle in text, f"{fmt} lost the vector DB reason"

    def test_hardware_constraints_in_machine_output(self, tmp_path: Path) -> None:
        out = _render_all(tmp_path)
        yaml_config = yaml.safe_load(out["yaml"])
        meta = yaml_config["metadata"]
        assert meta["hardware"] == "cpu_only"
        assert meta["ram_gb"] == 16.0
        assert meta["budget"] == "free"
        assert meta["update_frequency"] == "never"
        assert meta["latency_budget"] == "<2s"
        # And still surfaced in the human reports.
        assert "cpu_only" in out["md"] and "cpu_only" in out["html"]

    def test_tokens_to_embed_in_every_format(self, tmp_path: Path) -> None:
        out = _render_all(tmp_path)
        for fmt, text in out.items():
            assert "70,144" in text or "70144" in text, (
                f"{fmt} is missing tokens_to_embed"
            )
