"""Shared test fixtures for ragadvisor."""

from __future__ import annotations

import pytest

from rag_adviser.models import (
    BudgetTier,
    ChunkingRecommendation,
    ContentType,
    DeploymentTarget,
    DocumentStats,
    EmbeddingModelRecommendation,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    QueryType,
    Recommendations,
    UpdateFrequency,
    UseCase,
    UserAnswers,
    VectorDBRecommendation,
)


@pytest.fixture
def sample_doc_stats() -> DocumentStats:
    """Typical English prose corpus stats."""
    return DocumentStats(
        total_files=100,
        total_size_bytes=50_000_000,
        file_types={".txt": 60, ".pdf": 30, ".docx": 10},
        languages_detected={"en": 0.85, "fr": 0.15},
        primary_language="en",
        has_cjk=False,
        avg_tokens_per_doc=1200.0,
        max_tokens=5000,
        min_tokens=50,
        avg_sentences_per_doc=45.0,
        total_tokens=120_000,
        detected_content_type=ContentType.PROSE,
        sample_texts=[
            "This is a sample document about artificial intelligence and machine learning.",
            "Another document discussing natural language processing techniques.",
        ],
    )


@pytest.fixture
def multilingual_doc_stats() -> DocumentStats:
    """Multilingual corpus with CJK content."""
    return DocumentStats(
        total_files=200,
        total_size_bytes=80_000_000,
        file_types={".txt": 100, ".pdf": 80, ".docx": 20},
        languages_detected={"en": 0.45, "es": 0.30, "zh": 0.25},
        primary_language="en",
        has_cjk=True,
        avg_tokens_per_doc=900.0,
        max_tokens=4000,
        min_tokens=30,
        avg_sentences_per_doc=35.0,
        total_tokens=180_000,
        detected_content_type=ContentType.PROSE,
        sample_texts=[
            "This is a sample English document about artificial intelligence.",
            "Este es un documento de ejemplo en espanol sobre tecnologia.",
        ],
    )


@pytest.fixture
def default_constraints() -> HardwareConstraints:
    """Typical local development constraints."""
    return HardwareConstraints(
        environment=DeploymentTarget.LOCAL,
        latency_budget=LatencyBudget.MODERATE,
        hardware=HardwareProfile.CPU_ONLY,
        ram_gb=16.0,
        vram_gb=0.0,
        budget=BudgetTier.FREE,
        privacy=PrivacyLevel.NONE,
    )


@pytest.fixture
def gpu_constraints() -> HardwareConstraints:
    """GPU-available constraints."""
    return HardwareConstraints(
        environment=DeploymentTarget.ON_PREM,
        latency_budget=LatencyBudget.FAST,
        hardware=HardwareProfile.GPU_AVAILABLE,
        ram_gb=32.0,
        vram_gb=8.0,
        budget=BudgetTier.SELF_HOSTED,
        privacy=PrivacyLevel.STRICT,
    )


@pytest.fixture
def sample_user_answers(sample_doc_stats, default_constraints) -> UserAnswers:
    """Complete set of user answers for testing."""
    return UserAnswers(
        document_stats=sample_doc_stats,
        future_languages=False,
        use_case=UseCase.QA,
        constraints=default_constraints,
        implementation_lang="python",
        query_type=QueryType.NATURAL_QUESTIONS,
        update_frequency=UpdateFrequency.NEVER,
        has_ground_truth=False,
    )


@pytest.fixture
def sample_embedding_model() -> EmbeddingModelRecommendation:
    """A sample embedding model recommendation."""
    return EmbeddingModelRecommendation(
        model_id="BAAI/bge-small-en-v1.5",
        likes=2000,
        downloads=500_000,
        tags=["sentence-similarity", "feature-extraction", "en"],
        dimension=384,
        max_tokens=512,
        estimated_size_gb=0.13,
        license="mit",
        release_date="2023-09-12",
        multilingual=False,
        score=0.85,
        reasons=["Good MTEB scores", "Small and fast"],
        warnings=[],
    )


@pytest.fixture
def sample_recommendations(sample_embedding_model) -> Recommendations:
    """Complete recommendations for report testing."""
    return Recommendations(
        embedding_models=[
            sample_embedding_model,
            EmbeddingModelRecommendation(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                likes=5000,
                downloads=1_000_000,
                dimension=384,
                max_tokens=256,
                estimated_size_gb=0.08,
                license="apache-2.0",
                multilingual=False,
                score=0.75,
                reasons=["Very fast", "Tiny footprint"],
            ),
        ],
        chunking=ChunkingRecommendation(
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=50,
            count_by="tokens",
            content_type="prose",
            notes=["Standard recursive splitting for English prose"],
        ),
        vector_db=VectorDBRecommendation(
            provider="ChromaDB",
            reason="Simple, embedded, no server needed",
            category="embedded",
            library="chromadb",
        ),
        warnings=["Consider multilingual model if adding languages later"],
    )
