"""Core data models for ragadvisor."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

# ── Exceptions ─────────────────────────────────────────────────────────────


class RagAdvisorError(Exception):
    """Base exception for ragadvisor."""


class DocumentAnalysisError(RagAdvisorError):
    """Failed to analyze documents."""


class InvalidInputError(RagAdvisorError):
    """User input validation failed."""


class ReportGenerationError(RagAdvisorError):
    """Report generation failed."""


class ModelFinderError(RagAdvisorError):
    """Failed to find embedding models."""


# ── Enums ──────────────────────────────────────────────────────────────────


class ContentType(enum.Enum):
    """Primary content type of the document corpus."""

    PROSE = "prose"
    CODE = "code"
    LEGAL = "legal"
    CHAT = "chat"
    TABULAR = "tabular"
    SCIENTIFIC = "scientific"
    MIXED = "mixed"


class UseCase(enum.Enum):
    """Primary use case for the RAG system."""

    QA = "question_answering"
    SEARCH = "semantic_search"
    SUMMARIZATION = "summarization"
    CODE = "code_assistance"
    LEGAL = "legal_analysis"


class DeploymentTarget(enum.Enum):
    """Deployment environment."""

    CLOUD = "cloud"
    ON_PREM = "on_prem"
    EDGE = "edge"
    LOCAL = "local"


class LatencyBudget(enum.Enum):
    """Acceptable latency for retrieval."""

    FAST = "<500ms"
    MODERATE = "<2s"
    BATCH = "batch"


class HardwareProfile(enum.Enum):
    """Available hardware."""

    CPU_ONLY = "cpu_only"
    GPU_AVAILABLE = "gpu_available"
    LIMITED_RAM = "limited_ram"


class BudgetTier(enum.Enum):
    """Budget level."""

    FREE = "free"
    PAID_API = "paid_api"
    SELF_HOSTED = "self_hosted"


class PrivacyLevel(enum.Enum):
    """Data privacy requirements."""

    NONE = "none"
    MODERATE = "moderate"
    STRICT = "strict"
    AIR_GAPPED = "air_gapped"


class QueryType(enum.Enum):
    """Expected query patterns."""

    SHORT_KEYWORDS = "short_keywords"
    NATURAL_QUESTIONS = "natural_questions"
    MULTI_TURN = "multi_turn"


class QueryComplexity(enum.Enum):
    """How complex are the expected queries."""

    SIMPLE_FACTUAL = "simple_factual"
    COMPARATIVE = "comparative"
    AGGREGATIVE = "aggregative"
    MULTI_HOP = "multi_hop"


class AnswerType(enum.Enum):
    """What kind of answer the user expects."""

    EXACT_PASSAGE = "exact_passage"
    SYNTHESIZED = "synthesized"
    YES_NO_WITH_EVIDENCE = "yes_no_with_evidence"
    LIST_ENUMERATION = "list_enumeration"


class RecommendedApproach(enum.Enum):
    """The overall approach recommended for the user's scenario."""

    RAG = "rag"
    DIRECT_CONTEXT = "direct_context"
    TEXT_TO_SQL = "text_to_sql"
    FULL_TEXT_SEARCH = "full_text_search"
    STRUCTURED_EXTRACTION = "structured_extraction"
    LONG_CONTEXT_LLM = "long_context_llm"


class UpdateFrequency(enum.Enum):
    """How often documents change."""

    NEVER = "never"
    WEEKLY = "weekly"
    DAILY = "daily"
    REALTIME = "realtime"


class ReportFormat(enum.Enum):
    """Output report format."""

    MARKDOWN = "markdown"
    HTML = "html"
    YAML = "yaml"
    ALL = "all"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class DocumentStats:
    """Statistics about the analyzed document corpus."""

    total_files: int = 0
    total_size_bytes: int = 0
    file_types: dict[str, int] = field(default_factory=dict)
    languages_detected: dict[str, float] = field(default_factory=dict)
    primary_language: str = "en"
    has_cjk: bool = False
    avg_tokens_per_doc: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    avg_sentences_per_doc: float = 0.0
    total_tokens: int = 0
    detected_content_type: ContentType = ContentType.PROSE
    sample_texts: list[str] = field(default_factory=list)
    # How many files were actually opened and tokenized. When smaller than
    # total_files, avg/total token figures are extrapolated estimates.
    sampled_files: int = 0
    content_type_confidence: float = 0.0


@dataclass
class HardwareConstraints:
    """Hardware and deployment constraints."""

    environment: DeploymentTarget = DeploymentTarget.LOCAL
    latency_budget: LatencyBudget = LatencyBudget.MODERATE
    hardware: HardwareProfile = HardwareProfile.CPU_ONLY
    ram_gb: float = 16.0
    vram_gb: float = 0.0
    budget: BudgetTier = BudgetTier.FREE
    privacy: PrivacyLevel = PrivacyLevel.NONE


@dataclass
class UserAnswers:
    """Complete set of answers from the question flow (13 interactive steps)."""

    # Phase 1: Document Discovery
    document_path: Path | None = None
    document_stats: DocumentStats | None = None
    future_languages: bool = False
    content_type_override: ContentType | None = None

    # Phase 2: Use Case & Constraints
    use_case: UseCase = UseCase.QA
    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    embedding_provider: str | None = None  # e.g. "openai", "huggingface", "cohere"
    llm_provider: str | None = None  # e.g. "gpt-4", "llama-3", "none"
    implementation_lang: str = "python"
    preferred_libraries: list[str] = field(default_factory=list)

    # Phase 3: Query & Operational Patterns
    query_type: QueryType = QueryType.NATURAL_QUESTIONS
    query_complexity: QueryComplexity = QueryComplexity.SIMPLE_FACTUAL
    expected_answer_type: AnswerType = AnswerType.EXACT_PASSAGE
    sample_queries: list[str] = field(default_factory=list)
    update_frequency: UpdateFrequency = UpdateFrequency.NEVER
    has_ground_truth: bool = False
    ground_truth_path: Path | None = None

    # Meta
    use_llm_verification: bool = False


@dataclass
class EmbeddingModelRecommendation:
    """A recommended embedding model."""

    model_id: str = ""
    likes: int = 0
    downloads: int = 0
    tags: list[str] = field(default_factory=list)
    dimension: int = 0
    max_tokens: int = 512
    estimated_size_gb: float = 0.5
    license: str = "unknown"
    release_date: str = "unknown"
    multilingual: bool = False
    # "huggingface" for self-hostable open models, or an API vendor
    # ("openai", "cohere", "voyage") for hosted embedding endpoints.
    provider: str = "huggingface"
    # Approximate MTEB retrieval quality (nDCG@10, 0-100). 0 = unknown.
    quality_score: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkingRecommendation:
    """Recommended chunking strategy."""

    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    count_by: str = "tokens"
    content_type: str = "prose"
    language_overrides: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    code_snippet: str = ""


@dataclass
class VectorDBRecommendation:
    """Recommended vector database."""

    provider: str = "chroma"
    reason: str = ""
    category: str = "embedded"
    supports_metadata_filter: bool = True
    supports_hybrid_search: bool = False
    estimated_capacity: str = ""
    library: str = ""
    code_snippet: str = ""


@dataclass
class RetrievalRecommendation:
    """Recommended retrieval settings."""

    top_k: int = 5
    rerank: bool = False
    rerank_model: str | None = None
    similarity_threshold: float = 0.7
    temperature: float = 0.1
    max_tokens: int = 500
    prompt_strategy: str = "stuff"
    query_preprocessing: dict[str, bool] = field(
        default_factory=lambda: {"lowercase": True, "remove_punctuation": True}
    )
    notes: list[str] = field(default_factory=list)


@dataclass
class ApproachAssessment:
    """Assessment of whether RAG is the right approach."""

    recommended_approach: RecommendedApproach = RecommendedApproach.RAG
    confidence: float = 1.0
    reasoning: str = ""
    alternative_description: str = ""
    proceed_with_rag: bool = True


@dataclass
class QueryTransformationRecommendation:
    """Recommended query transformation techniques."""

    techniques: list[dict[str, str]] = field(default_factory=list)
    code_snippet: str = ""
    latency_impact_ms: int = 0
    requires_llm: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class LLMVerification:
    """Result of LLM verification of recommendations."""

    provider: str = ""
    model: str = ""
    summary: str = ""
    agreements: list[str] = field(default_factory=list)
    refinements: list[str] = field(default_factory=list)
    additional_considerations: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class Recommendations:
    """Complete set of recommendations."""

    approach: ApproachAssessment | None = None
    embedding_models: list[EmbeddingModelRecommendation] = field(default_factory=list)
    chunking: ChunkingRecommendation | None = None
    vector_db: VectorDBRecommendation | None = None
    retrieval: RetrievalRecommendation | None = None
    query_transformation: QueryTransformationRecommendation | None = None
    llm_verification: LLMVerification | None = None
    warnings: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
