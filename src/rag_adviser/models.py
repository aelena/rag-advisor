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
    # Everything found in the corpus, including non-text files. ``total_files``
    # counts text-extractable documents only; ``modalities`` counts per kind
    # ("document", "image", "video", "audio", "spreadsheet", "presentation",
    # "cad", "other").
    total_files_all: int = 0
    modalities: dict[str, int] = field(default_factory=dict)
    # PDFs sampled that had no extractable text layer (scans needing OCR).
    sampled_pdfs: int = 0
    scanned_pdfs: int = 0


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
    # Expected traffic; drives the monthly cost and capacity estimates.
    expected_queries_per_day: int = 1000

    # Meta
    use_llm_verification: bool = False
    # Run the recommended configuration against ground_truth_path and report
    # retrieval metrics (requires the [eval] extra).
    run_validation: bool = False
    # How many of the recommended local embedding models --validate compares.
    validate_models: int = 1
    # Extra chunk sizes (tokens) to sweep during --validate; empty = recommended only.
    validate_chunk_sizes: list[int] = field(default_factory=list)


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
    # Model ships custom code on the Hub; sentence-transformers needs
    # trust_remote_code=True to load it.
    trust_remote_code: bool = False
    # Hosted APIs: indicative USD per 1M input tokens. None for local models.
    price_per_million_tokens: float | None = None
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
    # Hybrid (sparse + dense) retrieval
    hybrid_search: bool = False
    sparse_method: str = "bm25"
    fusion_method: str = "rrf"
    hybrid_native: bool = False  # chosen vector DB supports hybrid natively
    hybrid_reasons: list[str] = field(default_factory=list)
    hybrid_code_snippet: str = ""
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
class ModalityRecommendation:
    """Ingestion advice for one non-text modality present in the corpus."""

    # image | video | audio | spreadsheet | presentation | cad | scanned_pdf | other
    modality: str = ""
    file_count: int = 0
    share: float = 0.0              # fraction of all files
    extensions: list[str] = field(default_factory=list)
    strategy: str = ""              # one-line headline
    ingestion: list[str] = field(default_factory=list)
    tools_local: list[str] = field(default_factory=list)
    tools_hosted: list[str] = field(default_factory=list)
    embedding: str = ""
    chunking: str = ""
    pip_packages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    code_snippet: str = ""


@dataclass
class RerankerRecommendation:
    """Recommended reranking stage.

    ``enabled`` False with a non-empty ``model_id`` means "not needed now, but
    this is what to reach for if precision turns out to be a problem".
    """

    enabled: bool = False
    model_id: str = ""
    provider: str = "huggingface"  # or "cohere", "voyage"
    quality_score: float = 0.0
    estimated_size_gb: float = 0.0
    max_tokens: int = 512
    multilingual: bool = False
    license: str = "unknown"
    trust_remote_code: bool = False
    fetch_k: int = 20          # candidates retrieved before reranking
    final_k: int = 5           # candidates kept after reranking
    estimated_latency_ms: int = 0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)        # why rerank at all
    model_reasons: list[str] = field(default_factory=list)  # why this model
    warnings: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    code_snippet: str = ""


@dataclass
class CostEstimate:
    """Order-of-magnitude footprint, latency and cost figures."""

    # Corpus / index
    corpus_tokens: int = 0
    corpus_tokens_estimated: bool = False
    chunk_count: int = 0
    tokens_to_embed: int = 0
    embedding_model: str = ""
    embedding_is_api: bool = False
    index_size_mb: float = 0.0      # vectors + payload (+ BM25) on disk
    index_memory_mb: float = 0.0    # vectors + ANN graph in RAM
    # One-off indexing
    indexing_cost_usd: float = 0.0
    indexing_time_min: float = 0.0
    # Recurring
    monthly_reindex_cost_usd: float = 0.0
    monthly_query_cost_usd: float = 0.0
    queries_per_day: int = 0
    # Latency
    query_latency_ms: int = 0
    query_latency_breakdown_ms: dict[str, int] = field(default_factory=dict)
    latency_budget_ms: int = 0
    fits_latency_budget: bool = True
    # Context
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Outcome of running the recommended configuration against ground truth.

    Produced by ``ragadvisor run --validate``. ``ran`` is False when validation
    could not execute (missing paths, missing optional dependencies, runtime
    error); ``error`` then explains why.
    """

    ran: bool = False
    error: str = ""
    # What was actually evaluated
    strategy: str = ""
    embedding_model: str = ""
    chunk_size_chars: int = 0
    chunk_overlap_chars: int = 0
    top_k: int = 0
    vector_backend: str = ""
    # Retrieval mode that was measured and, when hybrid/rerank were on, the
    # plain dense-only numbers from the same index for comparison.
    retrieval_mode: str = "dense"
    reranker_model: str = ""
    has_baseline: bool = False
    baseline_hit_rate: float = 0.0
    baseline_mrr: float = 0.0
    # When several models were compared: [{model, hit_rate, mrr, recall_at_k}], best first.
    model_comparison: list[dict] = field(default_factory=list)
    # When chunk sizes were swept: [{chunk_size_tokens, chunk_size_chars, hit_rate, mrr}],
    # best first.
    chunk_size_comparison: list[dict] = field(default_factory=list)
    recommended_chunk_size_tokens: int = 0
    best_chunk_size_tokens: int = 0
    # Corpus / query counts
    num_queries: int = 0
    num_chunks: int = 0
    # Metrics (0-1)
    hit_rate: float = 0.0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    # Interpretation
    verdict: str = ""  # "strong" | "acceptable" | "weak"
    suggestions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Recommendations:
    """Complete set of recommendations."""

    approach: ApproachAssessment | None = None
    embedding_models: list[EmbeddingModelRecommendation] = field(default_factory=list)
    chunking: ChunkingRecommendation | None = None
    vector_db: VectorDBRecommendation | None = None
    retrieval: RetrievalRecommendation | None = None
    query_transformation: QueryTransformationRecommendation | None = None
    reranker: RerankerRecommendation | None = None
    modalities: list[ModalityRecommendation] = field(default_factory=list)
    estimates: CostEstimate | None = None
    llm_verification: LLMVerification | None = None
    validation: ValidationResult | None = None
    warnings: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
