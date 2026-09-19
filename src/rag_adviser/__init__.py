"""RAG Configuration Adviser — rule-based recommendations for building RAG systems.

Three consumption surfaces:

- **CLI** — ``ragadvisor …`` (Typer entry point, see ``pyproject.toml``).
- **Claude Code skill** — ``/rag-advisor <folder>`` in Claude Code, defined
  under ``.claude/skills/rag-advisor/``.
- **Python API** — this package. The load-bearing types are re-exported
  from the top level so downstream callers do not need to know the
  sub-module layout:

  .. code-block:: python

      from rag_adviser import (
          RAGAdviser,
          UserAnswers, HardwareConstraints,
          UseCase, LatencyBudget, HardwareProfile, ReportFormat,
      )

      recs = RAGAdviser().run(
          UserAnswers(
              document_path=Path("./corpus"),
              use_case=UseCase.QA,
              constraints=HardwareConstraints(ram_gb=32),
          ),
          output_dir=Path("./out"),
          formats=[ReportFormat.MARKDOWN, ReportFormat.YAML],
      )
      print(recs.embedding_models[0].model_id)
      print(recs.estimates.index_memory_mb)

Everything returned by ``RAGAdviser.run`` is a plain dataclass in
``rag_adviser.models`` — inspect it programmatically, serialise it,
diff it in tests. No framework lock-in.
"""

from rag_adviser.main import RAGAdviser
from rag_adviser.models import (
    AnswerType,
    ApproachAssessment,
    ApproachCandidate,
    BudgetTier,
    ChunkingRecommendation,
    CitationGranularity,
    ContentType,
    CostEstimate,
    DeploymentTarget,
    DocumentAnalysisError,
    DocumentStats,
    EmbeddingModelRecommendation,
    ErrorCost,
    HardwareConstraints,
    HardwareProfile,
    InvalidInputError,
    LatencyBudget,
    LLMVerification,
    ModalityRecommendation,
    ModelFinderError,
    PrivacyLevel,
    QueryComplexity,
    QueryTransformationRecommendation,
    QueryType,
    RagAdvisorError,
    Recommendations,
    RecommendedApproach,
    ReportFormat,
    ReportGenerationError,
    RerankerRecommendation,
    RetrievalRecommendation,
    SizingProfile,
    UpdateFrequency,
    UseCase,
    UserAnswers,
    ValidationResult,
    VectorDBRecommendation,
)

__version__ = "0.6.2"

__all__ = [
    # Version
    "__version__",
    # Top-level runner
    "RAGAdviser",
    # Enums
    "AnswerType",
    "BudgetTier",
    "CitationGranularity",
    "ContentType",
    "DeploymentTarget",
    "ErrorCost",
    "HardwareProfile",
    "LatencyBudget",
    "PrivacyLevel",
    "QueryComplexity",
    "QueryType",
    "RecommendedApproach",
    "ReportFormat",
    "UpdateFrequency",
    "UseCase",
    # Input dataclasses
    "DocumentStats",
    "HardwareConstraints",
    "SizingProfile",
    "UserAnswers",
    # Output dataclasses
    "ApproachAssessment",
    "ApproachCandidate",
    "ChunkingRecommendation",
    "CostEstimate",
    "EmbeddingModelRecommendation",
    "LLMVerification",
    "ModalityRecommendation",
    "QueryTransformationRecommendation",
    "Recommendations",
    "RerankerRecommendation",
    "RetrievalRecommendation",
    "ValidationResult",
    "VectorDBRecommendation",
    # Errors
    "DocumentAnalysisError",
    "InvalidInputError",
    "ModelFinderError",
    "RagAdvisorError",
    "ReportGenerationError",
]
