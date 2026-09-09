"""Build UserAnswers from CLI flags (non-interactive mode)."""

from __future__ import annotations

from pathlib import Path

from rag_adviser.models import (
    AnswerType,
    BudgetTier,
    ContentType,
    DeploymentTarget,
    HardwareProfile,
    InvalidInputError,
    LatencyBudget,
    PrivacyLevel,
    QueryComplexity,
    QueryType,
    UpdateFrequency,
    UseCase,
    UserAnswers,
)


class CliParamsCollector:
    """Collect and validate UserAnswers from CLI flag values."""

    @staticmethod
    def from_options(
        base: UserAnswers | None = None,
        document_path: Path | None = None,
        future_languages: bool | None = None,
        content_type: str | None = None,
        use_case: str | None = None,
        environment: str | None = None,
        latency: str | None = None,
        hardware: str | None = None,
        ram_gb: float | None = None,
        vram_gb: float | None = None,
        budget: str | None = None,
        privacy: str | None = None,
        embedding_provider: str | None = None,
        llm_provider: str | None = None,
        preferred_libraries: list[str] | None = None,
        query_type: str | None = None,
        query_complexity: str | None = None,
        expected_answer_type: str | None = None,
        sample_queries: list[str] | None = None,
        update_frequency: str | None = None,
        has_ground_truth: bool | None = None,
        ground_truth_path: Path | None = None,
        queries_per_day: int | None = None,
    ) -> UserAnswers:
        """Build UserAnswers from CLI options.

        Unset options keep the value from ``base`` (typically a loaded preset)
        or, when no base is given, the dataclass defaults.
        """
        answers = base if base is not None else UserAnswers()

        # Phase 1: Document Discovery
        if document_path is not None:
            if not document_path.exists():
                raise InvalidInputError(f"Document path does not exist: {document_path}")
            answers.document_path = document_path

        if future_languages is not None:
            answers.future_languages = future_languages

        if content_type is not None:
            try:
                answers.content_type_override = ContentType(content_type)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --content-type: '{content_type}'. "
                    f"Valid: {[ct.value for ct in ContentType]}"
                ) from e

        # Phase 2: Use Case & Constraints
        if use_case is not None:
            try:
                answers.use_case = UseCase(use_case)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --use-case: '{use_case}'. "
                    f"Valid: {[uc.value for uc in UseCase]}"
                ) from e

        hw = answers.constraints

        if environment is not None:
            try:
                hw.environment = DeploymentTarget(environment)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --environment: '{environment}'. "
                    f"Valid: {[d.value for d in DeploymentTarget]}"
                ) from e

        if latency is not None:
            try:
                hw.latency_budget = LatencyBudget(latency)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --latency: '{latency}'. "
                    f"Valid: {[lb.value for lb in LatencyBudget]}"
                ) from e

        if hardware is not None:
            try:
                hw.hardware = HardwareProfile(hardware)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --hardware: '{hardware}'. "
                    f"Valid: {[hp.value for hp in HardwareProfile]}"
                ) from e

        if ram_gb is not None:
            hw.ram_gb = ram_gb
        if vram_gb is not None:
            hw.vram_gb = vram_gb

        if budget is not None:
            try:
                hw.budget = BudgetTier(budget)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --budget: '{budget}'. "
                    f"Valid: {[b.value for b in BudgetTier]}"
                ) from e

        if privacy is not None:
            try:
                hw.privacy = PrivacyLevel(privacy)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --privacy: '{privacy}'. "
                    f"Valid: {[p.value for p in PrivacyLevel]}"
                ) from e

        if embedding_provider is not None:
            answers.embedding_provider = embedding_provider
        if llm_provider is not None:
            answers.llm_provider = llm_provider
        if preferred_libraries:
            answers.preferred_libraries = preferred_libraries

        # Phase 3: Query & Operational Patterns
        if query_type is not None:
            try:
                answers.query_type = QueryType(query_type)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --query-type: '{query_type}'. "
                    f"Valid: {[qt.value for qt in QueryType]}"
                ) from e

        if query_complexity is not None:
            try:
                answers.query_complexity = QueryComplexity(query_complexity)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --query-complexity: '{query_complexity}'. "
                    f"Valid: {[qc.value for qc in QueryComplexity]}"
                ) from e

        if expected_answer_type is not None:
            try:
                answers.expected_answer_type = AnswerType(expected_answer_type)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --answer-type: '{expected_answer_type}'. "
                    f"Valid: {[at.value for at in AnswerType]}"
                ) from e

        if sample_queries:
            answers.sample_queries = sample_queries

        if update_frequency is not None:
            try:
                answers.update_frequency = UpdateFrequency(update_frequency)
            except ValueError as e:
                raise InvalidInputError(
                    f"Invalid --update-frequency: '{update_frequency}'. "
                    f"Valid: {[uf.value for uf in UpdateFrequency]}"
                ) from e

        if has_ground_truth is not None:
            answers.has_ground_truth = has_ground_truth

        if queries_per_day is not None:
            if queries_per_day < 0:
                raise InvalidInputError("--queries-per-day must be >= 0")
            answers.expected_queries_per_day = queries_per_day

        if ground_truth_path is not None:
            if not ground_truth_path.exists():
                raise InvalidInputError(f"Ground truth file not found: {ground_truth_path}")
            answers.ground_truth_path = ground_truth_path
            answers.has_ground_truth = True

        return answers
