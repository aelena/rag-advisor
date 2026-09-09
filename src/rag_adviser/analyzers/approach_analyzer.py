"""Approach analyzer — determines whether RAG is the right solution."""

from __future__ import annotations

from rag_adviser.models import (
    ApproachAssessment,
    ContentType,
    DocumentStats,
    RecommendedApproach,
    UseCase,
    UserAnswers,
)

# Threshold: if the entire corpus fits comfortably in a single LLM context
# window, RAG may be overkill. Current frontier models accept 200K-1M tokens,
# but stuffing that much on every request is slow and expensive, so we use a
# conservative 100K where "just put it all in the prompt" is clearly cheaper
# than running a retrieval stack.
CONTEXT_WINDOW_THRESHOLD = 100_000

# Below this file count, the corpus is trivially small
SMALL_CORPUS_FILES = 10

# Below this token count, direct context stuffing is almost always better
TINY_CORPUS_TOKENS = 50_000


class ApproachAnalyzer:
    """Analyze whether RAG is the right approach for the user's scenario."""

    def assess(self, answers: UserAnswers) -> ApproachAssessment:
        """Evaluate the user's scenario and recommend an approach.

        Returns an ApproachAssessment. If proceed_with_rag is False, the tool
        should present the alternative and optionally continue with RAG anyway.
        """
        stats = answers.document_stats
        use_case = answers.use_case
        content_type = self._effective_content_type(answers)

        # Run checks in priority order — first match wins
        checks = [
            self._check_tabular_sql(content_type, use_case),
            self._check_tiny_corpus(stats),
            self._check_fits_context_window(stats, use_case),
            self._check_single_structured_doc(stats, content_type),
            self._check_keyword_search_sufficient(use_case, answers),
        ]

        for result in checks:
            if result is not None:
                return result

        # Default: RAG is appropriate
        return ApproachAssessment(
            recommended_approach=RecommendedApproach.RAG,
            confidence=0.9,
            reasoning="Your scenario is well-suited for a RAG pipeline.",
            proceed_with_rag=True,
        )

    def _effective_content_type(self, answers: UserAnswers) -> ContentType:
        """Get the effective content type from override or detection."""
        if answers.content_type_override:
            return answers.content_type_override
        if answers.document_stats:
            return answers.document_stats.detected_content_type
        return ContentType.PROSE

    def _check_tabular_sql(
        self, content_type: ContentType, use_case: UseCase
    ) -> ApproachAssessment | None:
        """Tabular data with Q&A is better served by Text-to-SQL."""
        if content_type == ContentType.TABULAR and use_case in (
            UseCase.QA,
            UseCase.SEARCH,
        ):
            return ApproachAssessment(
                recommended_approach=RecommendedApproach.TEXT_TO_SQL,
                confidence=0.85,
                reasoning=(
                    "Your corpus is tabular data and your use case involves querying it. "
                    "RAG over tables is fragile — embedding table rows loses structure. "
                    "A Text-to-SQL approach (e.g., LLM generates SQL/pandas queries) "
                    "will be far more accurate."
                ),
                alternative_description=(
                    "Use an LLM to generate SQL or pandas queries against your structured data. "
                    "Libraries: LangChain SQLDatabaseChain, LlamaIndex NLSQLTableQueryEngine, "
                    "or Vanna.ai for Text-to-SQL."
                ),
                proceed_with_rag=False,
            )
        return None

    def _check_tiny_corpus(
        self, stats: DocumentStats | None
    ) -> ApproachAssessment | None:
        """Very small corpus — just stuff it into the context window."""
        if stats is None or stats.total_files == 0:
            return None

        if stats.total_tokens < TINY_CORPUS_TOKENS and stats.total_files <= SMALL_CORPUS_FILES:
            return ApproachAssessment(
                recommended_approach=RecommendedApproach.DIRECT_CONTEXT,
                confidence=0.90,
                reasoning=(
                    f"Your corpus is very small ({stats.total_files} files, "
                    f"~{stats.total_tokens:,} tokens). It fits entirely in a single "
                    f"LLM context window. RAG adds unnecessary complexity here — "
                    f"just include all documents as context in your prompt."
                ),
                alternative_description=(
                    "Concatenate all documents and pass them directly to an LLM. "
                    "No chunking, no embeddings, no vector DB needed. Any current "
                    "long-context model (200K+ tokens) handles this; use prompt "
                    "caching so the corpus is not re-billed on every request."
                ),
                proceed_with_rag=False,
            )
        return None

    def _check_fits_context_window(
        self, stats: DocumentStats | None, use_case: UseCase
    ) -> ApproachAssessment | None:
        """Corpus fits in context window and use case is summarization."""
        if stats is None or stats.total_files == 0:
            return None

        if (
            stats.total_tokens < CONTEXT_WINDOW_THRESHOLD
            and use_case == UseCase.SUMMARIZATION
        ):
            return ApproachAssessment(
                recommended_approach=RecommendedApproach.LONG_CONTEXT_LLM,
                confidence=0.80,
                reasoning=(
                    f"Your corpus (~{stats.total_tokens:,} tokens) fits within a long-context "
                    f"LLM's window, and your primary use case is summarization. "
                    f"A long-context LLM can process all documents at once for better "
                    f"summaries than RAG's chunk-by-chunk approach."
                ),
                alternative_description=(
                    "Use a long-context LLM (200K-1M token windows are standard) "
                    "and pass the full corpus with prompt caching. For summarization, "
                    "this gives better coherence than retrieving and summarizing "
                    "individual chunks."
                ),
                proceed_with_rag=False,
            )
        return None

    def _check_single_structured_doc(
        self, stats: DocumentStats | None, content_type: ContentType
    ) -> ApproachAssessment | None:
        """Single large structured document — better parsed than chunked."""
        if stats is None:
            return None

        if (
            stats.total_files == 1
            and content_type in (ContentType.LEGAL, ContentType.SCIENTIFIC)
            and stats.total_tokens < CONTEXT_WINDOW_THRESHOLD
        ):
            return ApproachAssessment(
                recommended_approach=RecommendedApproach.STRUCTURED_EXTRACTION,
                confidence=0.70,
                reasoning=(
                    "You have a single structured document (legal/scientific). "
                    "Rather than chunking it for RAG, structured extraction "
                    "(parsing sections, clauses, headings) gives more precise access "
                    "to specific parts."
                ),
                alternative_description=(
                    "Parse the document into its natural structure (sections, clauses, "
                    "headings) and query against that structure. Libraries: "
                    "docling, unstructured.io, or custom parsers. If the document "
                    "fits in context, you can also use a long-context LLM directly."
                ),
                proceed_with_rag=False,
            )
        return None

    def _check_keyword_search_sufficient(
        self, use_case: UseCase, answers: UserAnswers
    ) -> ApproachAssessment | None:
        """Pure search with keyword queries — full-text search may suffice."""
        from rag_adviser.models import QueryType

        if (
            use_case == UseCase.SEARCH
            and answers.query_type == QueryType.SHORT_KEYWORDS
        ):
            return ApproachAssessment(
                recommended_approach=RecommendedApproach.FULL_TEXT_SEARCH,
                confidence=0.65,
                reasoning=(
                    "Your use case is search with short keyword queries. "
                    "Full-text search (BM25) is simpler, faster, and often more "
                    "accurate for keyword-style queries than semantic search. "
                    "Consider using RAG only if you need semantic understanding."
                ),
                alternative_description=(
                    "Use full-text search: Elasticsearch, OpenSearch, SQLite FTS5, "
                    "or Typesense. These handle keyword queries with proven ranking "
                    "algorithms (BM25/TF-IDF). Add semantic search later if needed."
                ),
                proceed_with_rag=False,
            )
        return None
