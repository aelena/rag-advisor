"""Approach analyzer — determines whether RAG is the right solution."""

from __future__ import annotations

from rag_adviser.models import (
    ApproachAssessment,
    ApproachCandidate,
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

        Returns the highest-scored candidate as the ``ApproachAssessment``,
        with every other candidate the tool considered attached to
        ``candidates`` so the reader can see the whole comparison. This
        is the "approach as first-class comparator" from review §2 —
        the tool no longer decides on the first rule that matches; it
        scores every applicable rule and picks the strongest.
        """
        candidates = self.assess_all(answers)
        top = candidates[0]
        assessment = ApproachAssessment(
            recommended_approach=top.approach,
            confidence=top.confidence,
            reasoning=top.reasoning,
            evidence=list(top.evidence),
            proceed_with_rag=(top.approach == RecommendedApproach.RAG),
            candidates=candidates,
        )
        # Preserve back-compat: the pre-comparator branches populated
        # ``alternative_description`` with a hand-written narrative. If
        # a specific rule fired, restore that description via the same
        # helper (idempotent — running the check again is cheap).
        if not assessment.proceed_with_rag:
            legacy = self._legacy_alternative_description(top.approach)
            assessment.alternative_description = legacy
        return assessment

    def assess_all(self, answers: UserAnswers) -> list[ApproachCandidate]:
        """Score every candidate approach and return them ranked."""
        stats = answers.document_stats
        use_case = answers.use_case
        content_type = self._effective_content_type(answers)

        candidates: list[ApproachCandidate] = []
        for check in (
            self._check_tabular_sql(content_type, use_case),
            self._check_tiny_corpus(stats),
            self._check_fits_context_window(stats, use_case),
            self._check_single_structured_doc(stats, content_type),
            self._check_keyword_search_sufficient(use_case, answers),
        ):
            if check is not None:
                candidates.append(
                    ApproachCandidate(
                        approach=check.recommended_approach,
                        confidence=check.confidence,
                        reasoning=check.reasoning,
                        evidence=list(check.evidence),
                    )
                )
        # RAG is always a candidate; its confidence is signal-derived so
        # it competes with the specific rules honestly.
        rag = self._default_rag_assessment(answers, content_type)
        candidates.append(
            ApproachCandidate(
                approach=rag.recommended_approach,
                confidence=rag.confidence,
                reasoning=rag.reasoning,
                evidence=list(rag.evidence),
            )
        )
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    _LEGACY_ALT = {
        RecommendedApproach.TEXT_TO_SQL: (
            "Use an LLM to generate SQL or pandas queries against your structured data. "
            "Libraries: LangChain SQLDatabaseChain, LlamaIndex NLSQLTableQueryEngine, "
            "or Vanna.ai for Text-to-SQL."
        ),
        RecommendedApproach.DIRECT_CONTEXT: (
            "Concatenate all documents and pass them directly to an LLM. "
            "No chunking, no embeddings, no vector DB needed. Any current "
            "long-context model (200K+ tokens) handles this; use prompt "
            "caching so the corpus is not re-billed on every request."
        ),
        RecommendedApproach.LONG_CONTEXT_LLM: (
            "Use a long-context LLM (200K-1M token windows are standard) "
            "and pass the full corpus with prompt caching. For summarization, "
            "this gives better coherence than retrieving and summarizing "
            "individual chunks."
        ),
        RecommendedApproach.STRUCTURED_EXTRACTION: (
            "Parse the document into its natural structure (sections, clauses, "
            "headings) and query against that structure. Libraries: "
            "docling, unstructured.io, or custom parsers. If the document "
            "fits in context, you can also use a long-context LLM directly."
        ),
        RecommendedApproach.FULL_TEXT_SEARCH: (
            "Use full-text search: Elasticsearch, OpenSearch, SQLite FTS5, "
            "or Typesense. These handle keyword queries with proven ranking "
            "algorithms (BM25/TF-IDF). Add semantic search later if needed."
        ),
    }

    def _legacy_alternative_description(self, approach: RecommendedApproach) -> str:
        return self._LEGACY_ALT.get(approach, "")

    def _default_rag_assessment(
        self, answers: UserAnswers, content_type: ContentType
    ) -> ApproachAssessment:
        stats = answers.document_stats
        use_case = answers.use_case
        evidence: list[str] = []
        signals_total = 5
        signals_matched = 0

        if stats and stats.total_files > SMALL_CORPUS_FILES:
            signals_matched += 1
            evidence.append(
                f"Corpus has {stats.total_files:,} files — more than direct-context "
                f"stuffing (>{SMALL_CORPUS_FILES}) can handle cleanly"
            )
        if stats and stats.total_tokens > CONTEXT_WINDOW_THRESHOLD:
            signals_matched += 1
            evidence.append(
                f"Corpus tokens (~{stats.total_tokens:,}) exceed the "
                f"{CONTEXT_WINDOW_THRESHOLD:,}-token direct-context threshold"
            )
        if use_case in (
            UseCase.QA,
            UseCase.SEARCH,
            UseCase.SUMMARIZATION,
            UseCase.LEGAL,
            UseCase.CODE,
        ):
            signals_matched += 1
            evidence.append(
                f"Use case '{use_case.value}' is well-served by retrieve-then-generate"
            )
        if content_type != ContentType.TABULAR:
            signals_matched += 1
            evidence.append(
                f"Content type '{content_type.value}' is prose-like (not tabular)"
            )
        if stats is None or stats.total_files > 1:
            signals_matched += 1
            evidence.append(
                "Multiple documents to retrieve from"
                if stats and stats.total_files > 1
                else "No corpus analysed — assessment is based on user intent alone"
            )

        # RAG confidence ranges [0.50, 0.85]. The 0.85 ceiling matters:
        # specific-rule confidences (0.85–0.95) must always beat "all
        # RAG signals matched" when they fire, otherwise the comparator
        # regresses to first-match-wins semantics with extra steps.
        # 0.5 floor: even a zero-signal RAG default is not a rejection,
        # it just means the tool is running blind.
        confidence = round(0.5 + 0.35 * (signals_matched / signals_total), 2)
        reasoning = (
            f"Your scenario matches {signals_matched}/{signals_total} typical RAG "
            "signals — starting point, not a measurement. Run `--validate` on "
            "your own queries before committing to indexing."
        )
        return ApproachAssessment(
            recommended_approach=RecommendedApproach.RAG,
            confidence=confidence,
            reasoning=reasoning,
            proceed_with_rag=True,
            evidence=evidence,
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
                confidence=0.90,
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
                confidence=0.92,
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
                confidence=0.88,
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
                confidence=0.86,
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
                confidence=0.86,
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
