"""RAGAdviser orchestrator — coordinates the full pipeline."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_adviser.analyzers.approach_analyzer import ApproachAnalyzer
from rag_adviser.analyzers.constraint_analyzer import ConstraintAnalyzer
from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.models import (
    AnswerType,
    PrivacyLevel,
    QueryComplexity,
    QueryType,
    Recommendations,
    ReportFormat,
    RetrievalRecommendation,
    UseCase,
    UserAnswers,
)
from rag_adviser.recommenders.chunking_recommender import ChunkingRecommender
from rag_adviser.recommenders.hybrid_recommender import HybridRecommender
from rag_adviser.recommenders.modality_recommender import ModalityRecommender
from rag_adviser.recommenders.model_finder import HFModelFinder
from rag_adviser.recommenders.query_recommender import QueryRecommender
from rag_adviser.recommenders.reranker_recommender import RerankerRecommender
from rag_adviser.recommenders.vector_db_recommender import VectorDBRecommender
from rag_adviser.reporters.report_generator import ReportGenerator


class RAGAdviser:
    """Main orchestrator: analyze -> recommend -> report."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def run(
        self,
        answers: UserAnswers,
        output_dir: Path,
        formats: list[ReportFormat],
    ) -> Recommendations:
        """Execute the full adviser pipeline."""
        recommendations = Recommendations()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            # Step 1: Analyze documents if path provided and not yet analyzed
            if answers.document_path and not answers.document_stats:
                task = progress.add_task("Analyzing document corpus...", total=None)
                analyzer = DocumentAnalyzer()
                answers.document_stats = analyzer.analyze(answers.document_path)
                progress.remove_task(task)

            # Step 1b: Non-text modalities (images, video, spreadsheets, CAD, scans)
            stats = answers.document_stats
            if stats and (
                any(k != "document" for k in stats.modalities) or stats.scanned_pdfs > 0
            ):
                task = progress.add_task("Assessing non-text modalities...", total=None)
                recommendations.modalities = ModalityRecommender().recommend(stats, answers)
                non_text = sum(c for k, c in stats.modalities.items() if k != "document")
                share = non_text / max(stats.total_files_all, 1)
                if share >= 0.3:
                    kinds = ", ".join(
                        f"{c} {k}" for k, c in sorted(stats.modalities.items())
                        if k != "document"
                    )
                    recommendations.warnings.append(
                        f"MULTIMODAL CORPUS: {share:.0%} of files are not text documents "
                        f"({kinds}). The text pipeline below covers only the documents; "
                        f"see the Modalities section for the rest."
                    )
                if stats.scanned_pdfs > 0:
                    recommendations.warnings.append(
                        f"SCANNED PDFs: {stats.scanned_pdfs} of {stats.sampled_pdfs} sampled "
                        f"PDFs have no text layer and need OCR before indexing."
                    )
                progress.remove_task(task)

            # Step 2: Approach assessment
            task = progress.add_task("Assessing approach...", total=None)
            approach_analyzer = ApproachAnalyzer()
            recommendations.approach = approach_analyzer.assess(answers)
            progress.remove_task(task)

            # If approach is non-RAG, add a prominent warning and generate
            # a minimal report with the alternative recommendation
            if not recommendations.approach.proceed_with_rag:
                approach_name = (
                    recommendations.approach.recommended_approach.value
                    .replace("_", " ")
                    .title()
                )
                recommendations.warnings.append(
                    f"APPROACH: RAG is not recommended for your scenario. "
                    f"Suggested alternative: "
                    f"{approach_name}. "
                    f"{recommendations.approach.reasoning}"
                )

            # Step 3: Evaluate constraints and collect warnings
            task = progress.add_task("Evaluating constraints...", total=None)
            constraint_analyzer = ConstraintAnalyzer()
            recommendations.warnings.extend(
                constraint_analyzer.get_warnings(answers.constraints)
            )
            if answers.future_languages:
                recommendations.warnings.append(
                    "FUTURE-LANGUAGE WARNING: You MUST use a multilingual embedding "
                    "model from the start. Adding a language later that is not supported "
                    "by your current model requires re-embedding ALL documents."
                )
            stats = answers.document_stats
            if stats and 0 < stats.sampled_files < stats.total_files:
                recommendations.warnings.append(
                    f"CORPUS ESTIMATE: only {stats.sampled_files} of {stats.total_files} "
                    f"files were opened; token totals (~{stats.total_tokens:,}) are "
                    f"extrapolated from that sample."
                )
            if (
                stats
                and 0 < stats.content_type_confidence < 0.6
                and not answers.content_type_override
            ):
                recommendations.warnings.append(
                    f"CONTENT TYPE: detected {stats.detected_content_type.value} with "
                    f"low confidence ({stats.content_type_confidence:.0%}). Pass "
                    f"--content-type to override if this looks wrong."
                )
            progress.remove_task(task)

            # Step 4: Find embedding models
            task = progress.add_task("Searching for embedding models...", total=None)
            offline = answers.constraints.privacy in (
                PrivacyLevel.STRICT,
                PrivacyLevel.AIR_GAPPED,
            )
            model_finder = HFModelFinder(offline=offline)
            recommendations.embedding_models = model_finder.find_models(
                doc_stats=answers.document_stats,
                constraints=answers.constraints,
                use_case=answers.use_case,
                future_languages=answers.future_languages,
            )
            progress.remove_task(task)

            # Step 5: Chunking recommendation
            task = progress.add_task("Determining chunking strategy...", total=None)
            chunking_rec = ChunkingRecommender()
            # An explicit override always wins; otherwise fall back to what the
            # analyzer detected (if the corpus was analyzed at all).
            content_type = answers.content_type_override
            if content_type is None and answers.document_stats:
                content_type = answers.document_stats.detected_content_type
            top_model = (
                recommendations.embedding_models[0]
                if recommendations.embedding_models
                else None
            )
            languages = (
                list(answers.document_stats.languages_detected.keys())
                if answers.document_stats
                else ["en"]
            )
            recommendations.chunking = chunking_rec.recommend(
                content_type=content_type,
                languages=languages,
                embedding_max_tokens=top_model.max_tokens if top_model else 512,
                use_case=answers.use_case,
            )
            progress.remove_task(task)

            # Step 6: Vector DB recommendation
            task = progress.add_task("Selecting vector database...", total=None)
            db_rec = VectorDBRecommender()
            doc_count = (
                answers.document_stats.total_files if answers.document_stats else 100
            )
            # Estimate chunk count from real token volume when we have it.
            estimated_chunks: int | None = None
            if answers.document_stats and answers.document_stats.total_tokens > 0:
                chunk_tokens = max(recommendations.chunking.chunk_size, 1)
                stride = max(chunk_tokens - recommendations.chunking.chunk_overlap, 1)
                estimated_chunks = max(
                    1, int(answers.document_stats.total_tokens / stride)
                )
            hybrid_rec = HybridRecommender()
            want_hybrid, hybrid_reasons = hybrid_rec.assess(answers, content_type, languages)
            recommendations.vector_db = db_rec.recommend(
                doc_count=doc_count,
                constraints=answers.constraints,
                update_frequency=answers.update_frequency,
                estimated_chunks=estimated_chunks,
                embedding_dimension=top_model.dimension if top_model else 0,
                prefer_hybrid=want_hybrid,
            )
            progress.remove_task(task)

            # Step 7: Retrieval settings (dense defaults + optional BM25 hybrid)
            recommendations.retrieval = self._recommend_retrieval(answers)
            hybrid_rec.apply(
                recommendations.retrieval, want_hybrid, hybrid_reasons, recommendations.vector_db
            )

            # Step 7b: Reranking stage (model choice + latency budget)
            multilingual_needed = (
                answers.future_languages
                or bool(answers.document_stats and answers.document_stats.has_cjk)
                or len(languages) > 1
                or languages[0] != "en"
            )
            recommendations.reranker = RerankerRecommender().recommend(
                answers,
                recommendations.retrieval,
                languages=languages,
                multilingual=multilingual_needed,
                chunk_tokens=recommendations.chunking.chunk_size,
            )
            recommendations.retrieval.rerank = recommendations.reranker.enabled
            recommendations.retrieval.rerank_model = (
                recommendations.reranker.model_id if recommendations.reranker.enabled else None
            )

            # Step 8: Query transformation pipeline
            task = progress.add_task("Designing query pipeline...", total=None)
            query_rec = QueryRecommender()
            recommendations.query_transformation = query_rec.recommend(answers)
            progress.remove_task(task)

            # Step 9: Implementation steps
            recommendations.implementation_steps = self._generate_steps(
                recommendations, answers
            )

        # Step 9b: Validate against the user's ground truth (optional). Runs
        # outside the spinner because the evaluation pipeline drives its own
        # progress display.
        if answers.run_validation:
            from rag_adviser.evaluators.validator import RecommendationValidator

            self.console.print(
                "\n[bold cyan]Validating the recommended configuration "
                "against your ground truth...[/]"
            )
            recommendations.validation = RecommendationValidator(
                console=self.console
            ).validate(answers, recommendations)
            if recommendations.validation.error:
                recommendations.warnings.append(
                    f"VALIDATION SKIPPED: {recommendations.validation.error}"
                )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            # Step 10: LLM verification (optional)
            if answers.use_llm_verification:
                task = progress.add_task("LLM verification...", total=None)
                try:
                    from rag_adviser.llm.client import LLMClient, detect_llm_config
                    from rag_adviser.llm.verifier import RecommendationVerifier

                    llm_config = detect_llm_config()
                    if llm_config:
                        client = LLMClient(llm_config)
                        verifier = RecommendationVerifier(client)
                        recommendations.llm_verification = verifier.verify(
                            answers, recommendations
                        )
                    else:
                        recommendations.warnings.append(
                            "LLM verification requested but no API key found. "
                            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
                        )
                except Exception as e:
                    recommendations.warnings.append(
                        f"LLM verification failed: {e}"
                    )
                progress.remove_task(task)

            # Step 11: Generate reports
            task = progress.add_task("Generating reports...", total=None)
            reporter = ReportGenerator()
            reporter.generate(
                answers=answers,
                recommendations=recommendations,
                output_dir=output_dir,
                formats=formats,
            )
            progress.remove_task(task)

        # Display summary
        self._display_summary(answers, recommendations, output_dir, formats)

        return recommendations

    def _recommend_retrieval(self, answers: UserAnswers) -> RetrievalRecommendation:
        """Generate retrieval settings based on use case and constraints."""
        rec = RetrievalRecommendation()

        # Use-case-specific defaults for top_k, temperature, max_tokens, threshold, strategy
        if answers.use_case == UseCase.QA:
            rec.top_k = 5
            rec.temperature = 0.1
            rec.max_tokens = 500
            rec.similarity_threshold = 0.7
            rec.prompt_strategy = "stuff"
            rec.notes.append("5 chunks balances precision and recall for Q&A")
            rec.notes.append("Low temperature (0.1) keeps answers factual and grounded")
        elif answers.use_case == UseCase.SUMMARIZATION:
            rec.top_k = 10
            rec.temperature = 0.3
            rec.max_tokens = 1500
            rec.similarity_threshold = 0.5
            rec.prompt_strategy = "map_reduce"
            rec.notes.append("More chunks provide broader context for summarization")
            rec.notes.append(
                "map_reduce strategy processes chunks independently then merges; "
                "handles large context well"
            )
        elif answers.use_case == UseCase.CODE:
            rec.top_k = 3
            rec.temperature = 0.0
            rec.max_tokens = 1000
            rec.similarity_threshold = 0.8
            rec.prompt_strategy = "stuff"
            rec.notes.append("Fewer, more precise chunks for code retrieval")
            rec.notes.append(
                "Zero temperature for deterministic code generation"
            )
        elif answers.use_case == UseCase.SEARCH:
            rec.top_k = 10
            rec.temperature = 0.0
            rec.max_tokens = 0
            rec.similarity_threshold = 0.6
            rec.prompt_strategy = "none"
            rec.notes.append("More results for semantic search browsing")
            rec.notes.append(
                "No generation step needed — results returned directly"
            )
        elif answers.use_case == UseCase.LEGAL:
            rec.top_k = 5
            rec.temperature = 0.0
            rec.max_tokens = 1000
            rec.similarity_threshold = 0.75
            rec.prompt_strategy = "refine"
            rec.notes.append(
                "Zero temperature for precise legal language; "
                "refine strategy iterates over chunks to build complete answers"
            )
        else:
            rec.top_k = 5

        # Reranking is decided by RerankerRecommender (see Recommendations.reranker)

        # Query preprocessing
        if answers.query_type == QueryType.SHORT_KEYWORDS:
            rec.notes.append(
                "Query expansion recommended for short keyword queries "
                "(see Query Pipeline section)"
            )
        elif answers.query_type == QueryType.MULTI_TURN:
            rec.notes.append(
                "Conversation condensation required before retrieval "
                "(see Query Pipeline section)"
            )

        # Adjust top_k based on query complexity
        if answers.query_complexity == QueryComplexity.AGGREGATIVE:
            rec.top_k = max(rec.top_k, 15)
            rec.notes.append(
                "Increased top_k to 15 for aggregative queries — "
                "need broad coverage before filtering"
            )
        elif answers.query_complexity == QueryComplexity.MULTI_HOP:
            rec.top_k = max(rec.top_k, 8)
            rec.notes.append(
                "Increased top_k for multi-hop queries — "
                "need to retrieve evidence across multiple documents"
            )

        # Adjust based on expected answer type
        if answers.expected_answer_type == AnswerType.SYNTHESIZED:
            rec.notes.append(
                "Synthesized answers need broader context — "
                "consider map_reduce or refine prompt strategies"
            )
            if rec.prompt_strategy == "stuff":
                rec.prompt_strategy = "refine"
        elif answers.expected_answer_type == AnswerType.EXACT_PASSAGE:
            rec.similarity_threshold = max(rec.similarity_threshold, 0.75)
            rec.notes.append(
                "High similarity threshold for exact passage retrieval — "
                "precision over recall"
            )
        elif answers.expected_answer_type == AnswerType.LIST_ENUMERATION:
            rec.top_k = max(rec.top_k, 10)
            rec.notes.append(
                "Increased top_k for list/enumeration answers — "
                "need to gather items from across the corpus"
            )

        return rec

    def _generate_steps(
        self, recs: Recommendations, answers: UserAnswers
    ) -> list[str]:
        """Generate implementation checklist steps."""
        steps = []
        top = recs.embedding_models[0] if recs.embedding_models else None
        if top and top.provider != "huggingface":
            steps.append(f"Set up API credentials for the {top.provider} embedding endpoint")
        else:
            steps.append("pip install sentence-transformers")
        steps.append("pip install langchain-text-splitters  # for the chunking snippet")

        if recs.chunking and recs.chunking.language_overrides:
            for override in recs.chunking.language_overrides.values():
                tokenizer = override.get("tokenizer", "")
                if tokenizer == "jieba":
                    steps.append("pip install jieba")
                elif tokenizer == "sudachi":
                    steps.append("pip install sudachipy sudachidict_core")
                elif tokenizer == "spacy":
                    model = override.get("sentence_model", "")
                    if model:
                        steps.append(f"python -m spacy download {model}")

        if recs.vector_db and recs.vector_db.library:
            steps.append(f"pip install {recs.vector_db.library}")

        if recs.reranker and recs.reranker.enabled:
            if recs.reranker.provider == "huggingface":
                steps.append(
                    f"pip install sentence-transformers  # cross-encoder reranker "
                    f"{recs.reranker.model_id}"
                )
            else:
                steps.append(
                    f"Set up {recs.reranker.provider} API credentials for the "
                    f"{recs.reranker.model_id} reranker"
                )

        if recs.retrieval and recs.retrieval.hybrid_search and not recs.retrieval.hybrid_native:
            steps.append("pip install rank-bm25  # sparse retriever for hybrid search")

        modality_pkgs: list[str] = []
        for m in recs.modalities:
            for pkg in m.pip_packages:
                if pkg not in modality_pkgs:
                    modality_pkgs.append(pkg)
        if modality_pkgs:
            steps.append(
                f"pip install {' '.join(modality_pkgs)}  # ingestion for "
                f"{', '.join(m.modality for m in recs.modalities)}"
            )

        if recs.validation and recs.validation.ran:
            steps.append(
                f"Validated: hit rate {recs.validation.hit_rate:.0%}, "
                f"MRR {recs.validation.mrr:.2f} ({recs.validation.verdict}); "
                f"see the Validation section"
            )
        else:
            steps.append(
                "Measure before shipping: ragadvisor run --validate "
                "--ground-truth-path <queries.jsonl> (or ragadvisor evaluate)"
            )
        steps.append(
            "If adding languages later: re-embed ALL documents with multilingual model"
        )

        return steps

    def _display_summary(
        self,
        answers: UserAnswers,
        recs: Recommendations,
        output_dir: Path,
        formats: list[ReportFormat],
    ) -> None:
        """Show a Rich panel with the recommendation summary."""
        self.console.print()

        # Approach assessment
        if recs.approach and not recs.approach.proceed_with_rag:
            self.console.print(
                Panel(
                    f"[bold yellow]Recommended approach:[/] "
                    f"{recs.approach.recommended_approach.value.replace('_', ' ').title()}\n"
                    f"{recs.approach.reasoning}\n\n"
                    f"{recs.approach.alternative_description}",
                    title="Alternative Approach Recommended",
                    border_style="yellow",
                )
            )
            self.console.print()

        # Warnings
        if recs.warnings:
            for warning in recs.warnings:
                self.console.print(f"[bold yellow]  Warning:[/] {warning}")
            self.console.print()

        # Summary table
        table = Table(title="RAG Configuration Summary", show_header=True, title_style="bold cyan")
        table.add_column("Component", style="bold")
        table.add_column("Recommendation", style="white")
        table.add_column("Details", style="dim")

        if recs.embedding_models:
            top = recs.embedding_models[0]
            table.add_row(
                "Embedding Model",
                top.model_id,
                f"dim={top.dimension}, max_tokens={top.max_tokens}, "
                f"size={top.estimated_size_gb}GB",
            )

        if recs.chunking:
            table.add_row(
                "Chunking",
                recs.chunking.strategy,
                f"size={recs.chunking.chunk_size}, overlap={recs.chunking.chunk_overlap}, "
                f"by={recs.chunking.count_by}",
            )

        if recs.vector_db:
            table.add_row(
                "Vector DB",
                recs.vector_db.provider,
                recs.vector_db.reason,
            )

        if recs.retrieval:
            table.add_row(
                "Retrieval",
                f"top_k={recs.retrieval.top_k}",
                f"rerank={'Yes' if recs.retrieval.rerank else 'No'}, "
                f"threshold={recs.retrieval.similarity_threshold}",
            )
            if recs.reranker and recs.reranker.enabled:
                table.add_row(
                    "Reranker",
                    recs.reranker.model_id,
                    f"top {recs.reranker.fetch_k} -> {recs.reranker.final_k}, "
                    f"~{recs.reranker.estimated_latency_ms}ms",
                )
            if recs.retrieval.hybrid_search:
                table.add_row(
                    "Hybrid Search",
                    f"{recs.retrieval.sparse_method.upper()} + dense "
                    f"({recs.retrieval.fusion_method.upper()})",
                    "native in vector DB" if recs.retrieval.hybrid_native
                    else "in-process (rank-bm25)",
                )
            if recs.retrieval.prompt_strategy != "none":
                table.add_row(
                    "Generation",
                    f"temperature={recs.retrieval.temperature}",
                    f"max_tokens={recs.retrieval.max_tokens}, "
                    f"strategy={recs.retrieval.prompt_strategy}",
                )

        if recs.modalities:
            table.add_row(
                "Modalities",
                ", ".join(f"{m.file_count} {m.modality}" for m in recs.modalities),
                "; ".join(m.strategy for m in recs.modalities[:2]),
            )

        if recs.query_transformation and recs.query_transformation.techniques:
            techniques = ", ".join(
                t["name"] for t in recs.query_transformation.techniques
            )
            table.add_row(
                "Query Pipeline",
                f"{len(recs.query_transformation.techniques)} technique(s)",
                techniques,
            )

        self.console.print(table)
        self.console.print()

        # LLM verification
        if recs.llm_verification:
            v = recs.llm_verification
            self.console.print(
                Panel(
                    f"[bold]Provider:[/] {v.provider} ({v.model})\n\n"
                    f"{v.summary}\n\n"
                    + (
                        "[bold green]Agreements:[/]\n"
                        + "\n".join(f"  - {a}" for a in v.agreements[:3])
                        + "\n\n"
                        if v.agreements
                        else ""
                    )
                    + (
                        "[bold yellow]Refinements:[/]\n"
                        + "\n".join(f"  - {r}" for r in v.refinements[:3])
                        if v.refinements
                        else ""
                    ),
                    title="LLM Verification",
                    border_style="magenta",
                )
            )
            self.console.print()

        # Validation
        if recs.validation and recs.validation.ran:
            v = recs.validation
            colour = {"strong": "green", "acceptable": "yellow"}.get(v.verdict, "red")
            body = (
                f"[bold]Verdict:[/] [{colour}]{v.verdict}[/]  "
                f"({v.num_queries} queries, {v.num_chunks} chunks, "
                f"{v.strategy} @ {v.chunk_size_chars} chars, {v.embedding_model})\n\n"
                f"Hit rate@{v.top_k}: {v.hit_rate:.1%}   MRR: {v.mrr:.3f}   "
                f"Recall@{v.top_k}: {v.recall_at_k:.1%}   nDCG@{v.top_k}: {v.ndcg_at_k:.3f}"
            )
            if v.suggestions:
                body += "\n\n[bold]Next steps:[/]\n" + "\n".join(f"  - {s}" for s in v.suggestions)
            self.console.print(Panel(body, title="Validation", border_style=colour))
            self.console.print()

        # Output files
        format_names = [f.value for f in formats]
        self.console.print(
            Panel(
                f"[bold green]Reports generated in:[/] {output_dir}\n"
                f"[bold green]Formats:[/] {', '.join(format_names)}",
                title="Output",
                border_style="green",
            )
        )
