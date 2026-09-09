"""Markdown audit report renderer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rag_adviser import __version__
from rag_adviser.models import Recommendations, UserAnswers


class MarkdownRenderer:
    """Generate a full audit report in Markdown format."""

    def render(
        self,
        answers: UserAnswers,
        recommendations: Recommendations,
        output_dir: Path,
    ) -> Path:
        """Render the Markdown report and write to disk."""
        lines = self._build_report(answers, recommendations)
        out_path = output_dir / "rag_report.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _build_report(
        self, answers: UserAnswers, recs: Recommendations
    ) -> list[str]:
        """Build the full report as a list of lines."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines: list[str] = []

        # Title
        lines.append("# RAG Configuration Audit Report")
        lines.append("")
        lines.append(f"**Generated:** {now}  ")
        lines.append(f"**Adviser Version:** {__version__}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Approach Assessment
        lines.extend(self._section_approach(recs))

        # Warnings
        lines.append("## Warnings & Critical Decisions")
        lines.append("")
        if recs.warnings:
            for w in recs.warnings:
                lines.append(f"- **Warning:** {w}")
        else:
            lines.append("- No critical warnings")
        lines.append("")
        lines.append("---")
        lines.append("")

        # User Input Summary
        lines.extend(self._section_user_input(answers))

        # Non-text modalities
        lines.extend(self._section_modalities(recs))

        # Embedding Models
        lines.extend(self._section_embedding_models(answers, recs))

        # Chunking
        lines.extend(self._section_chunking(recs))

        # Vector DB
        lines.extend(self._section_vector_db(recs))

        # Retrieval
        lines.extend(self._section_retrieval(recs))

        # Reranking
        lines.extend(self._section_reranker(recs))

        # Query Pipeline
        lines.extend(self._section_query_pipeline(recs))

        # Cost, footprint and latency estimates
        lines.extend(self._section_estimates(recs))

        # Validation against ground truth
        lines.extend(self._section_validation(recs))

        # LLM Verification
        lines.extend(self._section_llm_verification(recs))

        # Implementation steps
        lines.extend(self._section_implementation(recs))

        # Re-indexing warning
        lines.extend(self._section_reindexing_warning(answers, recs))

        return lines

    def _section_user_input(self, answers: UserAnswers) -> list[str]:
        """Render the user input audit trail section."""
        lines = ["## User Input Summary", ""]

        lines.append(f"- **Use case:** {answers.use_case.value}")
        lines.append(f"- **Deployment:** {answers.constraints.environment.value}")
        lines.append(f"- **Hardware:** {answers.constraints.hardware.value}")
        lines.append(f"- **RAM:** {answers.constraints.ram_gb}GB")
        if answers.constraints.vram_gb > 0:
            lines.append(f"- **VRAM:** {answers.constraints.vram_gb}GB")
        lines.append(f"- **Latency budget:** {answers.constraints.latency_budget.value}")
        lines.append(f"- **Privacy:** {answers.constraints.privacy.value}")
        lines.append(f"- **Budget:** {answers.constraints.budget.value}")
        lines.append(f"- **Query type:** {answers.query_type.value}")
        lines.append(f"- **Query complexity:** {answers.query_complexity.value}")
        lines.append(f"- **Expected answer type:** {answers.expected_answer_type.value}")
        if answers.sample_queries:
            lines.append(f"- **Sample queries:** {'; '.join(answers.sample_queries)}")
        lines.append(f"- **Update frequency:** {answers.update_frequency.value}")
        future = "Yes" if answers.future_languages else "No"
        lines.append(f"- **Future languages planned:** {future}")
        lines.append(f"- **Ground truth available:** {'Yes' if answers.has_ground_truth else 'No'}")

        if answers.document_stats:
            stats = answers.document_stats
            lines.append("")
            lines.append("### Document Corpus")
            lines.append(f"- **Total files:** {stats.total_files}")
            lines.append(f"- **Total size:** {stats.total_size_bytes / 1_048_576:.1f} MB")
            lines.append(f"- **Primary language:** {stats.primary_language}")
            lines.append(f"- **CJK content:** {'Yes' if stats.has_cjk else 'No'}")
            lines.append(f"- **Content type:** {stats.detected_content_type.value}")
            lines.append(f"- **Avg tokens/doc:** {stats.avg_tokens_per_doc:.0f}")

            if stats.languages_detected:
                lines.append("- **Languages detected:**")
                for lang, pct in sorted(
                    stats.languages_detected.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"  - {lang}: {pct:.0%}")

        lines.extend(["", "---", ""])
        return lines

    def _section_modalities(self, recs: Recommendations) -> list[str]:
        """Render ingestion advice for non-text modalities."""
        if not recs.modalities:
            return []

        lines = ["## Non-Text Modalities", ""]
        lines.append(
            "The text pipeline in this report covers documents only. "
            "The corpus also contains the following, each needing its own ingestion path."
        )
        lines.append("")
        lines.append("| Modality | Files | Share | Strategy |")
        lines.append("|----------|-------|-------|----------|")
        for m in recs.modalities:
            lines.append(
                f"| {m.modality.replace('_', ' ')} | {m.file_count} | "
                f"{m.share:.0%} | {m.strategy} |"
            )
        lines.append("")

        for m in recs.modalities:
            lines.append(f"### {m.modality.replace('_', ' ').title()}")
            lines.append("")
            if m.extensions:
                lines.append(f"- **Extensions:** {', '.join(m.extensions)}")
            lines.append(f"- **Embedding:** {m.embedding}")
            lines.append(f"- **Chunking:** {m.chunking}")
            lines.append("")
            if m.ingestion:
                lines.append("**Ingestion steps:**")
                for i, step in enumerate(m.ingestion, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")
            if m.tools_local:
                lines.append(f"- **Local tools:** {'; '.join(m.tools_local)}")
            if m.tools_hosted:
                lines.append(f"- **Hosted services:** {'; '.join(m.tools_hosted)}")
            for n in m.notes:
                lines.append(f"- {n}")
            for w in m.warnings:
                lines.append(f"- **Warning:** {w}")
            if m.code_snippet:
                lines.append("")
                lines.append("```python")
                lines.append(m.code_snippet)
                lines.append("```")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_embedding_models(
        self, answers: UserAnswers, recs: Recommendations
    ) -> list[str]:
        """Render the embedding model recommendation section."""
        lines = ["## Embedding Model", ""]

        if not recs.embedding_models:
            lines.append("*No suitable models found matching your constraints.*")
            lines.extend(["", "---", ""])
            return lines

        top = recs.embedding_models[0]
        lines.append(f"- **Model:** `{top.model_id}`")
        lines.append(f"- **Provider:** {top.provider}")
        if top.quality_score > 0:
            lines.append(
                f"- **Retrieval quality:** ~{top.quality_score:.0f} (approx. MTEB nDCG@10)"
            )
        lines.append(f"- **Dimension:** {top.dimension}")
        lines.append(f"- **Max Tokens:** {top.max_tokens}")
        lines.append(f"- **Estimated Size:** {top.estimated_size_gb} GB")
        lines.append(f"- **License:** {top.license}")
        lines.append(f"- **Multilingual:** {'Yes' if top.multilingual else 'No'}")
        if top.trust_remote_code:
            lines.append("- **Loading:** requires `trust_remote_code=True`")
        lines.append(f"- **Fitness Score:** {top.score:.2f}")
        lines.append("")

        if top.reasons:
            lines.append("**Why selected:**")
            for r in top.reasons:
                lines.append(f"- {r}")
            lines.append("")

        if top.warnings:
            lines.append("**Warnings:**")
            for w in top.warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Alternative models table
        if len(recs.embedding_models) > 1:
            lines.append("### Alternative Models Considered")
            lines.append("")
            lines.append(
                "| Model | Provider | Score | Quality | Dimension | Max Tokens | "
                "Size (GB) | Multilingual |"
            )
            lines.append(
                "|-------|----------|-------|---------|-----------|------------|"
                "-----------|--------------|"
            )
            for m in recs.embedding_models:
                quality = f"~{m.quality_score:.0f}" if m.quality_score > 0 else "n/a"
                size = "API" if m.provider != "huggingface" else str(m.estimated_size_gb)
                lines.append(
                    f"| `{m.model_id}` | {m.provider} | {m.score:.2f} | {quality} | "
                    f"{m.dimension} | {m.max_tokens} | {size} | "
                    f"{'Yes' if m.multilingual else 'No'} |"
                )
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_chunking(self, recs: Recommendations) -> list[str]:
        """Render the chunking strategy section."""
        lines = ["## Chunking Strategy", ""]

        if not recs.chunking:
            lines.append("*No chunking recommendation generated.*")
            lines.extend(["", "---", ""])
            return lines

        c = recs.chunking
        lines.append(f"- **Strategy:** {c.strategy}")
        lines.append(f"- **Chunk Size:** {c.chunk_size} {c.count_by}")
        lines.append(f"- **Overlap:** {c.chunk_overlap} {c.count_by}")
        lines.append(f"- **Content Type:** {c.content_type}")
        lines.append("")

        if c.notes:
            lines.append("**Notes:**")
            for n in c.notes:
                lines.append(f"- {n}")
            lines.append("")

        if c.language_overrides:
            lines.append("### Language-Specific Overrides")
            lines.append("")
            for lang, override in c.language_overrides.items():
                parts = [f"**{lang.upper()}:**"]
                if "chunk_size" in override:
                    parts.append(f"chunk_size={override['chunk_size']}")
                if "count_by" in override:
                    parts.append(f"count_by={override['count_by']}")
                if "tokenizer" in override:
                    parts.append(f"tokenizer=`{override['tokenizer']}`")
                if "sentence_model" in override:
                    parts.append(f"spaCy model=`{override['sentence_model']}`")
                lines.append(f"- {' '.join(parts)}")
            lines.append("")

        if c.code_snippet:
            lines.append("### Code Snippet")
            lines.append("")
            lines.append("```python")
            lines.append(c.code_snippet)
            lines.append("```")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_vector_db(self, recs: Recommendations) -> list[str]:
        """Render the vector DB section."""
        lines = ["## Vector Database", ""]

        if not recs.vector_db:
            lines.append("*No vector DB recommendation generated.*")
            lines.extend(["", "---", ""])
            return lines

        db = recs.vector_db
        lines.append(f"- **Provider:** {db.provider}")
        lines.append(f"- **Category:** {db.category}")
        lines.append(f"- **Reason:** {db.reason}")
        lines.append(f"- **Library:** `{db.library}`")
        if db.estimated_capacity:
            lines.append(f"- **Capacity:** {db.estimated_capacity}")
        lines.append(
            f"- **Metadata filtering:** {'Yes' if db.supports_metadata_filter else 'No'}"
        )
        lines.append(
            f"- **Hybrid search:** {'Yes' if db.supports_hybrid_search else 'No'}"
        )
        lines.append("")

        if db.code_snippet:
            lines.append("### Code Snippet")
            lines.append("")
            lines.append("```python")
            lines.append(db.code_snippet)
            lines.append("```")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_retrieval(self, recs: Recommendations) -> list[str]:
        """Render the retrieval settings section."""
        lines = ["## Retrieval Settings", ""]

        if not recs.retrieval:
            lines.extend(["", "---", ""])
            return lines

        r = recs.retrieval
        lines.append(f"- **Top-K:** {r.top_k}")
        lines.append(f"- **Similarity Threshold:** {r.similarity_threshold}")
        lines.append(f"- **Reranking:** {'Yes' if r.rerank else 'No'}")
        if r.rerank and r.rerank_model:
            lines.append(f"- **Reranker:** `{r.rerank_model}`")
        lines.append(f"- **Hybrid Search:** {'Yes' if r.hybrid_search else 'No'}")
        lines.append("")

        if r.hybrid_search:
            lines.append("### Hybrid Retrieval")
            lines.append("")
            native = "native in the vector DB" if r.hybrid_native else "in-process (rank-bm25)"
            lines.append(
                f"- **Sparse retriever:** {r.sparse_method.upper()} ({native})"
            )
            lines.append(f"- **Fusion:** {r.fusion_method.upper()}")
            if r.hybrid_reasons:
                lines.append("")
                lines.append("**Why:**")
                for reason in r.hybrid_reasons:
                    lines.append(f"- {reason}")
            if r.hybrid_code_snippet:
                lines.append("")
                lines.append("```python")
                lines.append(r.hybrid_code_snippet)
                lines.append("```")
            lines.append("")

        if r.prompt_strategy != "none":
            lines.append("### Generation Settings")
            lines.append("")
            lines.append(f"- **Temperature:** {r.temperature}")
            lines.append(f"- **Max Tokens:** {r.max_tokens}")
            lines.append(f"- **Prompt Strategy:** {r.prompt_strategy}")
            lines.append("")

        if r.notes:
            lines.append("**Notes:**")
            for n in r.notes:
                lines.append(f"- {n}")

        lines.extend(["", "---", ""])
        return lines

    def _section_reranker(self, recs: Recommendations) -> list[str]:
        """Render the reranking stage section."""
        rr = recs.reranker
        if rr is None:
            return []

        lines = ["## Reranking", ""]
        if not rr.enabled:
            lines.append("**Reranking:** not recommended for this profile")
            for n in rr.notes:
                lines.append(f"- {n}")
            for w in rr.warnings:
                lines.append(f"- **Warning:** {w}")
            lines.extend(["", "---", ""])
            return lines

        lines.append(f"- **Model:** `{rr.model_id}`")
        lines.append(f"- **Provider:** {rr.provider}")
        lines.append(f"- **Pipeline:** retrieve top {rr.fetch_k} -> rerank -> keep {rr.final_k}")
        lines.append(f"- **Estimated reranking latency:** ~{rr.estimated_latency_ms} ms")
        lines.append(f"- **Relative quality:** ~{rr.quality_score:.0f}/100 (approx.)")
        lines.append(f"- **Input window:** {rr.max_tokens} tokens")
        lines.append(f"- **Multilingual:** {'Yes' if rr.multilingual else 'No'}")
        lines.append(f"- **License:** {rr.license}")
        if rr.trust_remote_code:
            lines.append("- **Loading:** requires `trust_remote_code=True`")
        lines.append("")

        if rr.reasons:
            lines.append("**Why rerank:**")
            for r in rr.reasons:
                lines.append(f"- {r}")
            lines.append("")
        if rr.model_reasons:
            lines.append("**Why this model:**")
            for r in rr.model_reasons:
                lines.append(f"- {r}")
            lines.append("")
        if rr.warnings:
            lines.append("**Warnings:**")
            for w in rr.warnings:
                lines.append(f"- {w}")
            lines.append("")
        if rr.alternatives:
            lines.append("| Alternative | Provider | Score | Quality | Latency |")
            lines.append("|-------------|----------|-------|---------|---------|")
            for a in rr.alternatives:
                lines.append(
                    f"| `{a['model_id']}` | {a['provider']} | {a['score']:.2f} | "
                    f"~{a['quality_score']:.0f} | ~{a['estimated_latency_ms']} ms |"
                )
            lines.append("")
        if rr.code_snippet:
            lines.append("```python")
            lines.append(rr.code_snippet)
            lines.append("```")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_approach(self, recs: Recommendations) -> list[str]:
        """Render the approach assessment section."""
        if not recs.approach or recs.approach.proceed_with_rag:
            return []

        a = recs.approach
        lines = [
            "## Approach Assessment",
            "",
            f"> **Recommended approach:** "
            f"{a.recommended_approach.value.replace('_', ' ').title()} "
            f"(confidence: {a.confidence:.0%})",
            ">",
            f"> {a.reasoning}",
            "",
            "### Alternative Approach",
            "",
            a.alternative_description,
            "",
            "---",
            "",
        ]
        return lines

    def _section_query_pipeline(self, recs: Recommendations) -> list[str]:
        """Render the query transformation pipeline section."""
        lines = ["## Query Pipeline", ""]

        if not recs.query_transformation or not recs.query_transformation.techniques:
            lines.append("*No query transformations needed for your configuration.*")
            lines.extend(["", "---", ""])
            return lines

        qt = recs.query_transformation

        lines.append(
            f"Estimated latency impact: **~{qt.latency_impact_ms}ms** | "
            f"Requires LLM: **{'Yes' if qt.requires_llm else 'No'}**"
        )
        lines.append("")

        for tech in qt.techniques:
            priority = tech.get("priority", "optional")
            badge = "REQUIRED" if priority == "required" else (
                "RECOMMENDED" if priority == "recommended" else "OPTIONAL"
            )
            lines.append(f"### {tech['name']} [{badge}]")
            lines.append("")
            lines.append(tech.get("description", ""))
            lines.append("")

        if qt.notes:
            lines.append("**Notes:**")
            for n in qt.notes:
                lines.append(f"- {n}")
            lines.append("")

        if qt.code_snippet:
            lines.append("### Code Snippet")
            lines.append("")
            lines.append("```python")
            lines.append(qt.code_snippet)
            lines.append("```")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_estimates(self, recs: Recommendations) -> list[str]:
        """Render order-of-magnitude cost, footprint and latency estimates."""
        e = recs.estimates
        if e is None:
            return []

        lines = ["## Cost, Footprint & Latency Estimates", ""]
        lines.append(
            "Order-of-magnitude figures for planning; compare them, do not bill against them."
        )
        lines.append("")
        approx = "~" if e.corpus_tokens_estimated else ""
        lines.append("| Item | Estimate |")
        lines.append("|------|----------|")
        if e.chunk_count:
            lines.append(f"| Corpus tokens | {approx}{e.corpus_tokens:,} |")
            lines.append(f"| Chunks to index | {approx}{e.chunk_count:,} |")
            lines.append(f"| Tokens to embed | {approx}{e.tokens_to_embed:,} |")
            lines.append(f"| Index on disk | ~{e.index_size_mb:,.0f} MB |")
            lines.append(f"| Vectors in RAM | ~{e.index_memory_mb:,.0f} MB |")
            if e.embedding_is_api:
                lines.append(f"| One-off embedding cost | ~${e.indexing_cost_usd:,.2f} |")
            lines.append(f"| Indexing time | ~{e.indexing_time_min:,.0f} min |")
            if e.monthly_reindex_cost_usd:
                lines.append(f"| Monthly re-indexing (API) | ~${e.monthly_reindex_cost_usd:,.2f} |")
        budget = f"{e.latency_budget_ms} ms" if e.latency_budget_ms < 10**9 else "unlimited"
        fit = "fits" if e.fits_latency_budget else "**exceeds**"
        lines.append(
            f"| Retrieval latency per query | ~{e.query_latency_ms} ms ({fit} {budget} budget) |"
        )
        if e.queries_per_day:
            lines.append(
                f"| Monthly query-side API cost | ~${e.monthly_query_cost_usd:,.2f} "
                f"at {e.queries_per_day:,} queries/day |"
            )
        lines.append("")

        if e.query_latency_breakdown_ms:
            lines.append("**Latency breakdown (ms):** " + ", ".join(
                f"{k.replace('_', ' ')} {v}" for k, v in e.query_latency_breakdown_ms.items()
            ))
            lines.append("")
        for w in e.warnings:
            lines.append(f"- **Warning:** {w}")
        for n in e.notes:
            lines.append(f"- {n}")
        if e.assumptions:
            lines.append("")
            lines.append("**Assumptions:**")
            for a in e.assumptions:
                lines.append(f"- {a}")
        lines.extend(["", "---", ""])
        return lines

    def _section_validation(self, recs: Recommendations) -> list[str]:
        """Render the ground-truth validation section."""
        v = recs.validation
        if v is None:
            return []

        lines = ["## Validation Against Ground Truth", ""]
        if not v.ran:
            lines.append(f"*Validation did not run:* {v.error}")
            lines.extend(["", "---", ""])
            return lines

        lines.append(
            f"**Verdict:** {v.verdict}  "
            f"({v.num_queries} queries, {v.num_chunks} chunks)"
        )
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Chunking strategy | {v.strategy} |")
        lines.append(
            f"| Chunk size / overlap | {v.chunk_size_chars} / {v.chunk_overlap_chars} chars |"
        )
        lines.append(f"| Embedding model | `{v.embedding_model}` |")
        lines.append(f"| Vector backend | {v.vector_backend} |")
        lines.append(f"| Top-K | {v.top_k} |")
        lines.append(f"| Retrieval mode | {v.retrieval_mode} |")
        if v.reranker_model:
            lines.append(f"| Reranker | `{v.reranker_model}` |")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Hit Rate@{v.top_k} | {v.hit_rate:.1%} |")
        lines.append(f"| MRR | {v.mrr:.3f} |")
        lines.append(f"| Precision@{v.top_k} | {v.precision_at_k:.1%} |")
        lines.append(f"| Recall@{v.top_k} | {v.recall_at_k:.1%} |")
        lines.append(f"| nDCG@{v.top_k} | {v.ndcg_at_k:.3f} |")
        if v.has_baseline:
            lines.append(
                f"| Dense-only baseline | hit rate {v.baseline_hit_rate:.1%}, "
                f"MRR {v.baseline_mrr:.3f} |"
            )
        lines.append("")
        if v.model_comparison:
            lines.append("**Embedding models compared on your data (best first):**")
            lines.append("")
            lines.append("| Model | Hit Rate | MRR | Recall |")
            lines.append("|-------|----------|-----|--------|")
            for c in v.model_comparison:
                lines.append(
                    f"| `{c['model']}` | {c['hit_rate']:.1%} | {c['mrr']:.3f} | "
                    f"{c['recall_at_k']:.1%} |"
                )
            lines.append("")

        if v.suggestions:
            lines.append("**Next steps:**")
            for s in v.suggestions:
                lines.append(f"- {s}")
            lines.append("")
        if v.notes:
            lines.append("**Notes:**")
            for n in v.notes:
                lines.append(f"- {n}")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_llm_verification(self, recs: Recommendations) -> list[str]:
        """Render the LLM verification section."""
        if not recs.llm_verification:
            return []

        v = recs.llm_verification
        lines = [
            "## LLM Verification",
            "",
            f"*Verified by {v.provider} ({v.model})*",
            "",
            v.summary,
            "",
        ]

        if v.agreements:
            lines.append("### Agreements")
            lines.append("")
            for a in v.agreements:
                lines.append(f"- {a}")
            lines.append("")

        if v.refinements:
            lines.append("### Suggested Refinements")
            lines.append("")
            for r in v.refinements:
                lines.append(f"- {r}")
            lines.append("")

        if v.additional_considerations:
            lines.append("### Additional Considerations")
            lines.append("")
            for c in v.additional_considerations:
                lines.append(f"- {c}")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _section_implementation(self, recs: Recommendations) -> list[str]:
        """Render the implementation checklist."""
        lines = ["## Implementation Checklist", ""]

        if recs.implementation_steps:
            for step in recs.implementation_steps:
                lines.append(f"- [ ] `{step}`")
        else:
            lines.append("- [ ] Install recommended packages")
            lines.append("- [ ] Run validation script with sample queries")

        lines.extend(["", "---", ""])
        return lines

    def _section_reindexing_warning(
        self, answers: UserAnswers, recs: Recommendations
    ) -> list[str]:
        """Render the critical re-indexing warning section."""
        languages = (
            list(answers.document_stats.languages_detected.keys())
            if answers.document_stats
            else ["en"]
        )
        multilingual = (
            recs.embedding_models[0].multilingual if recs.embedding_models else False
        )

        lines = [
            "## Re-Indexing Warning",
            "",
            "> **CRITICAL:** If you change the embedding model in the future, "
            "you MUST re-index all documents.",
            "> Vector spaces are not compatible across different models.",
            ">",
            f"> **Current languages supported:** {', '.join(languages)}  ",
            f"> **Future languages planned:** {'Yes' if answers.future_languages else 'No'}  ",
            f"> **Multilingual model selected:** {'Yes' if multilingual else 'No'}",
            "",
        ]

        return lines
