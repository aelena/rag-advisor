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

        # Embedding Models
        lines.extend(self._section_embedding_models(answers, recs))

        # Chunking
        lines.extend(self._section_chunking(recs))

        # Vector DB
        lines.extend(self._section_vector_db(recs))

        # Retrieval
        lines.extend(self._section_retrieval(recs))

        # Query Pipeline
        lines.extend(self._section_query_pipeline(recs))

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
