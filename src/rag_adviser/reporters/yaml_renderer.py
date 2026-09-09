"""Machine-readable YAML configuration renderer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from rag_adviser import __version__
from rag_adviser.models import Recommendations, UserAnswers


class YamlRenderer:
    """Generate a machine-readable YAML config from recommendations."""

    def render(
        self,
        answers: UserAnswers,
        recommendations: Recommendations,
        output_dir: Path,
    ) -> Path:
        """Render the YAML config and write to disk."""
        config = self._build_config(answers, recommendations)
        out_path = output_dir / "rag_config.yaml"
        out_path.write_text(
            yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return out_path

    def _build_config(
        self, answers: UserAnswers, recs: Recommendations
    ) -> dict:
        """Build the YAML configuration dict."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        languages = (
            list(answers.document_stats.languages_detected.keys())
            if answers.document_stats
            else ["en"]
        )
        lang_distribution = (
            answers.document_stats.languages_detected
            if answers.document_stats
            else {"en": 1.0}
        )

        config: dict = {
            "ragadvisor_version": __version__,
            "generated_at": now,
            "metadata": {
                "use_case": answers.use_case.value,
                "languages": languages,
                "language_distribution": lang_distribution,
                "future_languages": answers.future_languages,
                "content_type": (
                    answers.document_stats.detected_content_type.value
                    if answers.document_stats
                    else "prose"
                ),
                "deployment": answers.constraints.environment.value,
                "privacy": answers.constraints.privacy.value,
                "query_type": answers.query_type.value,
                "query_complexity": answers.query_complexity.value,
                "expected_answer_type": answers.expected_answer_type.value,
            },
        }

        # Approach assessment
        if recs.approach:
            config["approach"] = {
                "recommended": recs.approach.recommended_approach.value,
                "proceed_with_rag": recs.approach.proceed_with_rag,
                "confidence": recs.approach.confidence,
                "reasoning": recs.approach.reasoning,
            }
            if not recs.approach.proceed_with_rag:
                config["approach"]["alternative"] = recs.approach.alternative_description

        # Document stats
        if answers.document_stats:
            stats = answers.document_stats
            config["corpus"] = {
                "path": str(answers.document_path) if answers.document_path else None,
                "total_files": stats.total_files,
                "total_size_bytes": stats.total_size_bytes,
                "primary_language": stats.primary_language,
                "has_cjk": stats.has_cjk,
                "avg_tokens_per_doc": round(stats.avg_tokens_per_doc, 1),
                "total_tokens": stats.total_tokens,
            }

        # Embedding model
        if recs.embedding_models:
            top = recs.embedding_models[0]
            config["embedding"] = {
                "model": top.model_id,
                "dimension": top.dimension,
                "max_tokens": top.max_tokens,
                "estimated_size_gb": top.estimated_size_gb,
                "license": top.license,
                "multilingual": top.multilingual,
            }
            if len(recs.embedding_models) > 1:
                config["embedding"]["alternatives"] = [
                    m.model_id for m in recs.embedding_models[1:]
                ]

        # Chunking
        if recs.chunking:
            c = recs.chunking
            config["chunking"] = {
                "strategy": c.strategy,
                "default": {
                    "chunk_size": c.chunk_size,
                    "overlap": c.chunk_overlap,
                    "count_by": c.count_by,
                },
                "content_type": c.content_type,
            }
            if c.language_overrides:
                config["chunking"]["language_overrides"] = c.language_overrides

        # Vector DB
        if recs.vector_db:
            db = recs.vector_db
            config["vector_db"] = {
                "provider": db.provider,
                "category": db.category,
                "library": db.library,
                "supports_metadata_filter": db.supports_metadata_filter,
                "supports_hybrid_search": db.supports_hybrid_search,
            }

        # Retrieval
        if recs.retrieval:
            r = recs.retrieval
            config["retrieval"] = {
                "top_k": r.top_k,
                "similarity_threshold": r.similarity_threshold,
                "rerank": r.rerank,
                "query_preprocessing": r.query_preprocessing,
            }
            if r.rerank and r.rerank_model:
                config["retrieval"]["rerank_model"] = r.rerank_model
            if r.prompt_strategy != "none":
                config["generation"] = {
                    "temperature": r.temperature,
                    "max_tokens": r.max_tokens,
                    "prompt_strategy": r.prompt_strategy,
                }

        # Query transformation
        if recs.query_transformation and recs.query_transformation.techniques:
            qt = recs.query_transformation
            config["query_pipeline"] = {
                "techniques": [
                    {
                        "name": t["name"],
                        "priority": t.get("priority", "optional"),
                    }
                    for t in qt.techniques
                ],
                "latency_impact_ms": qt.latency_impact_ms,
                "requires_llm": qt.requires_llm,
            }

        # LLM Verification
        if recs.llm_verification:
            v = recs.llm_verification
            config["llm_verification"] = {
                "provider": v.provider,
                "model": v.model,
                "summary": v.summary,
                "agreements": v.agreements,
                "refinements": v.refinements,
                "additional_considerations": v.additional_considerations,
            }

        # Evaluation
        config["evaluation"] = {
            "has_ground_truth": answers.has_ground_truth,
            "recommended_metrics": ["hit_rate", "mrr", "context_precision"],
        }

        return config
