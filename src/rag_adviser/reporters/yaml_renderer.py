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
                "sampled_files": stats.sampled_files,
                "total_files_all_modalities": stats.total_files_all,
                "modalities": dict(stats.modalities),
                "scanned_pdfs_in_sample": stats.scanned_pdfs,
            }

        # Non-text modalities
        if recs.modalities:
            config["modalities"] = [
                {
                    "modality": m.modality,
                    "file_count": m.file_count,
                    "share": round(m.share, 3),
                    "extensions": m.extensions,
                    "strategy": m.strategy,
                    "ingestion": m.ingestion,
                    "tools_local": m.tools_local,
                    "tools_hosted": m.tools_hosted,
                    "embedding": m.embedding,
                    "chunking": m.chunking,
                    "pip_packages": m.pip_packages,
                    "warnings": m.warnings,
                }
                for m in recs.modalities
            ]

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
                "provider": top.provider,
                "trust_remote_code": top.trust_remote_code,
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
            config["retrieval"]["hybrid"] = {
                "enabled": r.hybrid_search,
                "sparse_method": r.sparse_method,
                "fusion": r.fusion_method,
                "native_in_vector_db": r.hybrid_native,
            }
            if r.prompt_strategy != "none":
                config["generation"] = {
                    "temperature": r.temperature,
                    "max_tokens": r.max_tokens,
                    "prompt_strategy": r.prompt_strategy,
                }

        # Reranking stage
        if recs.reranker:
            rr = recs.reranker
            config["reranker"] = {
                "enabled": rr.enabled,
                "model": rr.model_id,
                "provider": rr.provider,
                "fetch_k": rr.fetch_k,
                "final_k": rr.final_k,
                "estimated_latency_ms": rr.estimated_latency_ms,
                "max_tokens": rr.max_tokens,
                "trust_remote_code": rr.trust_remote_code,
            }
            if rr.alternatives:
                config["reranker"]["alternatives"] = [a["model_id"] for a in rr.alternatives]

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

        # Cost / footprint / latency estimates
        if recs.estimates:
            e = recs.estimates
            config["estimates"] = {
                "corpus_tokens": e.corpus_tokens,
                "corpus_tokens_estimated": e.corpus_tokens_estimated,
                "chunk_count": e.chunk_count,
                "tokens_to_embed": e.tokens_to_embed,
                "index_size_mb": e.index_size_mb,
                "index_memory_mb": e.index_memory_mb,
                "indexing_cost_usd": e.indexing_cost_usd,
                "indexing_time_min": e.indexing_time_min,
                "monthly_reindex_cost_usd": e.monthly_reindex_cost_usd,
                "queries_per_day": e.queries_per_day,
                "monthly_query_cost_usd": e.monthly_query_cost_usd,
                "query_latency_ms": e.query_latency_ms,
                "query_latency_breakdown_ms": e.query_latency_breakdown_ms,
                "latency_budget_ms": e.latency_budget_ms if e.latency_budget_ms < 10**9 else None,
                "fits_latency_budget": e.fits_latency_budget,
                "assumptions": e.assumptions,
                "warnings": e.warnings,
            }

        # Validation against ground truth
        if recs.validation:
            v = recs.validation
            config["validation"] = {
                "ran": v.ran,
            }
            if v.ran:
                config["validation"].update({
                    "verdict": v.verdict,
                    "strategy": v.strategy,
                    "embedding_model": v.embedding_model,
                    "chunk_size_chars": v.chunk_size_chars,
                    "chunk_overlap_chars": v.chunk_overlap_chars,
                    "vector_backend": v.vector_backend,
                    "top_k": v.top_k,
                    "retrieval_mode": v.retrieval_mode,
                    "model_comparison": v.model_comparison or None,
                    "reranker_model": v.reranker_model or None,
                    "dense_baseline": (
                        {"hit_rate": round(v.baseline_hit_rate, 4), "mrr": round(v.baseline_mrr, 4)}
                        if v.has_baseline else None
                    ),
                    "num_queries": v.num_queries,
                    "num_chunks": v.num_chunks,
                    "metrics": {
                        "hit_rate": round(v.hit_rate, 4),
                        "mrr": round(v.mrr, 4),
                        "precision_at_k": round(v.precision_at_k, 4),
                        "recall_at_k": round(v.recall_at_k, 4),
                        "ndcg_at_k": round(v.ndcg_at_k, 4),
                    },
                    "suggestions": v.suggestions,
                })
            else:
                config["validation"]["error"] = v.error

        # Evaluation
        config["evaluation"] = {
            "has_ground_truth": answers.has_ground_truth,
            "recommended_metrics": ["hit_rate", "mrr", "context_precision"],
        }

        return config
