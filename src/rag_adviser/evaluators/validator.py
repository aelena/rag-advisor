"""Validate rule-based recommendations against the user's own ground truth.

This closes the loop between advice and measurement: the recommended chunking
strategy, chunk size, embedding model and top-k are run through the existing
evaluation pipeline on the user's corpus and queries, and the resulting
retrieval metrics are attached to the report.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from rich.console import Console

from rag_adviser.models import Recommendations, UserAnswers, ValidationResult

logger = logging.getLogger(__name__)

# Recommended chunking strategy -> evaluation pipeline strategy.
_STRATEGY_MAP = {
    "recursive": "recursive",
    "hierarchical": "hierarchical",
    "language_aware": "adaptive",   # per-file code/markdown/prose detection
    "speaker_split": "recursive",   # no dedicated eval implementation yet
    "row_based": "recursive",
}

# The evaluation pipeline sizes chunks in characters; recommendations are in
# tokens. ~4 characters per token is the usual English approximation.
_CHARS_PER_TOKEN = 4

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many recommended local models to try before giving up. A model can fail
# to load for reasons unrelated to the advice (missing extra package, gated
# repo, no network); the next best local model is then used instead.
_MAX_MODEL_ATTEMPTS = 3

# Verdict thresholds on hit rate (fraction of queries with >= 1 relevant hit).
_STRONG_HIT_RATE = 0.80
_ACCEPTABLE_HIT_RATE = 0.60

_INSTALL_HINT = "install with: pip install ragadvisor[eval]"


def map_strategy(recommended: str) -> str:
    """Map a recommended chunking strategy name onto an evaluable one."""
    return _STRATEGY_MAP.get(recommended, "recursive")


class RecommendationValidator:
    """Run the recommended configuration through the evaluation pipeline."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def validate(self, answers: UserAnswers, recs: Recommendations) -> ValidationResult:
        """Evaluate the recommendations. Never raises; errors land in ``result.error``."""
        result = ValidationResult()

        if not answers.document_path:
            result.error = "no document path was provided (use --document-path)"
            return result
        if not answers.ground_truth_path:
            result.error = "no ground truth file was provided (use --ground-truth-path)"
            return result
        gt_path = Path(answers.ground_truth_path)
        if not gt_path.exists():
            result.error = f"ground truth file not found: {gt_path}"
            return result

        # ── Translate recommendations into an eval config ──────────────────
        chunking = recs.chunking
        if chunking is None:
            result.error = "no chunking recommendation to validate"
            return result

        result.strategy = map_strategy(chunking.strategy)
        if result.strategy != chunking.strategy:
            result.notes.append(
                f"Recommended strategy '{chunking.strategy}' evaluated as "
                f"'{result.strategy}' (closest available in the evaluation pipeline)"
            )

        factor = _CHARS_PER_TOKEN if chunking.count_by == "tokens" else 1
        result.chunk_size_chars = max(chunking.chunk_size * factor, 100)
        result.chunk_overlap_chars = chunking.chunk_overlap * factor
        result.top_k = recs.retrieval.top_k if recs.retrieval else 5
        result.vector_backend = self._pick_backend()

        candidates, model_note = self._local_candidates(recs)
        if model_note:
            result.notes.append(model_note)

        try:
            from rag_adviser.evaluators.pipeline_runner import EvalConfig
        except ImportError as e:  # pragma: no cover - core module, defensive
            result.error = f"evaluation pipeline unavailable: {e}"
            return result

        config = EvalConfig(
            corpus_path=Path(answers.document_path),
            ground_truth_path=gt_path,
            strategies=[result.strategy],
            top_k=result.top_k,
            chunk_size=result.chunk_size_chars,
            chunk_overlap=result.chunk_overlap_chars,
            vector_backend=result.vector_backend,
        )

        # ── Run, falling back to the next local model on load failures ─────
        report = None
        last_error = ""
        for model_id, trust_remote_code in candidates:
            config.embedding_model = model_id
            config.trust_remote_code = trust_remote_code
            try:
                report = self._run(config)
                result.embedding_model = model_id
                break
            except ImportError as e:
                last_error = f"missing optional dependency for {model_id} ({e}); {_INSTALL_HINT}"
            except Exception as e:  # evaluation must never break the main run
                logger.debug("Validation with %s failed", model_id, exc_info=True)
                last_error = f"{type(e).__name__}: {e}"
            result.notes.append(f"Could not evaluate with {model_id}: {last_error[:200]}")

        if report is None:
            result.error = last_error or "no evaluable embedding model"
            return result

        if not report.strategy_results:
            result.error = "evaluation produced no results"
            return result

        m = report.strategy_results[0]
        result.ran = True
        result.num_queries = m.num_queries
        result.num_chunks = m.num_chunks
        result.hit_rate = m.hit_rate
        result.mrr = m.mrr
        result.precision_at_k = m.mean_precision_at_k
        result.recall_at_k = m.mean_recall_at_k
        result.ndcg_at_k = m.mean_ndcg_at_k
        result.verdict, result.suggestions = self._interpret(result, recs)
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    def _run(self, config):  # noqa: ANN001 - EvalConfig imported lazily
        """Execute the evaluation pipeline (separated for easy mocking)."""
        from rag_adviser.evaluators.pipeline_runner import EvalPipelineRunner

        return EvalPipelineRunner(console=self.console).run(config)

    @staticmethod
    def _local_candidates(recs: Recommendations) -> tuple[list[tuple[str, bool]], str]:
        """Locally runnable recommended models, best first, plus a default.

        Returns ``([(model_id, trust_remote_code), ...], note)``.
        """
        models = recs.embedding_models
        local = [
            (m.model_id, m.trust_remote_code)
            for m in models
            if m.provider == "huggingface" and m.model_id
        ][:_MAX_MODEL_ATTEMPTS]

        note = ""
        if not models:
            note = f"No model recommendation; validated with {_DEFAULT_MODEL}"
        elif not local:
            note = (
                "All recommended models are hosted APIs; validated with "
                f"{_DEFAULT_MODEL} as a local proxy"
            )
        elif local[0][0] != models[0].model_id:
            note = (
                f"Top recommendation '{models[0].model_id}' is a hosted API; "
                f"validated with the best local alternative '{local[0][0]}'"
            )

        if all(model_id != _DEFAULT_MODEL for model_id, _ in local):
            local.append((_DEFAULT_MODEL, False))
        return local, note

    @staticmethod
    def _pick_backend() -> str:
        """Prefer an installed ANN backend; fall back to exact numpy search."""
        candidates = (("faiss", "faiss"), ("chromadb", "chroma"), ("sqlite_vec", "sqlite"))
        for module, backend in candidates:
            if importlib.util.find_spec(module) is not None:
                return backend
        return "memory"  # numpy brute force; needs only sentence-transformers

    @staticmethod
    def _interpret(result: ValidationResult, recs: Recommendations) -> tuple[str, list[str]]:
        """Turn raw metrics into a verdict and concrete next steps."""
        suggestions: list[str] = []
        if result.hit_rate >= _STRONG_HIT_RATE:
            verdict = "strong"
        elif result.hit_rate >= _ACCEPTABLE_HIT_RATE:
            verdict = "acceptable"
        else:
            verdict = "weak"

        retrieval = recs.retrieval
        if result.hit_rate < _STRONG_HIT_RATE and result.top_k < 10:
            suggestions.append(
                f"Raise top_k from {result.top_k} to 10 and rerank down to {result.top_k}"
            )
        if result.mrr < 0.5 and retrieval and not retrieval.rerank:
            suggestions.append(
                "Relevant chunks are found but rank low: add a cross-encoder reranker"
            )
        if verdict != "strong" and retrieval and not retrieval.hybrid_search:
            suggestions.append(
                "Enable hybrid retrieval (BM25 + dense with RRF) to catch exact-term queries"
            )
        if verdict == "weak":
            suggestions.append(
                "Compare chunking strategies directly: "
                "ragadvisor evaluate <corpus> <ground_truth> --strategy recursive "
                "--strategy semantic --strategy hierarchical"
            )
            suggestions.append(
                "Check the ground truth: relevant_docs must match corpus file names exactly"
            )
        if result.num_queries < 20:
            suggestions.append(
                f"Only {result.num_queries} evaluation queries; metrics are noisy below ~20"
            )
        return verdict, suggestions
