"""Reranker recommender: decide whether to rerank and pick a cross-encoder.

A cross-encoder scores each (query, passage) pair jointly, which is far more
precise than bi-encoder similarity but costs one forward pass per candidate.
Whether that is worth it, and which model fits, depends on latency budget,
hardware, languages, chunk length, privacy and budget. This module turns
those constraints into a concrete pick with an estimated latency.
"""

from __future__ import annotations

from rag_adviser.analyzers.constraint_analyzer import ConstraintAnalyzer
from rag_adviser.config import load_defaults
from rag_adviser.models import (
    AnswerType,
    BudgetTier,
    HardwareProfile,
    LatencyBudget,
    QueryComplexity,
    RerankerRecommendation,
    RetrievalRecommendation,
    UseCase,
    UserAnswers,
)

# Portion of the end-to-end latency budget we allow the reranking stage to use.
_RERANK_ALLOWANCE_MS = {
    LatencyBudget.FAST: 150,
    LatencyBudget.MODERATE: 800,
    LatencyBudget.BATCH: 10**9,
}

# Rough CPU -> GPU speed-up for cross-encoder inference.
_GPU_SPEEDUP = 8.0

_NON_COMMERCIAL_MARKERS = ("nc", "non-commercial", "noncommercial", "research-only")


class RerankerRecommender:
    """Recommend a reranking stage (or explain why not)."""

    def __init__(self) -> None:
        defaults = load_defaults()
        self._catalogue: list[dict] = defaults.get("reranker_models", [])

    # ── Public API ─────────────────────────────────────────────────────────

    def recommend(
        self,
        answers: UserAnswers,
        retrieval: RetrievalRecommendation,
        languages: list[str] | None = None,
        multilingual: bool = False,
        chunk_tokens: int = 512,
    ) -> RerankerRecommendation:
        rec = RerankerRecommendation(final_k=retrieval.top_k)
        constraints = answers.constraints

        # ── Should we rerank at all? ───────────────────────────────────────
        signals = self._signals(answers, retrieval)
        rec.reasons.extend(signals)

        # ── Candidate pool ─────────────────────────────────────────────────
        rec.fetch_k = self._fetch_k(retrieval, answers)
        allowance = _RERANK_ALLOWANCE_MS[constraints.latency_budget]
        max_size = ConstraintAnalyzer().get_max_model_size(constraints)
        api_allowed = (
            ConstraintAnalyzer().is_api_allowed(constraints)
            and constraints.budget == BudgetTier.PAID_API
        )

        scored: list[tuple[float, dict, int, list[str], list[str]]] = []
        for model in self._catalogue:
            if model.get("provider", "huggingface") != "huggingface" and not api_allowed:
                continue
            latency = self._estimate_latency(model, rec.fetch_k, constraints.hardware)
            score, reasons, warnings = self._score(
                model, latency, allowance, max_size, multilingual, chunk_tokens
            )
            scored.append((score, model, latency, reasons, warnings))

        scored.sort(key=lambda s: s[0], reverse=True)
        fitting = [s for s in scored if s[2] <= allowance]

        if not signals:
            rec.enabled = False
            rec.notes.append(
                "Reranking not needed for this profile (simple factual queries, no "
                "hybrid fusion, no precision-critical answer type)"
            )
            if scored:
                self._fill_pick(rec, scored[0], suggested_only=True)
            return rec

        if not fitting:
            rec.enabled = False
            rec.warnings.append(
                f"No reranker fits the ~{allowance}ms reranking allowance of a "
                f"{constraints.latency_budget.value} budget on "
                f"{constraints.hardware.value.replace('_', ' ')}; "
                f"rerank offline/asynchronously or add a GPU"
            )
            if scored:
                self._fill_pick(rec, scored[0], suggested_only=True)
            return rec

        rec.enabled = True
        self._fill_pick(rec, fitting[0])
        rec.alternatives = [
            {
                "model_id": m["model_id"],
                "provider": m.get("provider", "huggingface"),
                "score": round(s, 2),
                "quality_score": m.get("quality_score", 0),
                "estimated_latency_ms": lat,
            }
            for s, m, lat, _, _ in fitting[1:4]
        ]
        rec.code_snippet = self._snippet(rec)
        return rec

    # ── Decision signals ───────────────────────────────────────────────────

    @staticmethod
    def _signals(answers: UserAnswers, retrieval: RetrievalRecommendation) -> list[str]:
        signals: list[str] = []
        if answers.constraints.latency_budget == LatencyBudget.BATCH:
            signals.append("Batch latency budget: reranking cost is irrelevant, precision is free")
        if answers.query_complexity == QueryComplexity.MULTI_HOP:
            signals.append("Multi-hop queries retrieve broadly; reranking picks the evidence chain")
        if answers.query_complexity == QueryComplexity.AGGREGATIVE:
            signals.append("Aggregative queries fetch many candidates that need re-ordering")
        if answers.query_complexity == QueryComplexity.COMPARATIVE:
            signals.append("Comparative queries need both sides ranked near the top")
        if retrieval.hybrid_search:
            signals.append(
                "Hybrid fusion produces a mixed candidate list; "
                "a cross-encoder re-scores it consistently"
            )
        if answers.expected_answer_type == AnswerType.EXACT_PASSAGE:
            signals.append("Exact-passage answers depend on the right chunk ranking first")
        if answers.use_case in (UseCase.LEGAL, UseCase.CODE):
            signals.append(
                f"{answers.use_case.value.replace('_', ' ').title()} rewards precision over recall"
            )
        return signals

    @staticmethod
    def _fetch_k(retrieval: RetrievalRecommendation, answers: UserAnswers) -> int:
        base = max(4 * retrieval.top_k, 20)
        if answers.query_complexity == QueryComplexity.AGGREGATIVE:
            base = max(base, 40)
        return min(base, 60)

    # ── Scoring ────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_latency(model: dict, fetch_k: int, hardware: HardwareProfile) -> int:
        if model.get("provider", "huggingface") != "huggingface":
            return int(model.get("latency_ms", 200))
        per_pair = float(model.get("latency_ms_per_pair_cpu", 30))
        total = per_pair * fetch_k
        if hardware == HardwareProfile.GPU_AVAILABLE:
            total /= _GPU_SPEEDUP
        return int(total)

    @staticmethod
    def _score(
        model: dict,
        latency_ms: int,
        allowance_ms: int,
        max_size_gb: float,
        multilingual: bool,
        chunk_tokens: int,
    ) -> tuple[float, list[str], list[str]]:
        score = 0.3
        reasons: list[str] = []
        warnings: list[str] = []
        is_api = model.get("provider", "huggingface") != "huggingface"
        quality = float(model.get("quality_score", 0))
        size = float(model.get("estimated_size_gb", 0))
        max_tokens = int(model.get("max_tokens", 512))
        license_ = str(model.get("license", "unknown")).lower()

        # Quality: 50 -> 0, 75 -> +0.30
        score += max(0.0, min(1.0, (quality - 50.0) / 25.0)) * 0.30
        reasons.append(f"Relative quality ~{quality:.0f}/100 (approx.)")

        # Latency fit
        if latency_ms <= allowance_ms:
            headroom = 1.0 - latency_ms / max(allowance_ms, 1)
            score += 0.10 + 0.10 * max(0.0, min(1.0, headroom))
            reasons.append(f"~{latency_ms}ms to rerank the candidate set")
        else:
            score -= 0.40
            warnings.append(f"~{latency_ms}ms exceeds the ~{allowance_ms}ms reranking allowance")

        # Hardware fit
        if is_api:
            score += 0.05
            reasons.append(f"Hosted {model['provider']} API; no local GPU/CPU cost")
            warnings.append("Query and candidate passages are sent to a third-party API")
        elif size <= max_size_gb:
            score += 0.10
        else:
            score -= 0.30
            warnings.append(f"{size}GB exceeds the {max_size_gb:.1f}GB local model limit")

        # Languages
        if multilingual:
            if model.get("multilingual"):
                score += 0.20
                reasons.append("Multilingual cross-encoder")
            else:
                score -= 0.40
                warnings.append("English-only reranker on a multilingual corpus")
        elif not model.get("multilingual"):
            score += 0.05

        # Chunk length vs reranker input window
        if max_tokens >= chunk_tokens + 64:
            score += 0.05
        else:
            score -= 0.10
            warnings.append(
                f"Input window ({max_tokens} tokens) is shorter than query + chunk "
                f"(~{chunk_tokens} tokens); passages will be truncated"
            )

        if any(marker in license_ for marker in _NON_COMMERCIAL_MARKERS):
            score -= 0.10
            warnings.append(f"License '{license_}' restricts commercial use")
        if model.get("trust_remote_code"):
            warnings.append("Loads custom model code from the Hub (trust_remote_code=True)")

        return max(0.0, min(1.0, score)), reasons, warnings

    @staticmethod
    def _fill_pick(
        rec: RerankerRecommendation,
        pick: tuple[float, dict, int, list[str], list[str]],
        suggested_only: bool = False,
    ) -> None:
        score, model, latency, reasons, warnings = pick
        rec.model_id = model["model_id"]
        rec.provider = model.get("provider", "huggingface")
        rec.quality_score = float(model.get("quality_score", 0))
        rec.estimated_size_gb = float(model.get("estimated_size_gb", 0))
        rec.max_tokens = int(model.get("max_tokens", 512))
        rec.multilingual = bool(model.get("multilingual", False))
        rec.license = str(model.get("license", "unknown"))
        rec.trust_remote_code = bool(model.get("trust_remote_code", False))
        rec.estimated_latency_ms = latency
        rec.score = score
        rec.model_reasons = reasons
        rec.warnings.extend(warnings)
        if model.get("notes"):
            rec.model_reasons.append(str(model["notes"]))
        if suggested_only:
            rec.notes.append(
                f"If you add reranking later, start with {rec.model_id} "
                f"(~{latency}ms for {rec.fetch_k} candidates)"
            )

    # ── Snippet ────────────────────────────────────────────────────────────

    @staticmethod
    def _snippet(rec: RerankerRecommendation) -> str:
        short = rec.model_id.split("/", 1)[-1]  # strip the provider prefix
        if rec.provider == "cohere":
            return "\n".join([
                "# pip install cohere",
                "import cohere",
                "",
                "co = cohere.ClientV2()  # COHERE_API_KEY",
                "",
                f"def rerank(query: str, candidates: list[str], k: int = {rec.final_k}):",
                f'    resp = co.rerank(model="{short}", query=query,',
                "                     documents=candidates, top_n=k)",
                "    return [candidates[r.index] for r in resp.results]",
            ])
        if rec.provider == "voyage":
            return "\n".join([
                "# pip install voyageai",
                "import voyageai",
                "",
                "vo = voyageai.Client()  # VOYAGE_API_KEY",
                "",
                f"def rerank(query: str, candidates: list[str], k: int = {rec.final_k}):",
                f'    resp = vo.rerank(query, candidates, model="{short}", top_k=k)',
                "    return [r.document for r in resp.results]",
            ])
        trc = ", trust_remote_code=True" if rec.trust_remote_code else ""
        return "\n".join([
            "# pip install sentence-transformers",
            "from sentence_transformers import CrossEncoder",
            "",
            f'reranker = CrossEncoder("{rec.model_id}", max_length={rec.max_tokens}{trc})',
            "",
            f"def rerank(query: str, candidates: list[str], k: int = {rec.final_k}) -> list[str]:",
            f"    # candidates: the top {rec.fetch_k} passages from dense/hybrid retrieval",
            "    scores = reranker.predict([(query, passage) for passage in candidates])",
            "    ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])",
            "    return [passage for passage, _ in ranked[:k]]",
        ])
