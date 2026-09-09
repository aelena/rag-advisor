"""Embedding model finder with HuggingFace Hub API integration and offline fallback."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from rag_adviser.analyzers.constraint_analyzer import ConstraintAnalyzer
from rag_adviser.config import load_defaults
from rag_adviser.models import (
    BudgetTier,
    DocumentStats,
    EmbeddingModelRecommendation,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    UseCase,
)

logger = logging.getLogger(__name__)

# Overall wall-clock budget for all HuggingFace Hub queries combined.
_HF_QUERY_TIMEOUT_S = 20.0
_HF_MAX_WORKERS = 4

# License strings that forbid commercial use.
_NON_COMMERCIAL_MARKERS = ("nc", "non-commercial", "noncommercial", "research-only")


class HFModelFinder:
    """Find and score embedding models from HuggingFace Hub + curated fallbacks.

    Candidate sources:
    - Curated open models from ``defaults.yaml`` (always).
    - Hosted API embedding endpoints from ``defaults.yaml`` when the budget
      allows paid APIs and privacy permits sending text to a third party.
    - Live HuggingFace Hub results from trusted organisations, unless offline
      or privacy is strict/air-gapped.

    Scoring combines retrieval quality (approximate MTEB nDCG@10), hardware
    fit, language coverage, use-case fit, latency, licensing and community
    adoption into a 0-1 fitness score.
    """

    def __init__(self, offline: bool = False) -> None:
        self._offline = offline or bool(os.environ.get("HF_HUB_OFFLINE"))
        self._defaults = load_defaults()
        self._trusted_orgs: list[str] = self._defaults.get("trusted_orgs", [])
        self._fallback_models: list[dict] = self._defaults.get("fallback_models", [])
        self._api_models: list[dict] = self._defaults.get("api_embedding_models", [])

    # ── Public API ─────────────────────────────────────────────────────────

    def find_models(
        self,
        doc_stats: DocumentStats | None,
        constraints: HardwareConstraints,
        use_case: UseCase,
        future_languages: bool = False,
    ) -> list[EmbeddingModelRecommendation]:
        """Find and rank embedding models matching the given criteria.

        Returns up to 5 models sorted by composite fitness score.
        """
        languages = list(doc_stats.languages_detected.keys()) if doc_stats else ["en"]
        languages = [lang for lang in languages if lang != "unknown"] or ["en"]
        multilingual = (
            future_languages
            or (doc_stats is not None and doc_stats.has_cjk)
            or len(languages) > 1
            or languages[0] != "en"
        )

        max_size_gb = ConstraintAnalyzer().get_max_model_size(constraints)
        candidates = self._get_candidates(languages, multilingual, constraints)

        scored = [
            self._score_model(
                c, languages, multilingual, max_size_gb, constraints, use_case
            )
            for c in candidates
        ]

        # Deduplicate by model_id, keeping the entry with richer metadata and
        # the union of reasons.
        best_by_id: dict[str, EmbeddingModelRecommendation] = {}
        for m in sorted(scored, key=lambda m: (m.score, m.likes), reverse=True):
            existing = best_by_id.get(m.model_id)
            if existing is None:
                best_by_id[m.model_id] = m
                continue
            merged = existing if existing.dimension > 0 else m
            other = m if merged is existing else existing
            merged.score = max(existing.score, m.score)
            merged.likes = max(existing.likes, m.likes)
            merged.downloads = max(existing.downloads, m.downloads)
            for r in other.reasons:
                if r not in merged.reasons:
                    merged.reasons.append(r)
            best_by_id[m.model_id] = merged

        unique = sorted(best_by_id.values(), key=lambda m: (m.score, m.likes), reverse=True)
        return unique[:5]

    # ── Candidate gathering ────────────────────────────────────────────────

    def _get_candidates(
        self,
        languages: list[str],
        multilingual: bool,
        constraints: HardwareConstraints,
    ) -> list[dict]:
        """Gather candidate models from curated lists and/or the Hub API."""
        candidates = [dict(m, provider=m.get("provider", "huggingface"))
                      for m in self._fallback_models]

        api_allowed = ConstraintAnalyzer().is_api_allowed(constraints)
        if api_allowed and constraints.budget == BudgetTier.PAID_API:
            candidates.extend(dict(m) for m in self._api_models)

        if not self._offline and api_allowed:
            candidates.extend(self._query_hf_api(languages, multilingual))

        return candidates

    def _query_hf_api(self, languages: list[str], multilingual: bool) -> list[dict]:
        """Query HuggingFace Hub for sentence-transformer models from trusted orgs.

        Queries run concurrently under a single wall-clock budget so a slow or
        unreachable Hub cannot stall the CLI.
        """
        try:
            from huggingface_hub import list_models
        except ImportError:
            logger.warning("huggingface_hub not installed, using fallback models only")
            return []

        def fetch(org: str) -> list[dict]:
            out: list[dict] = []
            models = list_models(
                author=org,
                filter="sentence-transformers",
                sort="likes",
                direction=-1,
                limit=10,
            )
            for model in models:
                if self._matches_criteria(model, languages, multilingual):
                    parsed = self._parse_api_model(model)
                    if parsed:
                        out.append(parsed)
            return out

        api_models: list[dict] = []
        with ThreadPoolExecutor(max_workers=_HF_MAX_WORKERS) as pool:
            futures = {pool.submit(fetch, org): org for org in self._trusted_orgs}
            try:
                for fut in as_completed(futures, timeout=_HF_QUERY_TIMEOUT_S):
                    org = futures[fut]
                    try:
                        api_models.extend(fut.result())
                    except Exception as e:  # network / API errors are non-fatal
                        logger.debug("Failed to query org %s: %s", org, e)
            except FuturesTimeoutError:  # distinct from builtin TimeoutError on 3.10
                logger.warning(
                    "HuggingFace Hub queries exceeded %.0fs; using partial results",
                    _HF_QUERY_TIMEOUT_S,
                )
                for fut in futures:
                    fut.cancel()
        return api_models

    def _matches_criteria(self, model: Any, languages: list[str], multilingual: bool) -> bool:
        """Check if an API model matches language/task criteria."""
        tags = [t.lower() for t in (model.tags or [])]

        if not any(t in tags for t in ["sentence-similarity", "feature-extraction", "embeddings"]):
            return False

        if multilingual:
            return "multilingual" in tags or all(lang in tags for lang in languages)
        return any(lang in tags for lang in languages) or "en" in tags

    def _parse_api_model(self, model: Any) -> dict | None:
        """Parse an API model response into a candidate dict."""
        try:
            card = getattr(model, "card_data", None)
            model_id = model.id or ""
            tags = [t.lower() for t in (model.tags or [])]
            license_ = getattr(card, "license", None) if card else None
            if isinstance(license_, list):
                license_ = license_[0] if license_ else None
            if not license_:
                license_ = next((t[8:] for t in tags if t.startswith("license:")), "unknown")

            return {
                "model_id": model_id,
                "likes": model.likes or 0,
                "downloads": model.downloads or 0,
                "tags": model.tags or [],
                # The Hub does not expose dimension/max length in list results;
                # the curated list fills these in for known models on merge.
                "dimension": 0,
                "max_tokens": 512,
                "estimated_size_gb": self._estimate_size(model_id),
                "license": str(license_),
                "release_date": (
                    model.created_at.strftime("%Y-%m-%d")
                    if getattr(model, "created_at", None)
                    else "unknown"
                ),
                "multilingual": "multilingual" in tags,
                "provider": "huggingface",
                "quality_score": 0.0,
            }
        except Exception as e:
            logger.debug("Failed to parse model: %s", e)
            return None

    @staticmethod
    def _estimate_size(model_id: str) -> float:
        """Estimate fp32 checkpoint size (GB) from name heuristics."""
        lower = model_id.lower()
        if "large" in lower or "7b" in lower:
            return 1.5
        if "small" in lower or "mini" in lower or "tiny" in lower:
            return 0.1
        return 0.5

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score_model(
        self,
        candidate: dict,
        languages: list[str],
        multilingual: bool,
        max_size_gb: float,
        constraints: HardwareConstraints,
        use_case: UseCase,
    ) -> EmbeddingModelRecommendation:
        """Score a candidate model based on fitness for the user's requirements."""
        score = 0.3  # Baseline; components below add up to ~1.0 for a perfect fit
        reasons: list[str] = []
        warnings: list[str] = []

        provider = candidate.get("provider", "huggingface")
        is_api = provider != "huggingface"
        model_size = float(candidate.get("estimated_size_gb", 0.5))
        is_multilingual = bool(candidate.get("multilingual", False))
        model_id = candidate.get("model_id", "")
        tags = [str(t).lower() for t in candidate.get("tags", [])]
        quality = float(candidate.get("quality_score", 0.0) or 0.0)
        license_ = str(candidate.get("license", "unknown")).lower()
        max_tokens = int(candidate.get("max_tokens", 512))
        notes = candidate.get("notes")

        # ── Retrieval quality (approximate MTEB nDCG@10) ───────────────────
        if quality > 0:
            # 40 -> +0.0, 60 -> +0.30 (linear, clamped)
            q_bonus = max(0.0, min(1.0, (quality - 40.0) / 20.0)) * 0.30
            score += q_bonus
            reasons.append(f"Retrieval quality ~{quality:.0f} (MTEB nDCG@10, approx.)")
        else:
            warnings.append("No benchmark quality data for this model")

        # ── Hardware fit ───────────────────────────────────────────────────
        if is_api:
            score += 0.10
            reasons.append(f"Hosted {provider} API: no local model hosting required")
            price = candidate.get("price_per_million_tokens")
            if price is not None:
                reasons.append(f"~${price}/1M tokens embedded")
            warnings.append("Documents and queries are sent to a third-party API")
        elif model_size <= max_size_gb:
            score += 0.15
            reasons.append(f"Fits hardware limit ({model_size}GB <= {max_size_gb:.1f}GB)")
        else:
            score -= 0.40
            warnings.append(
                f"Model size ({model_size}GB fp32) exceeds hardware limit "
                f"({max_size_gb:.1f}GB); consider fp16/int8 loading"
            )

        # ── Language coverage ──────────────────────────────────────────────
        if multilingual:
            if is_multilingual:
                score += 0.25
                reasons.append("Supports multilingual content")
            else:
                score -= 0.30
                warnings.append(
                    f"Not multilingual; corpus languages: {', '.join(languages)}"
                )
        elif not is_multilingual:
            score += 0.05
            reasons.append("English-optimised model for a single-language corpus")

        # ── Use case fitness ───────────────────────────────────────────────
        if use_case == UseCase.CODE:
            if any("code" in t for t in tags):
                score += 0.20
                reasons.append("Code-specialised model matches your use case")
            else:
                warnings.append("General-purpose model; consider a code-specialised embedder")
        elif use_case in (UseCase.LEGAL, UseCase.SUMMARIZATION) and max_tokens >= 2048:
            score += 0.05
            reasons.append("Long input window suits long, structured documents")

        # ── Latency fitness ────────────────────────────────────────────────
        if constraints.latency_budget == LatencyBudget.FAST:
            if is_api:
                score -= 0.05
                warnings.append("API round-trip adds ~50-200ms to each query")
            elif model_size < 0.3:
                score += 0.10
                reasons.append("Small model supports fast inference")
            elif model_size > 1.0 and constraints.hardware == HardwareProfile.CPU_ONLY:
                score -= 0.15
                warnings.append("Large model may not meet <500ms latency on CPU")

        # ── Licensing ──────────────────────────────────────────────────────
        if any(marker in license_ for marker in _NON_COMMERCIAL_MARKERS):
            score -= 0.10
            warnings.append(f"License '{license_}' restricts commercial use")

        # ── Popularity / community trust ───────────────────────────────────
        likes = int(candidate.get("likes", 0) or 0)
        downloads = int(candidate.get("downloads", 0) or 0)
        if likes > 1000 or downloads > 500_000:
            score += 0.05
            reasons.append(f"Well-validated by community ({likes} likes)")

        # ── Long context bonus ─────────────────────────────────────────────
        if max_tokens >= 8192:
            score += 0.05
            reasons.append(f"Supports long inputs ({max_tokens} tokens)")

        trust_remote_code = bool(candidate.get("trust_remote_code", False))
        if trust_remote_code:
            warnings.append(
                "Loads custom model code from the Hub: pass trust_remote_code=True "
                "and review the repository before use in restricted environments"
            )

        if notes:
            reasons.append(str(notes))

        score = max(0.0, min(1.0, score))

        return EmbeddingModelRecommendation(
            model_id=model_id,
            likes=likes,
            downloads=downloads,
            tags=candidate.get("tags", []),
            dimension=int(candidate.get("dimension", 0) or 0),
            max_tokens=max_tokens,
            estimated_size_gb=model_size,
            license=str(candidate.get("license", "unknown")),
            release_date=str(candidate.get("release_date", "unknown")),
            multilingual=is_multilingual,
            provider=provider,
            trust_remote_code=trust_remote_code,
            price_per_million_tokens=(
                float(candidate["price_per_million_tokens"])
                if candidate.get("price_per_million_tokens") is not None else None
            ),
            quality_score=quality,
            score=score,
            reasons=reasons,
            warnings=warnings,
        )
