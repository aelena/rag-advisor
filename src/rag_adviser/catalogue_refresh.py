"""Refresh embedding-model quality scores in defaults.yaml from Hugging Face model cards.

Most embedding model repositories publish their MTEB results in the model
card's ``model-index`` block. This module reads those results, averages the
English MTEB *retrieval* nDCG@10 datasets the same way the leaderboard does
(the twelve CQADupstack subsets count once), and rewrites the
``quality_score`` lines of the curated catalogue in place, preserving every
comment and the file layout.

Hosted API models have no model card and are left untouched; their prices and
scores stay manual. Rerankers rarely publish model-index data, so they are
skipped as well.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_adviser.config import _CONFIG_DIR

logger = logging.getLogger(__name__)

DEFAULTS_PATH = _CONFIG_DIR / "defaults.yaml"

# MTEB (English, v1) retrieval datasets. CQADupstack appears as 12 subsets in
# model cards and is averaged into a single entry before the overall mean.
RETRIEVAL_DATASETS = (
    "ArguAna", "ClimateFEVER", "CQADupstack", "DBPedia", "FEVER", "FiQA2018",
    "HotpotQA", "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS",
    "SciFact", "Touche2020", "TRECCOVID",
)
_NDCG_METRICS = {"ndcg_at_10", "ndcg@10", "ndcg_at_10_std"}
DEFAULT_MIN_DATASETS = 10


@dataclass
class RefreshResult:
    """Outcome for one catalogue entry."""

    model_id: str
    old_score: float
    new_score: float | None = None
    datasets_used: int = 0
    status: str = ""  # updated | unchanged | insufficient_data | no_card | error | skipped
    detail: str = ""

    @property
    def delta(self) -> float | None:
        return None if self.new_score is None else round(self.new_score - self.old_score, 1)


@dataclass
class RefreshReport:
    results: list[RefreshResult] = field(default_factory=list)
    written: bool = False

    @property
    def updated(self) -> list[RefreshResult]:
        return [r for r in self.results if r.status == "updated"]


# ── Score computation ──────────────────────────────────────────────────────


def _dataset_key(name: str) -> str | None:
    """Map a model-card dataset name onto one of RETRIEVAL_DATASETS, or None."""
    lowered = name.lower()
    for ds in RETRIEVAL_DATASETS:
        if ds.lower() in lowered:
            # "NQ" is a substring of several names; require a word boundary-ish match.
            if ds == "NQ" and not re.search(r"\bnq\b", lowered):
                continue
            return ds
    return None


def retrieval_score_from_eval_results(eval_results: list[Any]) -> tuple[float | None, int]:
    """Average MTEB retrieval nDCG@10 (0-100) from model-card eval results.

    ``eval_results`` items need ``task_type``, ``dataset_name``, ``metric_type``
    and ``metric_value`` attributes (huggingface_hub.EvalResult shape).
    Returns ``(score, number_of_datasets)``; score is None when nothing matched.
    """
    per_dataset: dict[str, list[float]] = {}
    for r in eval_results:
        task = str(getattr(r, "task_type", "") or "").lower()
        metric = str(getattr(r, "metric_type", "") or "").lower()
        if task != "retrieval" or metric not in _NDCG_METRICS or metric.endswith("_std"):
            continue
        key = _dataset_key(str(getattr(r, "dataset_name", "") or ""))
        if key is None:
            continue
        try:
            value = float(r.metric_value)
        except (TypeError, ValueError):
            continue
        if value <= 1.0:  # some cards report fractions
            value *= 100.0
        per_dataset.setdefault(key, []).append(value)

    if not per_dataset:
        return None, 0
    # Average subsets (CQADupstack) first, then across datasets.
    means = [sum(v) / len(v) for v in per_dataset.values()]
    return round(sum(means) / len(means), 1), len(means)


def fetch_retrieval_score(model_id: str) -> tuple[float | None, int, str]:
    """Load a model card from the Hub and compute its retrieval score.

    Returns ``(score, datasets, status)`` where status is "ok", "no_card" or "error".
    """
    try:
        from huggingface_hub import ModelCard
    except ImportError:
        return None, 0, "error: huggingface_hub not installed"
    try:
        card = ModelCard.load(model_id)
    except Exception as e:  # network, 404, gated
        return None, 0, f"no_card: {type(e).__name__}"
    eval_results = getattr(card.data, "eval_results", None) or []
    score, n = retrieval_score_from_eval_results(eval_results)
    return score, n, "ok"


# ── YAML surgery ───────────────────────────────────────────────────────────

_MODEL_LINE = re.compile(r"^(\s*)- model_id:\s*(\S+)\s*$")
_QUALITY_LINE = re.compile(r"^(\s*)quality_score:\s*([0-9.]+)\s*(#.*)?$")


def catalogue_entries(text: str, section: str = "fallback_models") -> dict[str, float]:
    """Return {model_id: quality_score} for a top-level list section of the YAML text."""
    entries: dict[str, float] = {}
    in_section = False
    current: str | None = None
    for line in text.splitlines():
        if re.match(rf"^{re.escape(section)}:\s*$", line):
            in_section = True
            continue
        if in_section and re.match(r"^\S", line) and not line.startswith("#"):
            break  # next top-level key
        if not in_section:
            continue
        m = _MODEL_LINE.match(line)
        if m:
            current = m.group(2)
            continue
        q = _QUALITY_LINE.match(line)
        if q and current:
            entries[current] = float(q.group(2))
    return entries


def rewrite_quality_scores(
    text: str, updates: dict[str, float], section: str = "fallback_models"
) -> str:
    """Replace quality_score values for the given model ids inside ``section`` only."""
    out: list[str] = []
    in_section = False
    current: str | None = None
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if re.match(rf"^{re.escape(section)}:\s*$", bare):
            in_section = True
        elif in_section and re.match(r"^\S", bare) and not bare.startswith("#"):
            in_section = False
        if in_section:
            m = _MODEL_LINE.match(bare)
            if m:
                current = m.group(2)
            else:
                q = _QUALITY_LINE.match(bare)
                if q and current in updates:
                    newline = line[len(bare):]
                    comment = f"  {q.group(3)}" if q.group(3) else ""
                    line = f"{q.group(1)}quality_score: {updates[current]:.1f}{comment}{newline}"
        out.append(line)
    return "".join(out)


# ── Orchestration ──────────────────────────────────────────────────────────


def refresh_catalogue(
    path: Path = DEFAULTS_PATH,
    write: bool = False,
    min_datasets: int = DEFAULT_MIN_DATASETS,
    min_delta: float = 0.5,
    fetch=fetch_retrieval_score,
) -> RefreshReport:
    """Recompute quality scores for every local model in the catalogue.

    ``fetch`` is injectable for tests. Scores that moved by less than
    ``min_delta`` are reported as unchanged and not rewritten.
    """
    text = path.read_text(encoding="utf-8")
    entries = catalogue_entries(text)
    report = RefreshReport()
    updates: dict[str, float] = {}

    for model_id, old in entries.items():
        result = RefreshResult(model_id=model_id, old_score=old)
        score, n, status = fetch(model_id)
        result.datasets_used = n
        if status != "ok":
            result.status = status.split(":")[0]
            result.detail = status
        elif score is None or n < min_datasets:
            result.status = "insufficient_data"
            result.detail = f"{n} retrieval datasets found (need {min_datasets})"
        else:
            result.new_score = score
            if abs(score - old) < min_delta:
                result.status = "unchanged"
            else:
                result.status = "updated"
                updates[model_id] = score
        report.results.append(result)

    if write and updates:
        path.write_text(rewrite_quality_scores(text, updates), encoding="utf-8")
        report.written = True
    return report
