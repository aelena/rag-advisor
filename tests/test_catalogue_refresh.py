"""Tests for the MTEB catalogue refresh (no network)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_adviser.catalogue_refresh import (
    catalogue_entries,
    refresh_catalogue,
    retrieval_score_from_eval_results,
    rewrite_quality_scores,
)
from rag_adviser.config import load_defaults


@dataclass
class _Eval:
    task_type: str
    dataset_name: str
    metric_type: str
    metric_value: float


YAML = """# header comment
fallback_models:
  - model_id: org/a
    dimension: 384
    quality_score: 41.9
    notes: keep me

  - model_id: org/b
    quality_score: 50.0   # trailing comment is dropped only on this line if rewritten

api_embedding_models:
  - model_id: openai/x
    quality_score: 55.4
"""


class TestScore:
    def test_averages_retrieval_ndcg_with_cqadupstack_grouped(self) -> None:
        results = [
            _Eval("Retrieval", "MTEB ArguAna", "ndcg_at_10", 60.0),
            _Eval("Retrieval", "MTEB NFCorpus", "ndcg_at_10", 0.40),  # fraction scale
            _Eval("Retrieval", "MTEB CQADupstackAndroidRetrieval", "ndcg_at_10", 30.0),
            _Eval("Retrieval", "MTEB CQADupstackGamingRetrieval", "ndcg_at_10", 50.0),
            _Eval("Retrieval", "MTEB SciFact", "ndcg_at_10_std", 1.0),   # ignored
            _Eval("Classification", "MTEB Banking77", "accuracy", 90.0),  # ignored
            _Eval("Retrieval", "MTEB SomethingElse", "ndcg_at_10", 99.0),  # not in set
        ]
        score, n = retrieval_score_from_eval_results(results)
        # ArguAna 60, NFCorpus 40, CQADupstack mean 40 -> 46.7 over 3 datasets
        assert n == 3
        assert score == 46.7

    def test_nq_requires_word_match(self) -> None:
        results = [_Eval("Retrieval", "MTEB NQ", "ndcg_at_10", 55.0),
                   _Eval("Retrieval", "MTEB SomethingNQish", "ndcg_at_10", 5.0)]
        score, n = retrieval_score_from_eval_results(results)
        assert (score, n) == (55.0, 1)

    def test_empty(self) -> None:
        assert retrieval_score_from_eval_results([]) == (None, 0)


class TestYamlSurgery:
    def test_entries_only_from_requested_section(self) -> None:
        assert catalogue_entries(YAML) == {"org/a": 41.9, "org/b": 50.0}
        assert catalogue_entries(YAML, "api_embedding_models") == {"openai/x": 55.4}

    def test_rewrite_preserves_everything_else(self) -> None:
        out = rewrite_quality_scores(YAML, {"org/a": 52.3})
        assert "quality_score: 52.3" in out
        assert "# header comment" in out and "notes: keep me" in out
        assert "quality_score: 50.0" in out            # untouched sibling
        assert "quality_score: 55.4" in out            # other section untouched
        assert out.count("\n") == YAML.count("\n")

    def test_real_catalogue_parses(self) -> None:
        from rag_adviser.catalogue_refresh import DEFAULTS_PATH

        entries = catalogue_entries(DEFAULTS_PATH.read_text(encoding="utf-8"))
        expected = {m["model_id"] for m in load_defaults()["fallback_models"]}
        assert set(entries) == expected


class TestRefresh:
    def test_dry_run_and_write(self, tmp_path: Path) -> None:
        path = tmp_path / "defaults.yaml"
        path.write_text(YAML, encoding="utf-8")

        def fake_fetch(model_id: str):
            return {
                "org/a": (53.3, 15, "ok"),           # moved -> updated
                "org/b": (50.2, 15, "ok"),           # within min_delta -> unchanged
            }.get(model_id, (None, 0, "no_card: 404"))

        report = refresh_catalogue(path, write=False, fetch=fake_fetch)
        statuses = {r.model_id: r.status for r in report.results}
        assert statuses == {"org/a": "updated", "org/b": "unchanged"}
        assert report.updated[0].delta == 11.4
        assert report.written is False
        assert "quality_score: 41.9" in path.read_text(encoding="utf-8")

        report = refresh_catalogue(path, write=True, fetch=fake_fetch)
        assert report.written is True
        assert "quality_score: 53.3" in path.read_text(encoding="utf-8")

    def test_insufficient_data_is_not_written(self, tmp_path: Path) -> None:
        path = tmp_path / "defaults.yaml"
        path.write_text(YAML, encoding="utf-8")
        report = refresh_catalogue(path, write=True, fetch=lambda m: (70.0, 3, "ok"))
        assert all(r.status == "insufficient_data" for r in report.results)
        assert report.written is False
