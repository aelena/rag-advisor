"""Tests for baseline comparison (CI regression mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_adviser.evaluators.baseline import BASELINE_VERSION, BaselineManager
from rag_adviser.evaluators.metrics import EvalMetrics
from rag_adviser.evaluators.pipeline_runner import EvalConfig, EvalReport


def _make_report(
    strategies: dict[str, dict[str, float]],
) -> EvalReport:
    """Helper to create an EvalReport from strategy metrics."""
    report = EvalReport(config=EvalConfig(), embedding_model="test-model")
    for name, metrics in strategies.items():
        report.strategy_results.append(
            EvalMetrics(
                strategy_name=name,
                num_queries=10,
                num_chunks=100,
                hit_rate=metrics.get("hit_rate", 0.8),
                mrr=metrics.get("mrr", 0.7),
                mean_precision_at_k=metrics.get("mean_precision_at_k", 0.6),
                mean_recall_at_k=metrics.get("mean_recall_at_k", 0.5),
                mean_ndcg_at_k=metrics.get("mean_ndcg_at_k", 0.65),
            )
        )
    return report


class TestBaselineManager:
    """Tests for save, load, and compare baseline."""

    def test_save_and_load(self, tmp_path: Path):
        mgr = BaselineManager()
        report = _make_report({
            "recursive": {"hit_rate": 0.85, "mrr": 0.72},
        })

        path = tmp_path / "baseline.json"
        mgr.save_baseline(report, path)
        assert path.exists()

        data = mgr.load_baseline(path)
        assert data["version"] == BASELINE_VERSION
        assert data["embedding_model"] == "test-model"
        assert "recursive" in data["strategies"]
        assert data["strategies"]["recursive"]["hit_rate"] == 0.85

    def test_load_nonexistent_raises(self, tmp_path: Path):
        mgr = BaselineManager()
        with pytest.raises(FileNotFoundError):
            mgr.load_baseline(tmp_path / "nope.json")

    def test_no_regression(self, tmp_path: Path):
        mgr = BaselineManager()

        # Save baseline
        baseline_report = _make_report({"recursive": {"hit_rate": 0.80, "mrr": 0.70}})
        path = tmp_path / "baseline.json"
        mgr.save_baseline(baseline_report, path)
        baseline = mgr.load_baseline(path)

        # Current results are the same
        current = _make_report({"recursive": {"hit_rate": 0.80, "mrr": 0.70}})
        comparison = mgr.compare(current, baseline)

        assert not comparison.has_regression
        assert len(comparison.regressions) == 0

    def test_detect_regression(self, tmp_path: Path):
        mgr = BaselineManager()

        baseline_report = _make_report({"recursive": {"hit_rate": 0.90, "mrr": 0.80}})
        path = tmp_path / "baseline.json"
        mgr.save_baseline(baseline_report, path)
        baseline = mgr.load_baseline(path)

        # Current results are worse
        current = _make_report({"recursive": {"hit_rate": 0.75, "mrr": 0.60}})
        comparison = mgr.compare(current, baseline, threshold=0.02)

        assert comparison.has_regression
        assert len(comparison.regressions) >= 2  # at least hit_rate and mrr

    def test_detect_improvement(self, tmp_path: Path):
        mgr = BaselineManager()

        baseline_report = _make_report({"recursive": {"hit_rate": 0.70, "mrr": 0.60}})
        path = tmp_path / "baseline.json"
        mgr.save_baseline(baseline_report, path)
        baseline = mgr.load_baseline(path)

        current = _make_report({"recursive": {"hit_rate": 0.90, "mrr": 0.85}})
        comparison = mgr.compare(current, baseline, threshold=0.02)

        assert not comparison.has_regression
        assert len(comparison.improvements) >= 2

    def test_custom_threshold(self, tmp_path: Path):
        mgr = BaselineManager()

        baseline_report = _make_report({"recursive": {"hit_rate": 0.80}})
        path = tmp_path / "baseline.json"
        mgr.save_baseline(baseline_report, path)
        baseline = mgr.load_baseline(path)

        # 1% drop with 5% threshold = no regression
        current = _make_report({"recursive": {"hit_rate": 0.79}})
        comparison = mgr.compare(current, baseline, threshold=0.05)
        assert not comparison.has_regression

        # Same drop with 0.5% threshold = regression
        comparison = mgr.compare(current, baseline, threshold=0.005)
        assert comparison.has_regression

    def test_missing_strategy_in_baseline(self, tmp_path: Path):
        mgr = BaselineManager()

        baseline_report = _make_report({"recursive": {"hit_rate": 0.80}})
        path = tmp_path / "baseline.json"
        mgr.save_baseline(baseline_report, path)
        baseline = mgr.load_baseline(path)

        # Current has a different strategy — no comparison possible
        current = _make_report({"semantic": {"hit_rate": 0.50}})
        comparison = mgr.compare(current, baseline)
        assert not comparison.has_regression
        assert len(comparison.regressions) == 0

    def test_multiple_strategies(self, tmp_path: Path):
        mgr = BaselineManager()

        baseline_report = _make_report({
            "recursive": {"hit_rate": 0.80, "mrr": 0.70},
            "semantic": {"hit_rate": 0.85, "mrr": 0.75},
        })
        path = tmp_path / "baseline.json"
        mgr.save_baseline(baseline_report, path)
        baseline = mgr.load_baseline(path)

        current = _make_report({
            "recursive": {"hit_rate": 0.82, "mrr": 0.72},  # improved
            "semantic": {"hit_rate": 0.60, "mrr": 0.50},  # regressed
        })
        comparison = mgr.compare(current, baseline, threshold=0.02)

        assert comparison.has_regression
        regression_strategies = {r.strategy for r in comparison.regressions}
        assert "semantic" in regression_strategies

    def test_baseline_json_structure(self, tmp_path: Path):
        mgr = BaselineManager()
        report = _make_report({"recursive": {}})
        path = tmp_path / "baseline.json"
        mgr.save_baseline(report, path)

        data = json.loads(path.read_text())
        assert "version" in data
        assert "embedding_model" in data
        assert "config" in data
        assert "strategies" in data
        assert "chunk_size" in data["config"]
