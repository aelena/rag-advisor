"""Baseline comparison for CI/regression testing of RAG evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag_adviser.evaluators.pipeline_runner import EvalReport

logger = logging.getLogger(__name__)

BASELINE_VERSION = 1

METRIC_KEYS = [
    "hit_rate",
    "mrr",
    "mean_precision_at_k",
    "mean_recall_at_k",
    "mean_ndcg_at_k",
]

METRIC_DISPLAY = {
    "hit_rate": "Hit Rate",
    "mrr": "MRR",
    "mean_precision_at_k": "Precision@k",
    "mean_recall_at_k": "Recall@k",
    "mean_ndcg_at_k": "NDCG@k",
}


@dataclass
class MetricDiff:
    """Difference in a single metric between baseline and current."""

    strategy: str = ""
    metric: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    delta: float = 0.0
    is_regression: bool = False
    is_improvement: bool = False


@dataclass
class BaselineComparison:
    """Result of comparing current metrics against a baseline."""

    regressions: list[MetricDiff] = field(default_factory=list)
    improvements: list[MetricDiff] = field(default_factory=list)
    unchanged: list[MetricDiff] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return len(self.regressions) > 0


class BaselineManager:
    """Save, load, and compare evaluation baselines."""

    DEFAULT_THRESHOLD = 0.02  # 2% regression threshold

    def save_baseline(self, report: EvalReport, path: Path) -> None:
        """Serialize EvalReport metrics to a baseline JSON file."""
        data = {
            "version": BASELINE_VERSION,
            "embedding_model": report.embedding_model,
            "config": {
                "chunk_size": report.config.chunk_size,
                "chunk_overlap": report.config.chunk_overlap,
                "top_k": report.config.top_k,
                "vector_backend": report.config.vector_backend,
            },
            "strategies": {},
        }

        for metrics in report.strategy_results:
            data["strategies"][metrics.strategy_name] = {
                key: getattr(metrics, key) for key in METRIC_KEYS
            }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_baseline(self, path: Path) -> dict:
        """Load a saved baseline JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def compare(
        self,
        current: EvalReport,
        baseline: dict,
        threshold: float | None = None,
    ) -> BaselineComparison:
        """Compare current results against a baseline.

        A regression is flagged when a metric drops by more than `threshold`.
        """
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD

        comparison = BaselineComparison()
        baseline_strategies = baseline.get("strategies", {})

        for metrics in current.strategy_results:
            name = metrics.strategy_name
            if name not in baseline_strategies:
                continue

            baseline_metrics = baseline_strategies[name]

            for key in METRIC_KEYS:
                current_val = getattr(metrics, key)
                baseline_val = baseline_metrics.get(key)
                if baseline_val is None:
                    continue

                delta = current_val - baseline_val
                diff = MetricDiff(
                    strategy=name,
                    metric=key,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    delta=delta,
                )

                if delta < -threshold:
                    diff.is_regression = True
                    comparison.regressions.append(diff)
                elif delta > threshold:
                    diff.is_improvement = True
                    comparison.improvements.append(diff)
                else:
                    comparison.unchanged.append(diff)

        return comparison

    def display_diff_table(
        self, comparison: BaselineComparison, console: Console | None = None
    ) -> None:
        """Show a Rich table comparing baseline vs current metrics."""
        console = console or Console()

        table = Table(
            title="Baseline Comparison",
            show_header=True,
            title_style="bold cyan",
        )
        table.add_column("Strategy", style="bold")
        table.add_column("Metric")
        table.add_column("Baseline", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Status")

        all_diffs = sorted(
            comparison.regressions + comparison.improvements + comparison.unchanged,
            key=lambda d: (d.strategy, d.metric),
        )

        for diff in all_diffs:
            display_name = METRIC_DISPLAY.get(diff.metric, diff.metric)

            if diff.is_regression:
                style = "bold red"
                status = "REGRESSION"
                delta_str = f"{diff.delta:+.3f}"
            elif diff.is_improvement:
                style = "bold green"
                status = "improved"
                delta_str = f"{diff.delta:+.3f}"
            else:
                style = "dim"
                status = "ok"
                delta_str = f"{diff.delta:+.3f}"

            table.add_row(
                diff.strategy,
                display_name,
                f"{diff.baseline_value:.3f}",
                f"{diff.current_value:.3f}",
                delta_str,
                status,
                style=style,
            )

        console.print()
        console.print(table)

        if comparison.has_regression:
            console.print(
                f"\n[bold red]Found {len(comparison.regressions)} regression(s)[/]"
            )
        else:
            console.print("\n[bold green]No regressions detected[/]")
