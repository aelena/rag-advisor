"""Report generation orchestrator — dispatches to format-specific renderers."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_adviser.models import (
    Recommendations,
    ReportFormat,
    ReportGenerationError,
    UserAnswers,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Orchestrate report generation across multiple formats."""

    def generate(
        self,
        answers: UserAnswers,
        recommendations: Recommendations,
        output_dir: Path,
        formats: list[ReportFormat],
    ) -> list[Path]:
        """Generate reports in all requested formats.

        Args:
            answers: The user's collected answers.
            recommendations: The generated recommendations.
            output_dir: Directory to write report files.
            formats: List of output formats to generate.

        Returns:
            List of paths to generated report files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Expand ALL to individual formats
        if ReportFormat.ALL in formats:
            formats = [
                ReportFormat.MARKDOWN,
                ReportFormat.HTML,
                ReportFormat.YAML,
            ]

        generated: list[Path] = []

        for fmt in formats:
            try:
                path = self._render_format(fmt, answers, recommendations, output_dir)
                generated.append(path)
                logger.info("Generated %s report: %s", fmt.value, path)
            except ReportGenerationError as e:
                # Log but don't fail the whole pipeline for one format
                logger.warning("Failed to generate %s report: %s", fmt.value, e)
            except Exception as e:
                logger.warning("Unexpected error generating %s: %s", fmt.value, e)

        return generated

    def _render_format(
        self,
        fmt: ReportFormat,
        answers: UserAnswers,
        recommendations: Recommendations,
        output_dir: Path,
    ) -> Path:
        """Render a single format using the appropriate renderer."""
        if fmt == ReportFormat.MARKDOWN:
            from rag_adviser.reporters.markdown_renderer import MarkdownRenderer
            return MarkdownRenderer().render(answers, recommendations, output_dir)

        elif fmt == ReportFormat.HTML:
            from rag_adviser.reporters.html_renderer import HtmlRenderer
            return HtmlRenderer().render(answers, recommendations, output_dir)

        elif fmt == ReportFormat.YAML:
            from rag_adviser.reporters.yaml_renderer import YamlRenderer
            return YamlRenderer().render(answers, recommendations, output_dir)

        else:
            raise ReportGenerationError(f"Unknown report format: {fmt}")
