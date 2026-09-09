"""HTML report renderer using Jinja2 templates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from rag_adviser import __version__
from rag_adviser.models import Recommendations, UserAnswers

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class HtmlRenderer:
    """Generate a styled HTML report using Jinja2 templates."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=True,
        )

    def render(
        self,
        answers: UserAnswers,
        recommendations: Recommendations,
        output_dir: Path,
    ) -> Path:
        """Render the HTML report and write to disk."""
        template = self._env.get_template("report.html.j2")

        # Load CSS
        css_path = _TEMPLATES_DIR / "report.css"
        css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = template.render(
            answers=answers,
            recommendations=recommendations,
            generated_at=now,
            version=__version__,
            css=css,
        )

        out_path = output_dir / "rag_report.html"
        out_path.write_text(html, encoding="utf-8")
        return out_path
