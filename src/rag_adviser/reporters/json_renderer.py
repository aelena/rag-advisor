"""Machine-readable JSON configuration renderer (same content as the YAML config)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_adviser.models import Recommendations, UserAnswers
from rag_adviser.reporters.yaml_renderer import YamlRenderer


class JsonRenderer(YamlRenderer):
    """Write the recommendation config as ``rag_config.json``."""

    def render(
        self,
        answers: UserAnswers,
        recommendations: Recommendations,
        output_dir: Path,
    ) -> Path:
        config = self._build_config(answers, recommendations)
        out_path = output_dir / "rag_config.json"
        out_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return out_path
