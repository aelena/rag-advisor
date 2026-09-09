"""Preset manager — load built-in presets, save/load custom presets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag_adviser.models import (
    AnswerType,
    BudgetTier,
    ContentType,
    DeploymentTarget,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    QueryComplexity,
    QueryType,
    UpdateFrequency,
    UseCase,
    UserAnswers,
)

logger = logging.getLogger(__name__)

_BUILTIN_DIR = Path(__file__).parent / "builtin"
_USER_DIR = Path.home() / ".ragadvisor" / "presets"


@dataclass
class PresetProfile:
    """A named preset with description and answer values."""

    name: str = ""
    description: str = ""
    answers: dict = field(default_factory=dict)
    source: str = "builtin"  # "builtin" or "custom"


class PresetManager:
    """Manage built-in and user-created presets."""

    def list_presets(self) -> list[PresetProfile]:
        """List all available presets (built-in + custom)."""
        presets = []
        # Built-in
        if _BUILTIN_DIR.exists():
            for f in sorted(_BUILTIN_DIR.glob("*.yaml")):
                try:
                    p = self._load_file(f, source="builtin")
                    presets.append(p)
                except Exception:
                    logger.warning("Failed to load preset: %s", f)
        # Custom
        if _USER_DIR.exists():
            for f in sorted(_USER_DIR.glob("*.yaml")):
                try:
                    p = self._load_file(f, source="custom")
                    presets.append(p)
                except Exception:
                    logger.warning("Failed to load custom preset: %s", f)
        return presets

    def load_preset(self, name: str) -> PresetProfile:
        """Load a preset by name. Searches built-in first, then custom."""
        # Try built-in
        builtin_path = _BUILTIN_DIR / f"{name}.yaml"
        if builtin_path.exists():
            return self._load_file(builtin_path, source="builtin")

        # Try custom
        custom_path = _USER_DIR / f"{name}.yaml"
        if custom_path.exists():
            return self._load_file(custom_path, source="custom")

        # Fuzzy match: try replacing hyphens/underscores
        for directory, source in [
            (_BUILTIN_DIR, "builtin"),
            (_USER_DIR, "custom"),
        ]:
            if directory.exists():
                for f in directory.glob("*.yaml"):
                    if f.stem.replace("-", "_") == name.replace("-", "_"):
                        return self._load_file(f, source=source)

        available = [p.name for p in self.list_presets()]
        raise ValueError(
            f"Preset '{name}' not found. Available: {', '.join(available) or 'none'}"
        )

    def apply_preset(self, preset: PresetProfile) -> UserAnswers:
        """Convert a preset profile into a UserAnswers object."""
        answers = UserAnswers()
        data = preset.answers

        # Phase 1
        if "content_type_override" in data:
            answers.content_type_override = ContentType(
                data["content_type_override"]
            )
        answers.future_languages = data.get("future_languages", False)

        # Phase 2
        if "use_case" in data:
            answers.use_case = UseCase(data["use_case"])

        constraints = HardwareConstraints()
        if "environment" in data:
            constraints.environment = DeploymentTarget(data["environment"])
        if "latency_budget" in data:
            constraints.latency_budget = LatencyBudget(data["latency_budget"])
        if "hardware" in data:
            constraints.hardware = HardwareProfile(data["hardware"])
        if "ram_gb" in data:
            constraints.ram_gb = float(data["ram_gb"])
        if "vram_gb" in data:
            constraints.vram_gb = float(data["vram_gb"])
        if "budget" in data:
            constraints.budget = BudgetTier(data["budget"])
        if "privacy" in data:
            constraints.privacy = PrivacyLevel(data["privacy"])
        answers.constraints = constraints

        if "embedding_provider" in data:
            answers.embedding_provider = data["embedding_provider"]
        if "llm_provider" in data:
            answers.llm_provider = data["llm_provider"]
        if "preferred_libraries" in data:
            answers.preferred_libraries = list(data["preferred_libraries"])

        # Phase 3
        if "query_type" in data:
            answers.query_type = QueryType(data["query_type"])
        if "query_complexity" in data:
            answers.query_complexity = QueryComplexity(data["query_complexity"])
        if "expected_answer_type" in data:
            answers.expected_answer_type = AnswerType(
                data["expected_answer_type"]
            )
        if "update_frequency" in data:
            answers.update_frequency = UpdateFrequency(
                data["update_frequency"]
            )
        if "has_ground_truth" in data:
            answers.has_ground_truth = data["has_ground_truth"]
        if "expected_queries_per_day" in data:
            answers.expected_queries_per_day = int(data["expected_queries_per_day"])

        return answers

    def save_preset(
        self,
        name: str,
        answers: UserAnswers,
        description: str = "",
        fmt: str = "yaml",
    ) -> Path:
        """Save current answers as a custom preset."""
        _USER_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "name": name,
            "description": description or f"Custom preset: {name}",
            "answers": self._answers_to_dict(answers),
        }

        ext = "json" if fmt == "json" else "yaml"
        path = _USER_DIR / f"{name}.{ext}"

        if ext == "json":
            import json

            path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        else:
            path.write_text(
                yaml.dump(
                    data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

        return path

    def _load_file(self, path: Path, source: str = "builtin") -> PresetProfile:
        """Load a preset from a YAML or JSON file."""
        text = path.read_text(encoding="utf-8")

        if path.suffix == ".json":
            import json

            data = json.loads(text)
        else:
            data = yaml.safe_load(text)

        return PresetProfile(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            answers=data.get("answers", {}),
            source=source,
        )

    def _answers_to_dict(self, answers: UserAnswers) -> dict:
        """Serialize UserAnswers to a flat dict for preset storage."""
        d: dict = {}

        if answers.content_type_override:
            d["content_type_override"] = answers.content_type_override.value
        d["future_languages"] = answers.future_languages
        d["use_case"] = answers.use_case.value
        d["environment"] = answers.constraints.environment.value
        d["latency_budget"] = answers.constraints.latency_budget.value
        d["hardware"] = answers.constraints.hardware.value
        d["ram_gb"] = answers.constraints.ram_gb
        if answers.constraints.vram_gb > 0:
            d["vram_gb"] = answers.constraints.vram_gb
        d["budget"] = answers.constraints.budget.value
        d["privacy"] = answers.constraints.privacy.value
        if answers.embedding_provider:
            d["embedding_provider"] = answers.embedding_provider
        if answers.llm_provider:
            d["llm_provider"] = answers.llm_provider
        if answers.preferred_libraries:
            d["preferred_libraries"] = answers.preferred_libraries
        d["query_type"] = answers.query_type.value
        d["query_complexity"] = answers.query_complexity.value
        d["expected_answer_type"] = answers.expected_answer_type.value
        d["update_frequency"] = answers.update_frequency.value
        d["has_ground_truth"] = answers.has_ground_truth
        d["expected_queries_per_day"] = answers.expected_queries_per_day

        return d
