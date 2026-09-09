"""Tests for preset manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_adviser.models import (
    BudgetTier,
    DeploymentTarget,
    PrivacyLevel,
    UseCase,
    UserAnswers,
)
from rag_adviser.presets.manager import PresetManager, PresetProfile


class TestPresetManager:
    """Tests for loading and saving presets."""

    def test_list_builtin_presets(self):
        mgr = PresetManager()
        presets = mgr.list_presets()
        names = [p.name for p in presets]
        assert "legal-discovery" in names
        assert "customer-support" in names
        assert "code-assistant" in names
        assert "internal-wiki" in names
        assert "research-papers" in names
        assert "air-gapped" in names

    def test_load_legal_discovery(self):
        mgr = PresetManager()
        preset = mgr.load_preset("legal-discovery")
        assert preset.name == "legal-discovery"
        assert preset.source == "builtin"
        assert preset.answers["use_case"] == "legal_analysis"
        assert preset.answers["privacy"] == "strict"

    def test_apply_preset(self):
        mgr = PresetManager()
        preset = mgr.load_preset("legal-discovery")
        answers = mgr.apply_preset(preset)

        assert isinstance(answers, UserAnswers)
        assert answers.use_case == UseCase.LEGAL
        assert answers.constraints.privacy == PrivacyLevel.STRICT
        assert answers.constraints.environment == DeploymentTarget.ON_PREM

    def test_apply_customer_support(self):
        mgr = PresetManager()
        preset = mgr.load_preset("customer-support")
        answers = mgr.apply_preset(preset)

        assert answers.constraints.budget == BudgetTier.PAID_API
        assert answers.constraints.environment == DeploymentTarget.CLOUD

    def test_load_nonexistent_raises(self):
        mgr = PresetManager()
        with pytest.raises(ValueError, match="not found"):
            mgr.load_preset("nonexistent-preset")

    def test_save_and_load_custom(self, tmp_path: Path, monkeypatch):
        # Override user dir
        monkeypatch.setattr(
            "rag_adviser.presets.manager._USER_DIR", tmp_path
        )

        mgr = PresetManager()
        answers = UserAnswers()
        answers.use_case = UseCase.SUMMARIZATION
        answers.constraints.budget = BudgetTier.SELF_HOSTED

        path = mgr.save_preset("test-preset", answers, "A test preset")
        assert path.exists()

        # Load it back
        loaded = mgr.load_preset("test-preset")
        assert loaded.name == "test-preset"
        assert loaded.description == "A test preset"

        applied = mgr.apply_preset(loaded)
        assert applied.use_case == UseCase.SUMMARIZATION
        assert applied.constraints.budget == BudgetTier.SELF_HOSTED

    def test_save_json_format(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "rag_adviser.presets.manager._USER_DIR", tmp_path
        )
        mgr = PresetManager()
        path = mgr.save_preset(
            "json-test", UserAnswers(), "test", fmt="json"
        )
        assert path.suffix == ".json"
        import json

        data = json.loads(path.read_text())
        assert data["name"] == "json-test"

    def test_answers_to_dict_roundtrip(self):
        mgr = PresetManager()
        answers = UserAnswers()
        answers.use_case = UseCase.CODE
        answers.constraints.privacy = PrivacyLevel.AIR_GAPPED

        d = mgr._answers_to_dict(answers)
        profile = PresetProfile(name="test", answers=d)
        restored = mgr.apply_preset(profile)

        assert restored.use_case == answers.use_case
        assert restored.constraints.privacy == answers.constraints.privacy

    def test_all_builtin_presets_load(self):
        """Ensure every builtin preset can be loaded and applied without error."""
        mgr = PresetManager()
        for preset in mgr.list_presets():
            if preset.source == "builtin":
                answers = mgr.apply_preset(preset)
                assert isinstance(answers, UserAnswers)
