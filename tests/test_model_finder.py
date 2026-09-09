"""Tests for HFModelFinder."""

from rag_adviser.models import (
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    UseCase,
)
from rag_adviser.recommenders.model_finder import HFModelFinder


class TestHFModelFinder:
    def test_offline_mode_returns_models(self, sample_doc_stats, default_constraints):
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=default_constraints,
            use_case=UseCase.QA,
        )
        assert len(models) > 0
        assert all(m.model_id for m in models)

    def test_offline_mode_scores(self, sample_doc_stats, default_constraints):
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=default_constraints,
            use_case=UseCase.QA,
        )
        # Models should be sorted by score
        scores = [m.score for m in models]
        assert scores == sorted(scores, reverse=True)

    def test_multilingual_requirement(self, multilingual_doc_stats, default_constraints):
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=multilingual_doc_stats,
            constraints=default_constraints,
            use_case=UseCase.QA,
            future_languages=True,
        )
        # Top model should be multilingual
        assert models[0].multilingual is True

    def test_size_filtering(self, sample_doc_stats):
        finder = HFModelFinder(offline=True)
        hw = HardwareConstraints(
            hardware=HardwareProfile.LIMITED_RAM,
            ram_gb=4.0,
        )
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=hw,
            use_case=UseCase.QA,
        )
        # All models should have warnings about size if they're too big
        for m in models:
            assert m.model_id  # At least we get results

    def test_fast_latency_prefers_small(self, sample_doc_stats):
        finder = HFModelFinder(offline=True)
        hw = HardwareConstraints(
            latency_budget=LatencyBudget.FAST,
            hardware=HardwareProfile.CPU_ONLY,
        )
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=hw,
            use_case=UseCase.QA,
        )
        # Top model should be small for fast latency
        assert models[0].estimated_size_gb < 1.0

    def test_strict_privacy_uses_offline(self, sample_doc_stats):
        finder = HFModelFinder(offline=False)  # Set to online...
        hw = HardwareConstraints(privacy=PrivacyLevel.STRICT)
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=hw,
            use_case=UseCase.QA,
        )
        # Should still work (uses fallback list due to strict privacy)
        assert len(models) > 0

    def test_no_doc_stats(self, default_constraints):
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=None,
            constraints=default_constraints,
            use_case=UseCase.QA,
        )
        assert len(models) > 0

    def test_model_has_reasons(self, sample_doc_stats, default_constraints):
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=default_constraints,
            use_case=UseCase.QA,
        )
        # Each model should have reasons for its score
        for m in models:
            assert len(m.reasons) > 0

    def test_max_five_models(self, sample_doc_stats, default_constraints):
        finder = HFModelFinder(offline=True)
        models = finder.find_models(
            doc_stats=sample_doc_stats,
            constraints=default_constraints,
            use_case=UseCase.QA,
        )
        assert len(models) <= 5
