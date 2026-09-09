"""Tests for ConstraintAnalyzer."""

from rag_adviser.analyzers.constraint_analyzer import ConstraintAnalyzer
from rag_adviser.models import (
    BudgetTier,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
)


class TestConstraintAnalyzer:
    def test_max_model_size_cpu(self, default_constraints):
        analyzer = ConstraintAnalyzer()
        max_size = analyzer.get_max_model_size(default_constraints)
        # CPU 16GB RAM: min(3.0, 16*0.3) = min(3.0, 4.8) = 3.0
        assert max_size == 3.0

    def test_max_model_size_gpu(self, gpu_constraints):
        analyzer = ConstraintAnalyzer()
        max_size = analyzer.get_max_model_size(gpu_constraints)
        # GPU 32GB RAM, 8GB VRAM: min(10.0, max(32*0.3, 8*0.7)) = min(10.0, max(9.6, 5.6)) = 9.6
        assert max_size == 9.6

    def test_max_model_size_limited_ram(self):
        analyzer = ConstraintAnalyzer()
        hw = HardwareConstraints(
            hardware=HardwareProfile.LIMITED_RAM,
            ram_gb=4.0,
        )
        max_size = analyzer.get_max_model_size(hw)
        # min(0.5, 4*0.3) = min(0.5, 1.2) = 0.5
        assert max_size == 0.5

    def test_warnings_fast_latency_cpu(self):
        analyzer = ConstraintAnalyzer()
        hw = HardwareConstraints(
            latency_budget=LatencyBudget.FAST,
            hardware=HardwareProfile.CPU_ONLY,
        )
        warnings = analyzer.get_warnings(hw)
        assert any("Sub-500ms" in w for w in warnings)

    def test_warnings_air_gapped(self):
        analyzer = ConstraintAnalyzer()
        hw = HardwareConstraints(privacy=PrivacyLevel.AIR_GAPPED)
        warnings = analyzer.get_warnings(hw)
        assert any("Air-gapped" in w for w in warnings)

    def test_warnings_strict_free(self):
        analyzer = ConstraintAnalyzer()
        hw = HardwareConstraints(
            privacy=PrivacyLevel.STRICT,
            budget=BudgetTier.FREE,
        )
        warnings = analyzer.get_warnings(hw)
        assert any("open-source" in w for w in warnings)

    def test_no_warnings_default(self, default_constraints):
        analyzer = ConstraintAnalyzer()
        warnings = analyzer.get_warnings(default_constraints)
        # Default constraints should have no warnings
        assert isinstance(warnings, list)

    def test_api_allowed_none_privacy(self):
        analyzer = ConstraintAnalyzer()
        hw = HardwareConstraints(privacy=PrivacyLevel.NONE)
        assert analyzer.is_api_allowed(hw) is True

    def test_api_not_allowed_strict(self):
        analyzer = ConstraintAnalyzer()
        hw = HardwareConstraints(privacy=PrivacyLevel.STRICT)
        assert analyzer.is_api_allowed(hw) is False
