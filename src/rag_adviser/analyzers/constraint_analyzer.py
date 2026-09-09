"""Analyze and validate hardware, deployment, and privacy constraints."""

from __future__ import annotations

from rag_adviser.models import (
    BudgetTier,
    DeploymentTarget,
    HardwareConstraints,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
)


class ConstraintAnalyzer:
    """Evaluate constraints and generate warnings about conflicts or limitations."""

    # Max model size estimates by hardware profile. Sizes in defaults.yaml are
    # fp32 checkpoint sizes; fp16/int8 loading roughly halves/quarters them,
    # so the CPU cap is deliberately generous enough to admit the ~2.2GB
    # multilingual large models on a 16GB machine.
    _MODEL_SIZE_LIMITS = {
        HardwareProfile.CPU_ONLY: 3.0,      # GB
        HardwareProfile.GPU_AVAILABLE: 10.0,  # GB
        HardwareProfile.LIMITED_RAM: 0.5,     # GB
    }

    def get_max_model_size(self, constraints: HardwareConstraints) -> float:
        """Derive maximum embedding model size in GB from constraints."""
        hw_limit = self._MODEL_SIZE_LIMITS.get(
            constraints.hardware, self._MODEL_SIZE_LIMITS[HardwareProfile.CPU_ONLY]
        )

        # Use 30% of available RAM as a cap
        ram_limit = constraints.ram_gb * 0.3

        # If GPU available, also consider VRAM (70% of VRAM)
        if constraints.hardware == HardwareProfile.GPU_AVAILABLE and constraints.vram_gb > 0:
            vram_limit = constraints.vram_gb * 0.7
            return min(hw_limit, max(ram_limit, vram_limit))

        return min(hw_limit, ram_limit)

    def get_warnings(self, constraints: HardwareConstraints) -> list[str]:
        """Generate warnings about constraint conflicts or limitations."""
        warnings: list[str] = []

        # Privacy + budget conflicts
        if (
            constraints.privacy in (PrivacyLevel.STRICT, PrivacyLevel.AIR_GAPPED)
            and constraints.budget == BudgetTier.PAID_API
        ):
            warnings.append(
                "Privacy is set to strict/air-gapped but budget allows paid APIs. "
                "API-based models will be excluded due to privacy constraints."
            )

        if constraints.privacy == PrivacyLevel.AIR_GAPPED:
            warnings.append(
                "Air-gapped mode: HuggingFace API will not be queried. "
                "Recommendations will use curated offline model list only."
            )

        # Hardware + latency conflicts
        if (
            constraints.latency_budget == LatencyBudget.FAST
            and constraints.hardware == HardwareProfile.CPU_ONLY
        ):
            warnings.append(
                "Sub-500ms latency on CPU-only requires small models (<500MB). "
                "This may limit embedding quality."
            )

        if constraints.hardware == HardwareProfile.LIMITED_RAM:
            warnings.append(
                f"Limited RAM ({constraints.ram_gb}GB) restricts model options. "
                "Only the smallest embedding models will be recommended."
            )

        # Edge deployment
        if constraints.environment == DeploymentTarget.EDGE:
            warnings.append(
                "Edge deployment: only lightweight, quantized models will be recommended."
            )
            if constraints.hardware != HardwareProfile.LIMITED_RAM:
                warnings.append(
                    "Consider setting --hardware limited_ram for edge to get "
                    "appropriately sized recommendations."
                )

        # Local + free = open source only
        if (
            constraints.privacy in (PrivacyLevel.STRICT, PrivacyLevel.AIR_GAPPED)
            and constraints.budget == BudgetTier.FREE
        ):
            warnings.append(
                "Local-only + free tier: limited to open-source models. "
                "This is still a strong option with models like BGE and E5."
            )

        return warnings

    def is_api_allowed(self, constraints: HardwareConstraints) -> bool:
        """Check if external API calls are permitted by privacy settings."""
        return constraints.privacy in (PrivacyLevel.NONE, PrivacyLevel.MODERATE)
