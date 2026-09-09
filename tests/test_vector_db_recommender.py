"""Tests for VectorDBRecommender."""

from rag_adviser.models import (
    DeploymentTarget,
    HardwareConstraints,
    PrivacyLevel,
    UpdateFrequency,
)
from rag_adviser.recommenders.vector_db_recommender import VectorDBRecommender


class TestVectorDBRecommender:
    def test_small_local_corpus(self, default_constraints):
        rec = VectorDBRecommender()
        result = rec.recommend(
            doc_count=100,
            constraints=default_constraints,
            update_frequency=UpdateFrequency.NEVER,
        )
        # Small + local should favor embedded DBs
        assert result.category in ("embedded", "in-memory")
        assert result.provider  # Should have a provider name

    def test_strict_privacy_excludes_managed(self):
        rec = VectorDBRecommender()
        hw = HardwareConstraints(privacy=PrivacyLevel.STRICT)
        result = rec.recommend(
            doc_count=100,
            constraints=hw,
            update_frequency=UpdateFrequency.NEVER,
        )
        assert result.category != "managed"

    def test_cloud_deployment(self):
        rec = VectorDBRecommender()
        hw = HardwareConstraints(
            environment=DeploymentTarget.CLOUD,
            privacy=PrivacyLevel.NONE,
        )
        result = rec.recommend(
            doc_count=500_000,
            constraints=hw,
            update_frequency=UpdateFrequency.DAILY,
        )
        assert result.provider  # Should recommend something

    def test_code_snippet_generated(self, default_constraints):
        rec = VectorDBRecommender()
        result = rec.recommend(
            doc_count=100,
            constraints=default_constraints,
        )
        assert result.code_snippet  # Should have a code snippet

    def test_library_field_populated(self, default_constraints):
        rec = VectorDBRecommender()
        result = rec.recommend(
            doc_count=100,
            constraints=default_constraints,
        )
        assert result.library  # Should have a pip package name

    def test_has_reason(self, default_constraints):
        rec = VectorDBRecommender()
        result = rec.recommend(
            doc_count=100,
            constraints=default_constraints,
        )
        assert result.reason  # Should explain why this DB was chosen
