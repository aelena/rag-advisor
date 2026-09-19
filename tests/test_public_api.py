"""Public Python-API surface tests.

Guards the promises the README makes about ``from rag_adviser import …``.
When a name is added to or removed from ``__all__``, either:

1. Update the README's Python API section to match, or
2. Fix the ``__all__`` list.

The tests are cheap and deliberate — they exist so the API surface can
be a first-class contract rather than an accidental one.
"""

from __future__ import annotations

import rag_adviser


class TestPublicApiSurface:
    def test_version_string_present(self) -> None:
        assert isinstance(rag_adviser.__version__, str)
        assert rag_adviser.__version__.count(".") >= 2

    def test_all_named_symbols_resolve(self) -> None:
        # Every name in __all__ must actually be importable at package
        # top level. A missing re-export is a broken promise.
        for name in rag_adviser.__all__:
            assert hasattr(rag_adviser, name), (
                f"__all__ names {name!r} but it is not attached to the package"
            )

    def test_load_bearing_names_are_public(self) -> None:
        # Names the README teaches users to import. If any of these drop
        # off __all__ it should fail here — renames need a doc update.
        expected = {
            "RAGAdviser",
            "UserAnswers",
            "HardwareConstraints",
            "SizingProfile",
            "DocumentStats",
            "Recommendations",
            "UseCase",
            "LatencyBudget",
            "HardwareProfile",
            "PrivacyLevel",
            "BudgetTier",
            "QueryType",
            "QueryComplexity",
            "AnswerType",
            "CitationGranularity",
            "ErrorCost",
            "UpdateFrequency",
            "ContentType",
            "DeploymentTarget",
            "ReportFormat",
            "RecommendedApproach",
            "RagAdvisorError",
        }
        assert expected.issubset(set(rag_adviser.__all__))

    def test_end_to_end_via_public_api_only(self, tmp_path) -> None:
        # A downstream caller should be able to run the recommender
        # using nothing but top-level imports. If this test needs a
        # sub-module import to work, the API is incomplete.
        from rag_adviser import (
            HardwareConstraints,
            PrivacyLevel,
            RAGAdviser,
            Recommendations,
            ReportFormat,
            UseCase,
            UserAnswers,
        )

        corpus = tmp_path / "docs"
        corpus.mkdir()
        (corpus / "a.txt").write_text(
            "Retrieval-augmented generation grounds answers in evidence. " * 20,
            encoding="utf-8",
        )
        recs = RAGAdviser().run(
            UserAnswers(
                document_path=corpus,
                use_case=UseCase.QA,
                constraints=HardwareConstraints(privacy=PrivacyLevel.STRICT),
            ),
            output_dir=tmp_path / "out",
            formats=[ReportFormat.YAML],
        )
        assert isinstance(recs, Recommendations)
        assert recs.embedding_models
        assert recs.chunking is not None
        assert recs.retrieval is not None
