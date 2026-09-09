"""Tests for the dependency-free in-memory vector store."""

from __future__ import annotations

import pytest

from rag_adviser.evaluators.chunking_strategies import Chunk
from rag_adviser.evaluators.vector_store import InMemoryVectorStore, create_vector_store

pytest.importorskip("numpy")


def _chunks() -> list[Chunk]:
    return [
        Chunk(text="alpha", source_file="a.txt", chunk_index=0, strategy="recursive"),
        Chunk(text="beta", source_file="b.txt", chunk_index=0, strategy="recursive"),
        Chunk(text="gamma", source_file="c.txt", chunk_index=0, strategy="recursive"),
    ]


class TestInMemoryVectorStore:
    def test_factory_returns_memory_backend(self) -> None:
        assert isinstance(create_vector_store("memory"), InMemoryVectorStore)

    def test_search_orders_by_cosine(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection("t", 3)
        # Unnormalised on purpose: cosine must ignore magnitude.
        store.add_chunks(_chunks(), [[10, 0, 0], [0, 1, 0], [1, 1, 0]])
        assert store.count() == 3

        results = store.search([1, 0, 0], top_k=2)
        assert [r.source_file for r in results] == ["a.txt", "c.txt"]
        assert results[0].score == pytest.approx(1.0)
        assert results[0].metadata["source"] == "a.txt"

    def test_top_k_capped_and_empty(self) -> None:
        store = InMemoryVectorStore()
        store.create_collection("t", 2)
        assert store.search([1, 0], top_k=5) == []
        store.add_chunks(_chunks()[:2], [[1, 0], [0, 1]])
        assert len(store.search([1, 0], top_k=10)) == 2
        store.delete_collection()
        assert store.count() == 0
