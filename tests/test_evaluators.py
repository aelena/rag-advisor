"""Tests for the evaluation subsystem — metrics, ground truth loading, chunking strategies."""

import json
from pathlib import Path

import pytest

from rag_adviser.evaluators.chunking_strategies import (
    Chunk,
    chunk_hierarchical,
    chunk_per_file_adaptive,
    chunk_recursive,
    load_documents,
)
from rag_adviser.evaluators.ground_truth_loader import (
    GroundTruthError,
    GroundTruthLoader,
)
from rag_adviser.evaluators.metrics import (
    QueryResult,
    compute_metrics,
    evaluate_single_query,
)

# ── Metrics Tests ─────────────────────────────────────────────────────────


class TestMetrics:
    """Test retrieval evaluation metrics."""

    def test_perfect_retrieval(self) -> None:
        """All relevant docs at top should give perfect scores."""
        result = evaluate_single_query(
            retrieved_texts=["relevant text", "other", "other2"],
            retrieved_doc_ids=["doc1.txt", "doc2.txt", "doc3.txt"],
            retrieved_scores=[0.9, 0.8, 0.7],
            relevant_doc_ids=["doc1.txt"],
            relevant_passages=[],
            query="test query",
            k=3,
        )
        assert result.hit is True
        assert result.reciprocal_rank == 1.0
        assert result.precision_at_k > 0

    def test_no_hit(self) -> None:
        """No relevant docs should give zero scores."""
        result = evaluate_single_query(
            retrieved_texts=["irrelevant", "also irrelevant"],
            retrieved_doc_ids=["wrong1.txt", "wrong2.txt"],
            retrieved_scores=[0.5, 0.4],
            relevant_doc_ids=["correct.txt"],
            relevant_passages=[],
            query="test",
            k=2,
        )
        assert result.hit is False
        assert result.reciprocal_rank == 0.0
        assert result.recall_at_k == 0.0

    def test_mrr_second_position(self) -> None:
        """Relevant doc at position 2 should give RR = 0.5."""
        result = evaluate_single_query(
            retrieved_texts=["wrong", "right"],
            retrieved_doc_ids=["wrong.txt", "right.txt"],
            retrieved_scores=[0.8, 0.7],
            relevant_doc_ids=["right.txt"],
            relevant_passages=[],
            query="test",
            k=5,
        )
        assert result.hit is True
        assert result.reciprocal_rank == 0.5

    def test_passage_matching(self) -> None:
        """Passage-based matching should work via substring containment."""
        result = evaluate_single_query(
            retrieved_texts=[
                "The capital of France is Paris, a beautiful city.",
                "Something else entirely.",
            ],
            retrieved_doc_ids=["doc1.txt", "doc2.txt"],
            retrieved_scores=[0.9, 0.5],
            relevant_doc_ids=[],
            relevant_passages=["capital of France is Paris"],
            query="What is the capital of France?",
            k=2,
        )
        assert result.hit is True
        assert result.reciprocal_rank == 1.0

    def test_recall_at_k(self) -> None:
        """Recall should count fraction of relevant docs found."""
        result = evaluate_single_query(
            retrieved_texts=["a", "b", "c"],
            retrieved_doc_ids=["doc1.txt", "doc2.txt", "doc3.txt"],
            retrieved_scores=[0.9, 0.8, 0.7],
            relevant_doc_ids=["doc1.txt", "doc3.txt", "doc5.txt"],
            relevant_passages=[],
            query="test",
            k=3,
        )
        # 2 out of 3 relevant docs found
        assert abs(result.recall_at_k - 2 / 3) < 0.01

    def test_recall_counts_each_document_once(self) -> None:
        """Several chunks from the same relevant document must not push recall past 1."""
        result = evaluate_single_query(
            retrieved_texts=["a1", "a2", "a3", "b", "c"],
            retrieved_doc_ids=["a.txt", "a.txt", "a.txt", "b.txt", "c.txt"],
            retrieved_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            relevant_doc_ids=["a.txt"],
            relevant_passages=[],
            query="test",
            k=5,
        )
        assert result.recall_at_k == 1.0
        assert result.precision_at_k == 0.6  # 3 of 5 chunks relevant
        assert result.hit is True and result.reciprocal_rank == 1.0

    def test_ndcg_perfect(self) -> None:
        """Perfect ordering should give NDCG = 1.0."""
        result = evaluate_single_query(
            retrieved_texts=["a", "b"],
            retrieved_doc_ids=["rel1.txt", "rel2.txt"],
            retrieved_scores=[0.9, 0.8],
            relevant_doc_ids=["rel1.txt", "rel2.txt"],
            relevant_passages=[],
            query="test",
            k=2,
        )
        assert abs(result.ndcg_at_k - 1.0) < 0.01

    def test_compute_metrics_aggregation(self) -> None:
        """Aggregated metrics should average over queries."""
        results = [
            QueryResult(hit=True, reciprocal_rank=1.0, precision_at_k=0.5,
                        recall_at_k=1.0, ndcg_at_k=1.0),
            QueryResult(hit=False, reciprocal_rank=0.0, precision_at_k=0.0,
                        recall_at_k=0.0, ndcg_at_k=0.0),
        ]
        metrics = compute_metrics(results, "test_strat", num_chunks=100)
        assert metrics.hit_rate == 0.5
        assert metrics.mrr == 0.5
        assert metrics.num_queries == 2
        assert metrics.strategy_name == "test_strat"

    def test_compute_metrics_empty(self) -> None:
        """Empty results should return zero metrics."""
        metrics = compute_metrics([], "empty")
        assert metrics.num_queries == 0
        assert metrics.hit_rate == 0.0


# ── Ground Truth Loader Tests ─────────────────────────────────────────


class TestGroundTruthLoader:
    """Test ground truth file loading."""

    def test_load_jsonl(self, tmp_path: Path) -> None:
        """Load a JSONL ground truth file."""
        gt_file = tmp_path / "gt.jsonl"
        gt_file.write_text(
            json.dumps({
                "query": "What is RAG?",
                "relevant_docs": ["rag.txt"],
                "answer": "Retrieval Augmented Generation",
            }) + "\n"
            + json.dumps({"query": "What is a vector DB?", "relevant_docs": ["vectordb.txt"]})
            + "\n"
        )

        loader = GroundTruthLoader()
        gt = loader.load(gt_file)
        assert gt.query_count == 2
        assert gt.entries[0].query == "What is RAG?"
        assert gt.entries[0].relevant_doc_ids == ["rag.txt"]
        assert gt.entries[0].expected_answer == "Retrieval Augmented Generation"
        assert gt.has_doc_ids is True

    def test_load_csv(self, tmp_path: Path) -> None:
        """Load a CSV ground truth file."""
        gt_file = tmp_path / "gt.csv"
        gt_file.write_text(
            "query,relevant_docs,answer\n"
            '"What is RAG?","rag.txt;intro.txt","It is RAG"\n'
            '"What is embedding?","embed.txt","Vector representation"\n'
        )

        loader = GroundTruthLoader()
        gt = loader.load(gt_file)
        assert gt.query_count == 2
        assert gt.entries[0].relevant_doc_ids == ["rag.txt", "intro.txt"]

    def test_load_nonexistent_file(self) -> None:
        """Loading a nonexistent file should raise."""
        loader = GroundTruthLoader()
        with pytest.raises(GroundTruthError, match="not found"):
            loader.load(Path("/nonexistent/ground_truth.jsonl"))

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Empty file should raise."""
        gt_file = tmp_path / "empty.jsonl"
        gt_file.write_text("")
        loader = GroundTruthLoader()
        with pytest.raises(GroundTruthError, match="No valid entries"):
            loader.load(gt_file)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON should raise."""
        gt_file = tmp_path / "bad.jsonl"
        gt_file.write_text("not json at all\n")
        loader = GroundTruthLoader()
        with pytest.raises(GroundTruthError, match="Invalid JSON"):
            loader.load(gt_file)

    def test_load_missing_query_field(self, tmp_path: Path) -> None:
        """Entry without query field should raise."""
        gt_file = tmp_path / "no_query.jsonl"
        gt_file.write_text(json.dumps({"answer": "something"}) + "\n")
        loader = GroundTruthLoader()
        with pytest.raises(GroundTruthError, match="query"):
            loader.load(gt_file)

    def test_passage_matching_in_gt(self, tmp_path: Path) -> None:
        """Ground truth with passages should be parseable."""
        gt_file = tmp_path / "passages.jsonl"
        gt_file.write_text(
            json.dumps({
                "query": "What is RAG?",
                "passages": ["RAG is retrieval augmented generation"],
            }) + "\n"
        )
        loader = GroundTruthLoader()
        gt = loader.load(gt_file)
        assert gt.has_passages is True
        assert gt.entries[0].relevant_passages == ["RAG is retrieval augmented generation"]


# ── Chunking Strategies Tests ─────────────────────────────────────────


class TestChunkingStrategies:
    """Test the chunking strategy implementations."""

    def _make_corpus(self, tmp_path: Path) -> Path:
        """Create a small test corpus."""
        doc1 = tmp_path / "intro.txt"
        doc1.write_text(
            "RAG stands for Retrieval Augmented Generation. "
            "It combines retrieval with generation. "
            "This is a powerful technique for building AI systems. "
            "Many companies use RAG in production.\n\n"
            "Vector databases store embeddings efficiently. "
            "Popular options include ChromaDB, Pinecone, and Qdrant. "
            "Each has different trade-offs for scale and performance."
        )

        doc2 = tmp_path / "code_example.py"
        doc2.write_text(
            "def hello_world():\n"
            '    print("Hello, world!")\n\n'
            "class Retriever:\n"
            "    def __init__(self, db):\n"
            "        self.db = db\n\n"
            "    def search(self, query):\n"
            "        return self.db.query(query)\n"
        )

        return tmp_path

    def test_load_documents(self, tmp_path: Path) -> None:
        """Should load text files from directory."""
        corpus = self._make_corpus(tmp_path)
        docs = load_documents(corpus)
        assert len(docs) == 2
        filenames = [d[0] for d in docs]
        assert "intro.txt" in filenames
        assert "code_example.py" in filenames

    def test_load_single_file(self, tmp_path: Path) -> None:
        """Should load a single file."""
        corpus = self._make_corpus(tmp_path)
        docs = load_documents(corpus / "intro.txt")
        assert len(docs) == 1
        assert docs[0][0] == "intro.txt"

    def test_recursive_chunking(self, tmp_path: Path) -> None:
        """Recursive chunking should produce chunks."""
        corpus = self._make_corpus(tmp_path)
        docs = load_documents(corpus)
        result = chunk_recursive(docs, chunk_size=100, chunk_overlap=10)
        assert result.total_chunks > 0
        assert result.strategy_name == "recursive"
        assert all(isinstance(c, Chunk) for c in result.chunks)
        # All chunks should have source file
        assert all(c.source_file for c in result.chunks)

    def test_hierarchical_chunking(self, tmp_path: Path) -> None:
        """Hierarchical chunking should produce parent-linked child chunks."""
        corpus = self._make_corpus(tmp_path)
        docs = load_documents(corpus)
        result = chunk_hierarchical(
            docs, parent_size=200, child_size=50, child_overlap=10
        )
        assert result.total_chunks > 0
        assert result.strategy_name == "hierarchical"
        # Child chunks should have parent_text in metadata
        for chunk in result.chunks:
            assert "parent_text" in chunk.metadata

    def test_adaptive_chunking(self, tmp_path: Path) -> None:
        """Adaptive chunking should detect file types."""
        corpus = self._make_corpus(tmp_path)
        docs = load_documents(corpus)
        result = chunk_per_file_adaptive(docs, chunk_size=100, chunk_overlap=10)
        assert result.total_chunks > 0
        assert result.strategy_name == "adaptive"
        # Should have detected code type for .py file
        py_chunks = [c for c in result.chunks if c.source_file == "code_example.py"]
        assert py_chunks
        assert py_chunks[0].metadata.get("detected_type") == "code"

    def test_recursive_small_doc(self) -> None:
        """Document smaller than chunk_size should produce single chunk."""
        docs = [("small.txt", "This is a small document.")]
        result = chunk_recursive(docs, chunk_size=1000, chunk_overlap=0)
        assert result.total_chunks == 1
        assert result.chunks[0].text == "This is a small document."

    def test_chunks_have_source(self, tmp_path: Path) -> None:
        """Every chunk should track its source file."""
        corpus = self._make_corpus(tmp_path)
        docs = load_documents(corpus)
        for strategy_fn in [chunk_recursive, chunk_hierarchical, chunk_per_file_adaptive]:
            if strategy_fn == chunk_hierarchical:
                result = strategy_fn(docs, parent_size=200, child_size=50)
            else:
                result = strategy_fn(docs, chunk_size=100, chunk_overlap=10)
            for chunk in result.chunks:
                assert chunk.source_file, f"Missing source in {strategy_fn.__name__}"
