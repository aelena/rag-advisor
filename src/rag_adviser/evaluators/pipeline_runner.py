"""Evaluation pipeline runner — orchestrates chunk -> embed -> index -> query -> metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from rag_adviser.evaluators.chunking_strategies import (
    ChunkedCorpus,
    chunk_hierarchical,
    chunk_per_file_adaptive,
    chunk_recursive,
    chunk_semantic,
    load_documents,
)
from rag_adviser.evaluators.ground_truth_loader import GroundTruthLoader, GroundTruthSet
from rag_adviser.evaluators.metrics import (
    EvalMetrics,
    compute_metrics,
    evaluate_single_query,
)
from rag_adviser.evaluators.vector_store import BaseVectorStore, create_vector_store
from rag_adviser.models import RagAdvisorError

logger = logging.getLogger(__name__)


class EvalPipelineError(RagAdvisorError):
    """Evaluation pipeline failed."""


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    corpus_path: Path = field(default_factory=lambda: Path("."))
    ground_truth_path: Path = field(default_factory=lambda: Path("ground_truth.jsonl"))
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    trust_remote_code: bool = False  # for models that ship custom code (nomic, gte, jina)
    strategies: list[str] = field(
        default_factory=lambda: ["recursive", "semantic", "hierarchical", "adaptive"]
    )
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    vector_backend: str = "chroma"
    db_connection: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("./eval_results"))


@dataclass
class EvalReport:
    """Complete evaluation results across all strategies."""

    config: EvalConfig = field(default_factory=EvalConfig)
    ground_truth: GroundTruthSet | None = None
    strategy_results: list[EvalMetrics] = field(default_factory=list)
    best_strategy: str = ""
    embedding_model: str = ""


class EvalPipelineRunner:
    """Orchestrate the full evaluation: chunk -> embed -> index -> query -> compare."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._model = None
        self._dimension = 0

    def run(self, config: EvalConfig) -> EvalReport:
        """Execute the full evaluation pipeline."""
        report = EvalReport(config=config, embedding_model=config.embedding_model)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            # Step 1: Load ground truth
            task = progress.add_task("Loading ground truth...", total=1)
            loader = GroundTruthLoader()
            report.ground_truth = loader.load(config.ground_truth_path)
            progress.update(task, completed=1)
            self.console.print(
                f"  Loaded {report.ground_truth.query_count} evaluation queries"
            )

            # Step 2: Load documents
            task = progress.add_task("Loading documents...", total=1)
            documents = load_documents(config.corpus_path)
            progress.update(task, completed=1)
            self.console.print(f"  Loaded {len(documents)} documents")

            if not documents:
                raise EvalPipelineError(
                    f"No documents found in {config.corpus_path}"
                )

            # Step 3: Load embedding model
            task = progress.add_task("Loading embedding model...", total=1)
            self._load_embedding_model(config.embedding_model, config.trust_remote_code)
            progress.update(task, completed=1)
            self.console.print(
                f"  Model: {config.embedding_model} (dim={self._dimension})"
            )

            # Step 4: Apply chunking strategies and evaluate each
            total_strategies = len(config.strategies)
            for strat_idx, strategy_name in enumerate(config.strategies):
                self.console.print(
                    f"\n[bold cyan]Strategy {strat_idx + 1}/{total_strategies}: "
                    f"{strategy_name}[/]"
                )

                # 4a: Chunk
                task = progress.add_task(
                    f"  Chunking ({strategy_name})...", total=1
                )
                chunked = self._apply_strategy(
                    strategy_name, documents, config
                )
                progress.update(task, completed=1)
                self.console.print(
                    f"  Produced {chunked.total_chunks} chunks"
                )

                # 4b: Embed all chunks
                task = progress.add_task(
                    f"  Embedding {chunked.total_chunks} chunks...",
                    total=chunked.total_chunks,
                )
                chunk_embeddings = self._embed_chunks(chunked, progress, task)

                # 4c: Index
                task = progress.add_task(
                    f"  Indexing in {config.vector_backend}...", total=1
                )
                store = create_vector_store(
                    backend=config.vector_backend,
                    connection_string=config.db_connection,
                )
                collection_name = f"eval_{strategy_name}"
                store.create_collection(collection_name, self._dimension)
                store.add_chunks(chunked.chunks, chunk_embeddings)
                progress.update(task, completed=1)
                self.console.print(
                    f"  Indexed {store.count()} chunks in {config.vector_backend}"
                )

                # 4d: Query and evaluate
                task = progress.add_task(
                    f"  Evaluating {report.ground_truth.query_count} queries...",
                    total=report.ground_truth.query_count,
                )
                metrics = self._evaluate_strategy(
                    store=store,
                    ground_truth=report.ground_truth,
                    strategy_name=strategy_name,
                    top_k=config.top_k,
                    num_chunks=chunked.total_chunks,
                    progress=progress,
                    task_id=task,
                )
                report.strategy_results.append(metrics)

                # Clean up
                store.delete_collection()

                self.console.print(
                    f"  [green]Hit Rate: {metrics.hit_rate:.1%} | "
                    f"MRR: {metrics.mrr:.3f} | "
                    f"Recall@{config.top_k}: {metrics.mean_recall_at_k:.1%}[/]"
                )

        # Determine best strategy
        if report.strategy_results:
            best = max(report.strategy_results, key=lambda m: m.mrr)
            report.best_strategy = best.strategy_name

        return report

    def _load_embedding_model(self, model_id: str, trust_remote_code: bool = False) -> None:
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EvalPipelineError(
                "sentence-transformers is required for evaluation. "
                "Install: pip install ragadvisor[eval]"
            ) from e

        self._model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code)
        # Get dimension from a test encode
        test_emb = self._model.encode(["test"], show_progress_bar=False)
        self._dimension = len(test_emb[0])

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]

    def _embed_chunks(
        self,
        chunked: ChunkedCorpus,
        progress: Progress,
        task_id: int,
    ) -> list[list[float]]:
        """Embed all chunks with progress tracking."""
        texts = [c.text for c in chunked.chunks]
        all_embeddings: list[list[float]] = []

        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embs = self._embed_texts(batch)
            all_embeddings.extend(batch_embs)
            progress.update(task_id, completed=i + len(batch))

        return all_embeddings

    def _apply_strategy(
        self,
        strategy_name: str,
        documents: list[tuple[str, str]],
        config: EvalConfig,
    ) -> ChunkedCorpus:
        """Apply a named chunking strategy to the documents."""
        if strategy_name == "recursive":
            return chunk_recursive(
                documents,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        elif strategy_name == "semantic":
            return chunk_semantic(
                documents,
                embedding_fn=self._embed_texts,
                threshold=0.5,
                min_chunk_size=100,
                max_chunk_size=config.chunk_size * 3,
            )
        elif strategy_name == "hierarchical":
            return chunk_hierarchical(
                documents,
                parent_size=config.chunk_size * 2,
                child_size=config.chunk_size // 2,
                child_overlap=config.chunk_overlap // 2,
            )
        elif strategy_name == "adaptive":
            return chunk_per_file_adaptive(
                documents,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        else:
            raise EvalPipelineError(f"Unknown chunking strategy: {strategy_name}")

    def _evaluate_strategy(
        self,
        store: BaseVectorStore,
        ground_truth: GroundTruthSet,
        strategy_name: str,
        top_k: int,
        num_chunks: int,
        progress: Progress,
        task_id: int,
    ) -> EvalMetrics:
        """Run all ground truth queries against the indexed store."""
        query_results = []

        for i, entry in enumerate(ground_truth.entries):
            # Embed the query
            query_emb = self._embed_texts([entry.query])[0]

            # Search
            search_results = store.search(query_emb, top_k=top_k)

            # Evaluate
            result = evaluate_single_query(
                retrieved_texts=[r.text for r in search_results],
                retrieved_doc_ids=[r.source_file for r in search_results],
                retrieved_scores=[r.score for r in search_results],
                relevant_doc_ids=entry.relevant_doc_ids,
                relevant_passages=entry.relevant_passages,
                query=entry.query,
                k=top_k,
            )
            query_results.append(result)
            progress.update(task_id, completed=i + 1)

        return compute_metrics(query_results, strategy_name, num_chunks)


def display_comparison_table(
    report: EvalReport,
    console: Console | None = None,
) -> None:
    """Display a Rich comparison table of all strategy results."""
    console = console or Console()
    k = report.config.top_k

    table = Table(
        title=f"Evaluation Results — {report.embedding_model}",
        show_header=True,
        title_style="bold cyan",
    )
    table.add_column("Strategy", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column(f"Hit Rate@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"Precision@{k}", justify="right")
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column(f"NDCG@{k}", justify="right")

    for metrics in report.strategy_results:
        is_best = metrics.strategy_name == report.best_strategy
        style = "bold green" if is_best else ""
        marker = " *" if is_best else ""

        table.add_row(
            f"{metrics.strategy_name}{marker}",
            str(metrics.num_chunks),
            f"{metrics.hit_rate:.1%}",
            f"{metrics.mrr:.3f}",
            f"{metrics.mean_precision_at_k:.1%}",
            f"{metrics.mean_recall_at_k:.1%}",
            f"{metrics.mean_ndcg_at_k:.3f}",
            style=style,
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold green]* Best strategy:[/] {report.best_strategy} (by MRR)")
    console.print(
        f"[dim]Backend: {report.config.vector_backend} | "
        f"Queries: {report.ground_truth.query_count if report.ground_truth else 0} | "
        f"Chunk size: {report.config.chunk_size}[/]"
    )


def generate_eval_report_markdown(report: EvalReport, output_dir: Path) -> Path:
    """Generate a Markdown evaluation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k = report.config.top_k
    lines: list[str] = []

    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Embedding Model:** `{report.embedding_model}`  ")
    lines.append(f"**Vector Backend:** {report.config.vector_backend}  ")
    lines.append(f"**Chunk Size:** {report.config.chunk_size}  ")
    lines.append(f"**Top-K:** {k}  ")
    if report.ground_truth:
        lines.append(f"**Evaluation Queries:** {report.ground_truth.query_count}  ")
        lines.append(f"**Ground Truth Source:** `{report.ground_truth.source_path}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Comparison table
    lines.append("## Strategy Comparison")
    lines.append("")
    lines.append(
        f"| Strategy | Chunks | Hit Rate@{k} | MRR | Precision@{k} | Recall@{k} | NDCG@{k} |"
    )
    lines.append("|----------|--------|------------|-----|-------------|-----------|---------|")

    for m in report.strategy_results:
        best = " **" if m.strategy_name == report.best_strategy else ""
        end = "**" if best else ""
        lines.append(
            f"| {best}{m.strategy_name}{end} | {m.num_chunks} | "
            f"{m.hit_rate:.1%} | {m.mrr:.3f} | {m.mean_precision_at_k:.1%} | "
            f"{m.mean_recall_at_k:.1%} | {m.mean_ndcg_at_k:.3f} |"
        )

    lines.append("")
    lines.append(f"**Best strategy:** {report.best_strategy} (by MRR)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-query breakdown for best strategy
    if report.strategy_results:
        best_metrics = next(
            (m for m in report.strategy_results if m.strategy_name == report.best_strategy),
            report.strategy_results[0],
        )

        lines.append(f"## Per-Query Breakdown ({best_metrics.strategy_name})")
        lines.append("")
        lines.append("| Query | Hit | RR | Top Result |")
        lines.append("|-------|-----|----|------------|")

        for qr in best_metrics.per_query[:50]:  # Limit to 50 queries in report
            hit_mark = "Y" if qr.hit else "-"
            top_text = qr.retrieved_texts[0][:80] + "..." if qr.retrieved_texts else "-"
            query_short = qr.query[:60] + "..." if len(qr.query) > 60 else qr.query
            lines.append(
                f"| {query_short} | {hit_mark} | {qr.reciprocal_rank:.2f} | {top_text} |"
            )

        lines.append("")

    out_path = output_dir / "eval_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
