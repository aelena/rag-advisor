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
from rag_adviser.evaluators.retrieval_modes import (
    BM25Index,
    CrossEncoderReranker,
    mode_label,
    rrf_fuse,
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
    # Evaluate several models in one run (overrides embedding_model when set).
    # Models that fail to load are skipped. trust_remote_code_models lists the
    # ids that need custom code; max_successful_models stops after N models
    # evaluated successfully (fallback-chain behaviour when set to 1).
    embedding_models: list[str] = field(default_factory=list)
    trust_remote_code_models: list[str] = field(default_factory=list)
    max_successful_models: int | None = None
    strategies: list[str] = field(
        default_factory=lambda: ["recursive", "semantic", "hierarchical", "adaptive"]
    )
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    # Chunk-size sweep: evaluate every strategy at each size (characters).
    # Empty -> [chunk_size]. When sweeping, overlap = size * overlap_ratio
    # unless overlap_ratio is None (then chunk_overlap is used for every size).
    chunk_sizes: list[int] = field(default_factory=list)
    overlap_ratio: float | None = 0.1
    vector_backend: str = "chroma"
    db_connection: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("./eval_results"))
    # Retrieval mode. hybrid = BM25 + dense fused with RRF; rerank_model = a
    # sentence-transformers CrossEncoder applied to the fetch_k candidates.
    hybrid: bool = False
    rerank_model: str | None = None
    rerank_trust_remote_code: bool = False
    fetch_k: int = 20
    # Also evaluate plain dense retrieval on the same index for comparison
    # (only meaningful when hybrid or rerank_model is set).
    dense_baseline: bool = False


@dataclass
class EvalReport:
    """Complete evaluation results across all strategies."""

    config: EvalConfig = field(default_factory=EvalConfig)
    ground_truth: GroundTruthSet | None = None
    strategy_results: list[EvalMetrics] = field(default_factory=list)
    best_strategy: str = ""
    best_model: str = ""
    best_chunk_size: int = 0
    embedding_model: str = ""
    embedding_models: list[str] = field(default_factory=list)
    model_errors: dict[str, str] = field(default_factory=dict)


class EvalPipelineRunner:
    """Orchestrate the full evaluation: chunk -> embed -> index -> query -> compare."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._model = None
        self._dimension = 0

    def run(self, config: EvalConfig) -> EvalReport:
        """Execute the full evaluation pipeline for every configured embedding model.

        Models that fail to load are recorded in ``report.model_errors`` and
        skipped; the run only fails when no model could be evaluated.
        """
        models = list(config.embedding_models) or [config.embedding_model]
        multi = len(models) > 1
        sizes = list(config.chunk_sizes) or [config.chunk_size]
        sweep = len(sizes) > 1
        report = EvalReport(config=config, embedding_model=models[0], embedding_models=models)

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

            # Step 3: Optional reranker (shared by every model)
            reranker: CrossEncoderReranker | None = None
            if config.rerank_model:
                task = progress.add_task("Loading reranker...", total=1)
                reranker = CrossEncoderReranker(
                    config.rerank_model, trust_remote_code=config.rerank_trust_remote_code
                )
                progress.update(task, completed=1)
                self.console.print(f"  Reranker: {config.rerank_model}")
            mode = mode_label(config.hybrid, reranker is not None)
            self.console.print(f"  Retrieval mode: {mode}")

            # Step 4: For each embedding model, chunk -> embed -> index -> evaluate
            trc_models = set(config.trust_remote_code_models)
            successes = 0
            for model_idx, model_id in enumerate(models):
                if config.max_successful_models and successes >= config.max_successful_models:
                    break
                if multi:
                    self.console.print(
                        f"\n[bold magenta]Model {model_idx + 1}/{len(models)}: {model_id}[/]"
                    )
                task = progress.add_task(f"Loading {model_id}...", total=1)
                try:
                    self._load_embedding_model(
                        model_id, config.trust_remote_code or model_id in trc_models
                    )
                except Exception as e:  # skip this model, keep evaluating the others
                    progress.remove_task(task)
                    report.model_errors[model_id] = f"{type(e).__name__}: {e}"
                    self.console.print(
                        f"  [yellow]Skipping {model_id}: {report.model_errors[model_id][:160]}[/]"
                    )
                    continue
                progress.update(task, completed=1)
                self.console.print(f"  Model: {model_id} (dim={self._dimension})")

                combos = [(s, size) for s in config.strategies for size in sizes]
                for combo_idx, (strategy_name, size) in enumerate(combos):
                    overlap = self._overlap_for(size, config)
                    label = self._label(strategy_name, model_id, multi, size, sweep)
                    self.console.print(
                        f"\n[bold cyan]Run {combo_idx + 1}/{len(combos)}: {label}[/]"
                    )

                    # 4a: Chunk
                    task = progress.add_task(f"  Chunking ({strategy_name} @ {size})...", total=1)
                    chunked = self._apply_strategy(
                        strategy_name, documents, config, chunk_size=size, chunk_overlap=overlap
                    )
                    progress.update(task, completed=1)
                    self.console.print(f"  Produced {chunked.total_chunks} chunks")

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
                    store.create_collection(f"eval_{strategy_name}", self._dimension)
                    store.add_chunks(chunked.chunks, chunk_embeddings)
                    progress.update(task, completed=1)
                    self.console.print(
                        f"  Indexed {store.count()} chunks in {config.vector_backend}"
                    )

                    # 4d: Evaluate in the configured mode, then (optionally) plain
                    # dense retrieval on the same index as a baseline.
                    bm25 = BM25Index(chunked.chunks) if config.hybrid else None
                    runs: list[tuple[str, BM25Index | None, CrossEncoderReranker | None]] = [
                        (label, bm25, reranker)
                    ]
                    if config.dense_baseline and (bm25 is not None or reranker is not None):
                        runs.append((f"{label} (dense baseline)", None, None))

                    for run_label, run_bm25, run_reranker in runs:
                        task = progress.add_task(
                            f"  Evaluating {report.ground_truth.query_count} queries "
                            f"[{mode_label(run_bm25 is not None, run_reranker is not None)}]...",
                            total=report.ground_truth.query_count,
                        )
                        metrics = self._evaluate_strategy(
                            store=store,
                            ground_truth=report.ground_truth,
                            strategy_name=run_label,
                            top_k=config.top_k,
                            num_chunks=chunked.total_chunks,
                            progress=progress,
                            task_id=task,
                            bm25=run_bm25,
                            reranker=run_reranker,
                            fetch_k=config.fetch_k,
                        )
                        metrics.embedding_model = model_id
                        metrics.chunk_size = size
                        metrics.chunk_overlap = overlap
                        report.strategy_results.append(metrics)
                        self.console.print(
                            f"  [green]{metrics.retrieval_mode}: "
                            f"Hit Rate: {metrics.hit_rate:.1%} | "
                            f"MRR: {metrics.mrr:.3f} | "
                            f"Recall@{config.top_k}: {metrics.mean_recall_at_k:.1%}[/]"
                        )

                    store.delete_collection()
                successes += 1

        if not report.strategy_results and report.model_errors:
            raise EvalPipelineError(
                "No embedding model could be evaluated: "
                + "; ".join(f"{m}: {e}" for m, e in report.model_errors.items())
            )

        # Best configuration across models, strategies and sizes (by MRR)
        if report.strategy_results:
            best = max(report.strategy_results, key=lambda m: m.mrr)
            report.best_strategy = best.strategy_name
            report.best_model = best.embedding_model
            report.best_chunk_size = best.chunk_size
        return report

    @staticmethod
    def _label(
        strategy_name: str, model_id: str, multi: bool, size: int = 0, sweep: bool = False
    ) -> str:
        """Result label; includes the model and/or chunk size when several are compared."""
        label = strategy_name
        if sweep:
            label += f" @{size}"
        if multi:
            label += f" [{model_id.split('/')[-1]}]"
        return label

    @staticmethod
    def _overlap_for(size: int, config: EvalConfig) -> int:
        """Overlap for a given chunk size: proportional when sweeping, else as configured."""
        if len(config.chunk_sizes) > 1 and config.overlap_ratio is not None:
            return int(size * config.overlap_ratio)
        return config.chunk_overlap

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
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> ChunkedCorpus:
        """Apply a named chunking strategy to the documents at the given size."""
        size = chunk_size if chunk_size is not None else config.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        if strategy_name == "recursive":
            return chunk_recursive(documents, chunk_size=size, chunk_overlap=overlap)
        elif strategy_name == "semantic":
            return chunk_semantic(
                documents,
                embedding_fn=self._embed_texts,
                threshold=0.5,
                min_chunk_size=100,
                max_chunk_size=size * 3,
            )
        elif strategy_name == "hierarchical":
            return chunk_hierarchical(
                documents,
                parent_size=size * 2,
                child_size=size // 2,
                child_overlap=overlap // 2,
            )
        elif strategy_name == "adaptive":
            return chunk_per_file_adaptive(documents, chunk_size=size, chunk_overlap=overlap)
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
        bm25: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
        fetch_k: int = 20,
    ) -> EvalMetrics:
        """Run all ground truth queries against the indexed store.

        ``bm25`` enables hybrid retrieval (RRF fusion with the dense results);
        ``reranker`` re-scores the ``fetch_k`` candidates and keeps ``top_k``.
        """
        query_results = []
        # Fetch more than top_k when a later stage narrows the list.
        candidate_k = max(fetch_k, top_k) if (bm25 is not None or reranker is not None) else top_k

        for i, entry in enumerate(ground_truth.entries):
            # Embed the query
            query_emb = self._embed_texts([entry.query])[0]

            # Search (dense, optionally fused with BM25, optionally reranked)
            search_results = store.search(query_emb, top_k=candidate_k)
            if bm25 is not None:
                sparse = bm25.search(entry.query, top_k=candidate_k)
                search_results = rrf_fuse([search_results, sparse], top_k=candidate_k)
            if reranker is not None:
                search_results = reranker.rerank(entry.query, search_results, top_k=top_k)
            else:
                search_results = search_results[:top_k]

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

        return compute_metrics(
            query_results,
            strategy_name,
            num_chunks,
            retrieval_mode=mode_label(bm25 is not None, reranker is not None),
        )


def display_comparison_table(
    report: EvalReport,
    console: Console | None = None,
) -> None:
    """Display a Rich comparison table of all strategy results."""
    console = console or Console()
    k = report.config.top_k

    multi = len(report.embedding_models) > 1
    title_model = (
        f"{len(report.embedding_models)} models" if multi else report.embedding_model
    )
    table = Table(
        title=f"Evaluation Results — {title_model}",
        show_header=True,
        title_style="bold cyan",
    )
    sweep = len(report.config.chunk_sizes) > 1
    table.add_column("Strategy", style="bold")
    if sweep:
        table.add_column("Size", justify="right")
    if multi:
        table.add_column("Model")
    table.add_column("Mode")
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

        row = [f"{metrics.strategy_name}{marker}"]
        if sweep:
            row.append(str(metrics.chunk_size))
        if multi:
            row.append(metrics.embedding_model.split("/")[-1])
        row += [
            metrics.retrieval_mode,
            str(metrics.num_chunks),
            f"{metrics.hit_rate:.1%}",
            f"{metrics.mrr:.3f}",
            f"{metrics.mean_precision_at_k:.1%}",
            f"{metrics.mean_recall_at_k:.1%}",
            f"{metrics.mean_ndcg_at_k:.3f}",
        ]
        table.add_row(*row, style=style)

    console.print()
    console.print(table)
    console.print(f"\n[bold green]* Best strategy:[/] {report.best_strategy} (by MRR)")
    if sweep:
        console.print(f"[bold green]* Best chunk size:[/] {report.best_chunk_size} characters")
    for model_id, err in report.model_errors.items():
        console.print(f"[yellow]Skipped {model_id}: {err[:120]}[/]")
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
    if len(report.embedding_models) > 1:
        lines.append(
            "**Embedding Models:** " + ", ".join(f"`{m}`" for m in report.embedding_models) + "  "
        )
    else:
        lines.append(f"**Embedding Model:** `{report.embedding_model}`  ")
    for model_id, err in report.model_errors.items():
        lines.append(f"**Skipped:** `{model_id}` ({err[:120]})  ")
    lines.append(f"**Vector Backend:** {report.config.vector_backend}  ")
    if len(report.config.chunk_sizes) > 1:
        lines.append(
            "**Chunk Sizes (sweep):** "
            + ", ".join(str(s) for s in report.config.chunk_sizes)
            + f" characters (best: {report.best_chunk_size})  "
        )
    else:
        lines.append(f"**Chunk Size:** {report.config.chunk_size}  ")
    lines.append(f"**Top-K:** {k}  ")
    lines.append(
        f"**Retrieval mode:** {mode_label(report.config.hybrid, bool(report.config.rerank_model))}"
        + (f" (reranker `{report.config.rerank_model}`)" if report.config.rerank_model else "")
        + "  "
    )
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
        f"| Strategy | Size | Mode | Chunks | Hit Rate@{k} | MRR | Precision@{k} | "
        f"Recall@{k} | NDCG@{k} |"
    )
    lines.append(
        "|----------|------|------|--------|------------|-----|-------------|-----------|---------|"
    )

    for m in report.strategy_results:
        best = " **" if m.strategy_name == report.best_strategy else ""
        end = "**" if best else ""
        lines.append(
            f"| {best}{m.strategy_name}{end} | {m.chunk_size} | {m.retrieval_mode} | "
            f"{m.num_chunks} | "
            f"{m.hit_rate:.1%} | {m.mrr:.3f} | {m.mean_precision_at_k:.1%} | "
            f"{m.mean_recall_at_k:.1%} | {m.mean_ndcg_at_k:.3f} |"
        )

    lines.append("")
    lines.append(f"**Best strategy:** {report.best_strategy} (by MRR)")
    if len(report.config.chunk_sizes) > 1:
        lines.append("")
        lines.append("### Chunk-size sweep (MRR by size, primary retrieval mode)")
        lines.append("")
        sizes = list(report.config.chunk_sizes)
        lines.append("| Strategy | " + " | ".join(str(s) for s in sizes) + " |")
        lines.append("|----------|" + "|".join("------" for _ in sizes) + "|")
        primary = [
            m for m in report.strategy_results
            if not m.strategy_name.endswith("(dense baseline)")
        ]
        groups: dict[str, dict[int, float]] = {}
        for m in primary:
            base = m.strategy_name.split(" @")[0]
            multi_model = len(report.embedding_models) > 1
            short = m.embedding_model.split("/")[-1]
            key = base + (f" [{short}]" if multi_model and short else "")
            groups.setdefault(key, {})[m.chunk_size] = m.mrr
        for key, by_size in groups.items():
            best_size = max(by_size, key=by_size.get) if by_size else None
            cells = []
            for s in sizes:
                v = by_size.get(s)
                cell = "-" if v is None else f"{v:.3f}"
                cells.append(f"**{cell}**" if s == best_size and v is not None else cell)
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"**Best chunk size:** {report.best_chunk_size} characters")
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
