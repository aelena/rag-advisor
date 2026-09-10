"""CLI entry point for ragadvisor."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from rag_adviser import __version__
from rag_adviser.models import (
    RagAdvisorError,
    ReportFormat,
)

app = typer.Typer(
    name="ragadvisor",
    help="RAG Configuration Adviser — rule-based recommendations for building RAG systems.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    # Input mode selectors
    from_xml: Annotated[
        Path | None,
        typer.Option("--from-xml", "-x", help="Path to XML file with pre-filled answers"),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help="Use a built-in or custom preset (e.g., legal-discovery, customer-support). "
                 "With --no-interactive, other flags override the preset values.",
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("--interactive/--no-interactive", help="Interactive mode (default)"),
    ] = True,
    # Phase 1: Document Discovery
    document_path: Annotated[
        Path | None,
        typer.Option("--document-path", "-d", help="Path to document corpus directory"),
    ] = None,
    future_languages: Annotated[
        bool | None,
        typer.Option("--future-languages/--no-future-languages",
                     help="Plan to add more languages later?"),
    ] = None,
    content_type: Annotated[
        str | None,
        typer.Option("--content-type", "-t",
                     help="Content type: prose, code, legal, chat, tabular"),
    ] = None,
    # Phase 2: Use Case & Constraints
    use_case: Annotated[
        str | None,
        typer.Option("--use-case", "-u",
                     help="Use case: question_answering, semantic_search, summarization, "
                          "code_assistance, legal_analysis"),
    ] = None,
    environment: Annotated[
        str | None,
        typer.Option("--environment", "-e",
                     help="Deployment: cloud, on_prem, edge, local"),
    ] = None,
    latency: Annotated[
        str | None,
        typer.Option("--latency", help="Latency budget: <500ms, <2s, batch"),
    ] = None,
    hardware: Annotated[
        str | None,
        typer.Option("--hardware", help="Hardware: cpu_only, gpu_available, limited_ram"),
    ] = None,
    ram_gb: Annotated[
        float | None,
        typer.Option("--ram-gb", help="Available RAM in GB"),
    ] = None,
    vram_gb: Annotated[
        float | None,
        typer.Option("--vram-gb", help="Available VRAM in GB"),
    ] = None,
    budget: Annotated[
        str | None,
        typer.Option("--budget", help="Budget: free, paid_api, self_hosted"),
    ] = None,
    privacy: Annotated[
        str | None,
        typer.Option("--privacy", help="Privacy: none, moderate, strict, air_gapped"),
    ] = None,
    embedding_provider: Annotated[
        str | None,
        typer.Option("--embedding-provider", help="Embedding provider preference"),
    ] = None,
    llm_provider: Annotated[
        str | None,
        typer.Option("--llm-provider", help="LLM provider (or 'none')"),
    ] = None,
    preferred_lib: Annotated[
        list[str] | None,
        typer.Option("--preferred-lib", help="Preferred libraries (repeatable)"),
    ] = None,
    # Phase 3: Query & Operational Patterns
    query_type: Annotated[
        str | None,
        typer.Option("--query-type",
                     help="Query type: short_keywords, natural_questions, multi_turn"),
    ] = None,
    query_complexity: Annotated[
        str | None,
        typer.Option("--query-complexity",
                     help="Query complexity: simple_factual, comparative, "
                          "aggregative, multi_hop"),
    ] = None,
    answer_type: Annotated[
        str | None,
        typer.Option("--answer-type",
                     help="Expected answer type: exact_passage, synthesized, "
                          "yes_no_with_evidence, list_enumeration"),
    ] = None,
    sample_query: Annotated[
        list[str] | None,
        typer.Option("--sample-query",
                     help="Sample queries for tuning (repeatable)"),
    ] = None,
    update_frequency: Annotated[
        str | None,
        typer.Option("--update-frequency",
                     help="Update frequency: never, weekly, daily, realtime"),
    ] = None,
    has_ground_truth: Annotated[
        bool | None,
        typer.Option("--ground-truth/--no-ground-truth",
                     help="Have ground truth evaluation data?"),
    ] = None,
    queries_per_day: Annotated[
        int | None,
        typer.Option("--queries-per-day",
                     help="Expected query volume, drives cost/capacity estimates (default: 1000)"),
    ] = None,
    ground_truth_path: Annotated[
        Path | None,
        typer.Option("--ground-truth-path",
                     help="Ground truth file (JSONL or CSV) used by --validate"),
    ] = None,
    validate_models: Annotated[
        int,
        typer.Option("--validate-models",
                     help="With --validate, compare the top N recommended local embedding "
                          "models on your data (default: 1)"),
    ] = 1,
    validate_chunk_sizes: Annotated[
        str | None,
        typer.Option("--validate-chunk-sizes",
                     help="With --validate, also try these chunk sizes in tokens, "
                          "comma-separated (e.g. 256,512,1024)"),
    ] = None,
    validate: Annotated[
        bool,
        typer.Option("--validate/--no-validate",
                     help="Run the recommended configuration against the ground truth "
                          "and report retrieval metrics (requires ragadvisor[eval])"),
    ] = False,
    # Output options
    output_format: Annotated[
        str,
        typer.Option("--format", "-f",
                     help="Report format: markdown, html, yaml, json, all"),
    ] = "all",
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for reports"),
    ] = Path("./rag_report"),
    use_llm: Annotated[
        bool,
        typer.Option("--use-llm/--no-llm",
                     help="Use LLM for final verification of recommendations"),
    ] = False,
) -> None:
    """Run the full RAG adviser pipeline.

    By default runs in interactive mode. Use --from-xml or --no-interactive with
    flags for non-interactive operation.
    """
    from rag_adviser.main import RAGAdviser

    try:
        # Load preset if specified
        preset_answers = None
        if preset:
            from rag_adviser.presets.manager import PresetManager

            mgr = PresetManager()
            try:
                profile = mgr.load_preset(preset)
                preset_answers = mgr.apply_preset(profile)
                console.print(
                    f"[bold green]Loaded preset:[/] {profile.name} — {profile.description}"
                )
            except ValueError as e:
                console.print(f"[bold red]Error:[/] {e}")
                raise typer.Exit(1) from e

        # Determine input mode
        if from_xml is not None:
            from rag_adviser.input_modes.xml_input import XmlInputParser

            answers = XmlInputParser(from_xml).parse()
        elif not interactive:
            from rag_adviser.input_modes.cli_params import CliParamsCollector

            # Start from the preset (if any) and let explicit flags override it.
            answers = CliParamsCollector.from_options(
                base=preset_answers,
                document_path=document_path,
                future_languages=future_languages,
                content_type=content_type,
                use_case=use_case,
                environment=environment,
                latency=latency,
                hardware=hardware,
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                budget=budget,
                privacy=privacy,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                preferred_libraries=preferred_lib or [],
                query_type=query_type,
                query_complexity=query_complexity,
                expected_answer_type=answer_type,
                sample_queries=sample_query or [],
                update_frequency=update_frequency,
                has_ground_truth=has_ground_truth,
                ground_truth_path=ground_truth_path,
                queries_per_day=queries_per_day,
            )
        else:
            from rag_adviser.input_modes.interactive import InteractiveFlow

            flow = InteractiveFlow(console, preset_answers=preset_answers)
            answers = flow.run()

        answers.use_llm_verification = use_llm
        if queries_per_day is not None:
            answers.expected_queries_per_day = queries_per_day

        # --ground-truth-path / --validate apply to every input mode
        if ground_truth_path is not None:
            if not ground_truth_path.exists():
                console.print(
                    f"[bold red]Error:[/] Ground truth file not found: {ground_truth_path}"
                )
                raise typer.Exit(1)
            answers.ground_truth_path = ground_truth_path
            answers.has_ground_truth = True
        if validate or answers.run_validation:
            if not answers.ground_truth_path:
                console.print(
                    "[bold red]Error:[/] --validate needs a ground truth file "
                    "(--ground-truth-path, or set it in the XML/interactive flow)"
                )
                raise typer.Exit(1)
            if not answers.document_path:
                console.print("[bold red]Error:[/] --validate needs a corpus (--document-path)")
                raise typer.Exit(1)
            answers.run_validation = True
            if validate_models != 1 or not answers.validate_models:
                answers.validate_models = max(validate_models, 1)
            if validate_chunk_sizes:
                try:
                    answers.validate_chunk_sizes = sorted({
                        int(s) for s in validate_chunk_sizes.split(",") if s.strip()
                    })
                except ValueError:
                    console.print(
                        "[bold red]Error:[/] --validate-chunk-sizes must be "
                        "comma-separated integers (tokens)"
                    )
                    raise typer.Exit(1) from None

        # Parse output format
        try:
            fmt = ReportFormat(output_format)
        except ValueError:
            fmt = ReportFormat.ALL

        formats = (
            [ReportFormat.MARKDOWN, ReportFormat.HTML, ReportFormat.YAML, ReportFormat.JSON]
            if fmt == ReportFormat.ALL
            else [fmt]
        )

        # Run the adviser
        adviser = RAGAdviser(console=console)
        adviser.run(answers, output_dir, formats)

    except RagAdvisorError as e:
        console.print(Panel(f"[bold red]Error:[/] {e}", border_style="red"))
        raise typer.Exit(1) from e


@app.command()
def evaluate(
    corpus: Annotated[
        Path,
        typer.Argument(help="Path to document corpus directory"),
    ],
    ground_truth: Annotated[
        Path,
        typer.Argument(help="Path to ground truth file (JSONL or CSV)"),
    ],
    # Embedding model
    model: Annotated[
        list[str] | None,
        typer.Option("--model", "-m",
                     help="Embedding model(s) to evaluate (repeatable; default: all-MiniLM-L6-v2)"),
    ] = None,
    trust_remote_code: Annotated[
        bool,
        typer.Option("--trust-remote-code",
                     help="Allow models that ship custom code (nomic, gte, jina)"),
    ] = False,
    # Chunking strategies to compare
    strategy: Annotated[
        list[str] | None,
        typer.Option("--strategy", "-s",
                     help="Chunking strategies to compare (repeatable). "
                          "Options: recursive, semantic, hierarchical, adaptive, "
                          "speaker_split, row_based"),
    ] = None,
    # Chunk parameters
    chunk_size: Annotated[
        list[int] | None,
        typer.Option("--chunk-size",
                     help="Chunk size in characters; repeat to sweep several sizes "
                          "(default: 512)"),
    ] = None,
    chunk_overlap: Annotated[
        int,
        typer.Option("--chunk-overlap",
                     help="Chunk overlap in characters for a single size (default: 50)"),
    ] = 50,
    overlap_ratio: Annotated[
        float,
        typer.Option("--overlap-ratio",
                     help="When sweeping sizes, overlap = size x ratio (default: 0.1)"),
    ] = 0.1,
    # Retrieval parameters
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of results to retrieve per query"),
    ] = 5,
    # Vector store backend
    backend: Annotated[
        str,
        typer.Option("--backend", "-b",
                     help="Vector store backend: chroma, faiss, pgvector, sqlite, memory"),
    ] = "chroma",
    db_connection: Annotated[
        str,
        typer.Option("--db-connection",
                     help="Connection string: PostgreSQL URL for pgvector, "
                          "file path for sqlite (default: in-memory)"),
    ] = "",
    # Retrieval mode
    hybrid: Annotated[
        bool,
        typer.Option("--hybrid/--no-hybrid",
                     help="Fuse BM25 with dense results (reciprocal rank fusion)"),
    ] = False,
    rerank: Annotated[
        str | None,
        typer.Option("--rerank",
                     help="Cross-encoder model id to rerank the fetched candidates "
                          "(e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)"),
    ] = None,
    fetch_k: Annotated[
        int,
        typer.Option("--fetch-k",
                     help="Candidates to fetch before fusion/reranking (default: 20)"),
    ] = 20,
    dense_baseline: Annotated[
        bool,
        typer.Option("--dense-baseline/--no-dense-baseline",
                     help="Also evaluate plain dense retrieval on the same index for comparison"),
    ] = True,
    # Baseline comparison (CI mode)
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline",
                     help="Path to baseline JSON to compare against (CI regression check)"),
    ] = None,
    save_baseline: Annotated[
        Path | None,
        typer.Option("--save-baseline",
                     help="Save current results as baseline JSON"),
    ] = None,
    regression_threshold: Annotated[
        float,
        typer.Option("--regression-threshold",
                     help="Minimum metric drop to flag as regression (default: 0.02)"),
    ] = 0.02,
    # Output
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for eval report"),
    ] = Path("./eval_results"),
) -> None:
    """Evaluate chunking strategies against ground truth data.

    Runs the full pipeline: chunk -> embed -> index -> query -> compare metrics.
    Requires: pip install ragadvisor[eval]

    Example:
        ragadvisor evaluate ./docs ./queries.jsonl
        ragadvisor evaluate ./docs ./queries.jsonl --strategy recursive --strategy semantic
        ragadvisor evaluate ./docs ./queries.jsonl --backend faiss
        ragadvisor evaluate ./docs ./queries.jsonl --save-baseline baseline.json
        ragadvisor evaluate ./docs ./queries.jsonl --baseline baseline.json
    """
    from rag_adviser.evaluators.pipeline_runner import (
        EvalConfig,
        EvalPipelineRunner,
        display_comparison_table,
        generate_eval_report_markdown,
    )

    models = model or ["sentence-transformers/all-MiniLM-L6-v2"]
    sizes = sorted({s for s in (chunk_size or [512]) if s > 0}) or [512]
    strategies = strategy or ["recursive", "semantic", "hierarchical", "adaptive"]
    valid_strategies = {
        "recursive", "semantic", "hierarchical", "adaptive", "speaker_split", "row_based",
    }
    for s in strategies:
        if s not in valid_strategies:
            console.print(
                f"[bold red]Error:[/] Unknown strategy '{s}'. "
                f"Valid: {', '.join(sorted(valid_strategies))}"
            )
            raise typer.Exit(1)

    config = EvalConfig(
        corpus_path=corpus,
        ground_truth_path=ground_truth,
        embedding_model=models[0],
        embedding_models=models,
        trust_remote_code=trust_remote_code,
        strategies=strategies,
        top_k=top_k,
        chunk_size=sizes[0],
        chunk_sizes=sizes,
        chunk_overlap=chunk_overlap,
        overlap_ratio=overlap_ratio,
        vector_backend=backend,
        db_connection=db_connection or None,
        output_dir=output_dir,
        hybrid=hybrid,
        rerank_model=rerank,
        rerank_trust_remote_code=trust_remote_code,
        fetch_k=fetch_k,
        dense_baseline=dense_baseline,
    )

    try:
        console.print()
        console.print(Panel(
            f"[bold]Corpus:[/] {corpus}\n"
            f"[bold]Ground truth:[/] {ground_truth}\n"
            f"[bold]Model(s):[/] {', '.join(models)}\n"
            f"[bold]Strategies:[/] {', '.join(strategies)}\n"
            f"[bold]Backend:[/] {backend}\n"
            f"[bold]Top-K:[/] {top_k} | [bold]Chunk size(s):[/] "
            f"{', '.join(str(s) for s in sizes)}\n"
            f"[bold]Mode:[/] {'hybrid' if hybrid else 'dense'}"
            f"{' + rerank (' + rerank + ')' if rerank else ''}",
            title="RAG Evaluation",
            border_style="cyan",
        ))
        console.print()

        runner = EvalPipelineRunner(console=console)
        report = runner.run(config)

        # Display comparison table
        display_comparison_table(report, console)

        # Generate report
        report_path = generate_eval_report_markdown(report, output_dir)
        console.print(f"\n[bold green]Report saved:[/] {report_path}")

        # Save baseline if requested
        if save_baseline:
            from rag_adviser.evaluators.baseline import BaselineManager

            mgr = BaselineManager()
            mgr.save_baseline(report, save_baseline)
            console.print(f"[bold green]Baseline saved:[/] {save_baseline}")

        # Compare against baseline if provided
        if baseline:
            from rag_adviser.evaluators.baseline import BaselineManager

            mgr = BaselineManager()
            try:
                saved = mgr.load_baseline(baseline)
                comparison = mgr.compare(
                    report, saved, threshold=regression_threshold
                )
                mgr.display_diff_table(comparison, console)
                if comparison.has_regression:
                    console.print(
                        "\n[bold red]REGRESSION DETECTED — exiting with code 1[/]"
                    )
                    raise typer.Exit(1)
            except FileNotFoundError:
                console.print(
                    f"[bold yellow]Warning:[/] Baseline file not found: {baseline}"
                )

    except RagAdvisorError as e:
        console.print(Panel(f"[bold red]Error:[/] {e}", border_style="red"))
        raise typer.Exit(1) from e
    except ImportError as e:
        console.print(Panel(
            f"[bold red]Missing dependency:[/] {e}\n\n"
            "Install evaluation dependencies:\n"
            "  pip install ragadvisor[eval]        # ChromaDB backend\n"
            "  pip install ragadvisor[eval-pgvector]  # pgvector backend",
            border_style="red",
        ))
        raise typer.Exit(1) from e


@app.command()
def analyze(
    path: Annotated[
        Path,
        typer.Argument(help="Path to document corpus directory"),
    ],
) -> None:
    """Quick document analysis — detect languages, content types, and stats."""
    from rich.table import Table

    from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer

    try:
        analyzer = DocumentAnalyzer()
        console.print(f"\n[bold]Analyzing documents in:[/] {path}\n")
        stats = analyzer.analyze(path)

        # Display results
        table = Table(title="Document Corpus Analysis", show_header=True)
        table.add_column("Property", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Text documents", str(stats.total_files))
        if stats.total_files_all != stats.total_files:
            table.add_row("All files (incl. non-text)", str(stats.total_files_all))
            non_text = ", ".join(
                f"{c} {k}" for k, c in sorted(stats.modalities.items()) if k != "document"
            )
            table.add_row("Non-text modalities", non_text)
        if stats.scanned_pdfs:
            table.add_row(
                "Scanned PDFs (no text layer)",
                f"{stats.scanned_pdfs} of {stats.sampled_pdfs} sampled",
            )
        table.add_row("Total size", f"{stats.total_size_bytes / 1_048_576:.1f} MB")
        table.add_row("Primary language", stats.primary_language)
        table.add_row("CJK content", "Yes" if stats.has_cjk else "No")
        table.add_row("Content type", stats.detected_content_type.value)
        table.add_row("Avg tokens/doc", f"{stats.avg_tokens_per_doc:.0f}")
        table.add_row("Total tokens", f"{stats.total_tokens:,}")

        for ext, count in sorted(stats.file_types.items()):
            table.add_row(f"  {ext} files", str(count))

        for lang, pct in sorted(stats.languages_detected.items(), key=lambda x: -x[1]):
            table.add_row(f"  Language: {lang}", f"{pct:.0%}")

        console.print(table)

    except RagAdvisorError as e:
        console.print(Panel(f"[bold red]Error:[/] {e}", border_style="red"))
        raise typer.Exit(1) from e


@app.command()
def ask(
    query: Annotated[
        str | None,
        typer.Argument(help="Question to ask (omit for interactive mode)"),
    ] = None,
    papers_dir: Annotated[
        Path | None,
        typer.Option("--papers-dir", "-p",
                     help="Path to research papers directory (default: bundled papers)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m",
                     help="Sentence-transformers model for embeddings"),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of results to retrieve"),
    ] = 5,
    use_llm: Annotated[
        bool,
        typer.Option("--use-llm/--no-llm",
                     help="Use LLM to synthesize answers from retrieved passages"),
    ] = False,
) -> None:
    """Ask questions about RAG configuration using curated research knowledge.

    Launches an interactive research assistant that retrieves answers from
    curated research papers. Use --use-llm to synthesize coherent answers.

    Requires: pip install ragadvisor[eval]

    Examples:
        ragadvisor ask
        ragadvisor ask "What chunking strategy should I use for legal documents?"
        ragadvisor ask --use-llm "How does HyDE work?"
    """
    try:
        from rag_adviser.research.assistant import ResearchAssistant

        assistant = ResearchAssistant(
            console=console,
            papers_dir=papers_dir,
            model_id=model,
            use_llm=use_llm,
        )

        if query:
            # Single-shot mode
            result = assistant.ask(query, top_k=top_k)
            assistant.display_answer(result)
        else:
            # Interactive mode
            assistant.run_interactive()

    except RagAdvisorError as e:
        console.print(Panel(f"[bold red]Error:[/] {e}", border_style="red"))
        raise typer.Exit(1) from e
    except ImportError as e:
        console.print(Panel(
            f"[bold red]Missing dependency:[/] {e}\n\n"
            "Install: pip install ragadvisor[eval]",
            border_style="red",
        ))
        raise typer.Exit(1) from e


@app.command()
def presets() -> None:
    """List available built-in and custom presets."""
    from rich.table import Table

    from rag_adviser.presets.manager import PresetManager

    mgr = PresetManager()
    all_presets = mgr.list_presets()

    if not all_presets:
        console.print("[dim]No presets found.[/]")
        return

    table = Table(title="Available Presets", show_header=True, title_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for p in all_presets:
        table.add_row(p.name, p.description, p.source)

    console.print()
    console.print(table)
    console.print("\n[dim]Usage: ragadvisor run --preset <name>[/]")


@app.command("example-corpus")
def example_corpus(
    dest: Annotated[
        Path,
        typer.Argument(help="Directory to create (default: ./ragadvisor-example)"),
    ] = Path("./ragadvisor-example"),
) -> None:
    """Export the bundled example corpus and its ground-truth queries.

    Writes 14 short research notes on RAG to DEST/corpus and 40 hand-written
    evaluation queries to DEST/queries.jsonl, so you can see --validate work
    before preparing your own queries.
    """
    import shutil

    import rag_adviser.research as research_pkg

    src_dir = Path(research_pkg.__file__).parent
    papers = sorted((src_dir / "papers").glob("*.md"))
    gt = src_dir / "ground_truth.jsonl"
    if not papers or not gt.exists():
        console.print("[bold red]Error:[/] bundled example data is missing from this install")
        raise typer.Exit(1)

    corpus_dir = dest / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        shutil.copy2(paper, corpus_dir / paper.name)
    shutil.copy2(gt, dest / "queries.jsonl")

    console.print(
        f"[bold green]Example written to:[/] {dest}\n"
        f"  corpus/        {len(papers)} documents\n"
        f"  queries.jsonl  {sum(1 for _ in gt.open(encoding='utf-8'))} ground-truth queries\n\n"
        "[bold]Try it:[/]\n"
        f"  ragadvisor run --no-interactive -d {corpus_dir} -u question_answering "
        f"--privacy strict \\\n"
        f"      --ground-truth-path {dest / 'queries.jsonl'} --validate\n\n"
        "[dim]Validation needs the [eval] extra: pip install ragadvisor[eval].\n"
        "Add --validate-models 2 or --validate-chunk-sizes 256,512,1024 to compare "
        "options; each extra model or size re-embeds the corpus (and reranks, when "
        "recommended), so budget several minutes per combination on CPU.[/]"
    )


@app.command("bootstrap-queries")
def bootstrap_queries(
    corpus: Annotated[Path, typer.Argument(help="Path to the document corpus")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the JSONL ground truth"),
    ] = Path("./queries.synthetic.jsonl"),
    n: Annotated[int, typer.Option("--n", help="Number of questions to generate")] = 30,
    chunk_chars: Annotated[
        int, typer.Option("--chunk-chars", help="Size of the passages shown to the LLM")
    ] = 1200,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Only show which passages would be used; no LLM calls"),
    ] = False,
) -> None:
    """Generate a synthetic evaluation set from your corpus with an LLM.

    Samples passages evenly across the corpus and asks the configured LLM
    (ANTHROPIC_API_KEY or OPENAI_API_KEY) to write one specific question and
    short answer per passage. Entries are marked "synthetic": true so reports
    flag the metrics as indicative. Only passages, never whole documents, are
    sent to the API.
    """
    from rich.progress import Progress

    from rag_adviser.evaluators.query_bootstrap import (
        generate_queries,
        sample_chunks,
        write_jsonl,
    )
    from rag_adviser.llm.client import LLMClient, detect_llm_config

    try:
        chunks = sample_chunks(corpus, n=n, chunk_chars=chunk_chars)
    except RagAdvisorError as e:
        console.print(Panel(f"[bold red]Error:[/] {e}", border_style="red"))
        raise typer.Exit(1) from e

    files = sorted({c.source_file for c in chunks})
    console.print(
        f"[bold]Sampled {len(chunks)} passages[/] from {len(files)} files "
        f"(~{chunk_chars} characters each)"
    )
    if dry_run:
        for c in chunks:
            preview = " ".join(c.text.split())[:90]
            console.print(f"  [dim]{c.source_file}#{c.chunk_index}[/] {preview}...")
        console.print("\n[dim]Dry run: no questions generated.[/]")
        return

    config = detect_llm_config()
    if config is None:
        console.print(Panel(
            "[bold red]No LLM configured.[/] Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(RAGADVISOR_LLM_MODEL / OPENAI_BASE_URL optional), or use --dry-run.",
            border_style="red",
        ))
        raise typer.Exit(1)
    console.print(f"[dim]Using {config.provider.value} / {config.model}[/]\n")

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Writing questions...", total=len(chunks))
        result = generate_queries(
            LLMClient(config), chunks,
            progress_cb=lambda done, total: progress.update(task, completed=done),
        )

    if not result.queries:
        console.print(Panel(
            f"[bold red]No questions generated[/] ({result.failures} failures). "
            "Check the API key, model name and network.",
            border_style="red",
        ))
        raise typer.Exit(1)

    path = write_jsonl(result.queries, output)
    console.print(
        f"[bold green]Wrote {len(result.queries)} synthetic queries[/] to {path}"
        + (f" [dim]({result.failures} passages skipped)[/]" if result.failures else "")
        + "\n[dim]Entries carry \"synthetic\": true; reports will flag the metrics as "
        "indicative. Review a sample and add real user questions over time.[/]\n\n"
        f"[bold]Next:[/] ragadvisor run --no-interactive -d {corpus} "
        f"--ground-truth-path {path} --validate"
    )


@app.command("refresh-catalogue")
def refresh_catalogue_cmd(
    write: Annotated[
        bool,
        typer.Option("--write/--dry-run",
                     help="Rewrite quality_score values in defaults.yaml (default: dry run)"),
    ] = False,
    min_datasets: Annotated[
        int,
        typer.Option("--min-datasets",
                     help="Minimum MTEB retrieval datasets a model card must report"),
    ] = 10,
    min_delta: Annotated[
        float,
        typer.Option("--min-delta", help="Ignore score changes smaller than this"),
    ] = 0.5,
) -> None:
    """Refresh embedding-model quality scores from Hugging Face MTEB model cards.

    Reads each curated model's card, averages the English MTEB retrieval
    nDCG@10 results, and reports (or writes) the new quality_score values.
    Hosted API models and rerankers are left for manual maintenance.
    """
    from rich.table import Table

    from rag_adviser.catalogue_refresh import DEFAULTS_PATH, refresh_catalogue

    console.print(f"\n[bold]Catalogue:[/] {DEFAULTS_PATH}")
    console.print("[dim]Fetching model cards from the Hugging Face Hub...[/]\n")
    report = refresh_catalogue(write=write, min_datasets=min_datasets, min_delta=min_delta)

    table = Table(title="MTEB retrieval quality (nDCG@10, 0-100)", title_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Old", justify="right")
    table.add_column("New", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Datasets", justify="right")
    table.add_column("Status")
    for r in report.results:
        style = {"updated": "green", "insufficient_data": "yellow", "no_card": "yellow",
                 "error": "red"}.get(r.status, "")
        table.add_row(
            r.model_id, f"{r.old_score:.1f}",
            "-" if r.new_score is None else f"{r.new_score:.1f}",
            "-" if r.delta is None else f"{r.delta:+.1f}",
            str(r.datasets_used), f"{r.status} {r.detail}".strip(), style=style,
        )
    console.print(table)

    if report.updated:
        verb = "Updated" if report.written else "Would update"
        console.print(f"\n[bold]{verb} {len(report.updated)} score(s).[/]")
        if not report.written:
            console.print("[dim]Re-run with --write to apply.[/]")
    else:
        console.print("\n[bold green]Catalogue is up to date.[/]")


@app.command()
def version() -> None:
    """Show ragadvisor version."""
    console.print(f"[bold]ragadvisor[/] v{__version__}")
