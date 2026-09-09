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
    ground_truth_path: Annotated[
        Path | None,
        typer.Option("--ground-truth-path",
                     help="Ground truth file (JSONL or CSV) used by --validate"),
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
                     help="Report format: markdown, html, yaml, all"),
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
            )
        else:
            from rag_adviser.input_modes.interactive import InteractiveFlow

            flow = InteractiveFlow(console, preset_answers=preset_answers)
            answers = flow.run()

        answers.use_llm_verification = use_llm

        # --ground-truth-path / --validate apply to every input mode
        if ground_truth_path is not None:
            if not ground_truth_path.exists():
                console.print(
                    f"[bold red]Error:[/] Ground truth file not found: {ground_truth_path}"
                )
                raise typer.Exit(1)
            answers.ground_truth_path = ground_truth_path
            answers.has_ground_truth = True
        if validate:
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

        # Parse output format
        try:
            fmt = ReportFormat(output_format)
        except ValueError:
            fmt = ReportFormat.ALL

        formats = (
            [ReportFormat.MARKDOWN, ReportFormat.HTML, ReportFormat.YAML]
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
        str,
        typer.Option("--model", "-m",
                     help="Embedding model to use for evaluation"),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
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
                          "Options: recursive, semantic, hierarchical, adaptive"),
    ] = None,
    # Chunk parameters
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Base chunk size in characters"),
    ] = 512,
    chunk_overlap: Annotated[
        int,
        typer.Option("--chunk-overlap", help="Chunk overlap in characters"),
    ] = 50,
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

    strategies = strategy or ["recursive", "semantic", "hierarchical", "adaptive"]
    valid_strategies = {"recursive", "semantic", "hierarchical", "adaptive"}
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
        embedding_model=model,
        trust_remote_code=trust_remote_code,
        strategies=strategies,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_backend=backend,
        db_connection=db_connection or None,
        output_dir=output_dir,
    )

    try:
        console.print()
        console.print(Panel(
            f"[bold]Corpus:[/] {corpus}\n"
            f"[bold]Ground truth:[/] {ground_truth}\n"
            f"[bold]Model:[/] {model}\n"
            f"[bold]Strategies:[/] {', '.join(strategies)}\n"
            f"[bold]Backend:[/] {backend}\n"
            f"[bold]Top-K:[/] {top_k} | [bold]Chunk size:[/] {chunk_size}",
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

        table.add_row("Total files", str(stats.total_files))
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


@app.command()
def version() -> None:
    """Show ragadvisor version."""
    console.print(f"[bold]ragadvisor[/] v{__version__}")
