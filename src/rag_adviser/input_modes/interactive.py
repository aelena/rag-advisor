"""Rich-powered interactive question flow for ragadvisor."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, FloatPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.models import (
    AnswerType,
    BudgetTier,
    ContentType,
    DeploymentTarget,
    DocumentStats,
    HardwareProfile,
    LatencyBudget,
    PrivacyLevel,
    QueryComplexity,
    QueryType,
    UpdateFrequency,
    UseCase,
    UserAnswers,
)


def _select(
    console: Console,
    prompt_text: str,
    choices: dict[str, str],
    default: str | None = None,
) -> str:
    """Display numbered choices and return the selected key."""
    console.print(f"\n[bold]{prompt_text}[/]")
    keys = list(choices.keys())
    for i, (key, desc) in enumerate(choices.items(), 1):
        marker = "[bold cyan]>[/] " if key == default else "  "
        console.print(f"  {marker}[bold]{i}.[/] {desc}")

    while True:
        raw = Prompt.ask(
            "Enter number",
            default=str(keys.index(default) + 1) if default else None,
        )
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except ValueError:
            # Try matching by key name
            if raw.lower() in [k.lower() for k in keys]:
                for k in keys:
                    if k.lower() == raw.lower():
                        return k
        console.print(f"[red]Please enter a number between 1 and {len(keys)}[/]")


class InteractiveFlow:
    """Guide the user through the question flow with Rich UI.

    Phase 1: Document Discovery (Steps 1-3)
    Phase 2: Use Case & Constraints (Steps 4-8)
    Approach Gate: Check if RAG is the right approach
    Phase 3: Query & Operational Patterns (Steps 9-13)
    """

    TOTAL_STEPS = 13

    def __init__(
        self, console: Console, preset_answers: UserAnswers | None = None
    ) -> None:
        self.console = console
        self.answers = preset_answers if preset_answers else UserAnswers()
        self._skip_rag = False
        self._has_preset = preset_answers is not None

    def run(self) -> UserAnswers:
        """Execute the full interactive flow and return collected answers."""
        self._show_welcome_banner()
        self._phase_1_document_discovery()
        self._phase_2_use_case_constraints()
        self._approach_gate()
        if not self._skip_rag:
            self._phase_3_query_patterns()
        self._show_summary()
        return self.answers

    # ── Welcome ────────────────────────────────────────────────────────────

    def _show_welcome_banner(self) -> None:
        """Display the welcome banner."""
        from rich.style import Style

        logo_lines = [
            r"   ____      _     ____    _    ______     _____ ____   ___  ____",
            r"  |  _ \    / \   / ___|  / \  |  _ \ \   / /_ _/ ___| / _ \|  _ \ ",
            r"  | |_) |  / _ \ | |  _  / _ \ | | | \ \ / / | |\___ \| | | | |_) |",
            r"  |  _ <  / ___ \| |_| |/ ___ \| |_| |\ V /  | | ___) | |_| |  _ < ",
            r"  |_| \_\/_/   \_\ \___/_/   \_\____/  \_/  |___|____/ \___/|_| \_\ ",
        ]

        rust = Style(color="#e8814a", bold=True)
        dark_rust = Style(color="#c0623a", bold=True)
        subtitle = Style(color="#d4845a", bold=True)
        tagline = Style(color="#b07050", dim=True)
        body = Style(color="#c89070")
        hint = Style(dim=True, italic=True)

        banner = Text()
        split = 21  # "RAG" portion width
        for line in logo_lines:
            banner.append(line[:split], style=rust)
            banner.append(line[split:] + "\n", style=dark_rust)

        banner.append("\n")
        banner.append("  RAG Configuration Adviser", style=subtitle)
        banner.append("  -- rule-based recommendations for RAG systems\n\n", style=tagline)
        banner.append(
            "  This tool will guide you through a series of questions to generate\n"
            "  optimal configuration recommendations for your RAG system.\n\n",
            style=body,
        )
        banner.append("  Press Ctrl+C at any time to abort.", style=hint)

        self.console.print(Panel(
            banner,
            border_style="#b06040",
            padding=(1, 2),
        ))
        self.console.print()

    # ── Phase 1: Document Discovery ────────────────────────────────────────

    def _phase_1_document_discovery(self) -> None:
        """Steps 1-3: Document path, auto-analysis, content type."""
        self.console.print(Rule("[bold blue]Phase 1: Document Discovery[/]", style="blue"))
        self.console.print()

        # Step 1: Document path
        self.console.print(
            f"[bold]Step 1/{self.TOTAL_STEPS}:[/] Provide a path to your document corpus"
        )
        self.console.print("[dim]This can be a directory with your documents or a single file.[/]")
        path_str = Prompt.ask(
            "Document path (or press Enter to skip)",
            default="",
        )

        if path_str.strip():
            doc_path = Path(path_str.strip())
            if doc_path.exists():
                self.answers.document_path = doc_path

                # Step 2: Auto-analysis with spinner
                self.console.print()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                    transient=True,
                ) as progress:
                    progress.add_task("Analyzing documents...", total=None)
                    analyzer = DocumentAnalyzer()
                    self.answers.document_stats = analyzer.analyze(doc_path)

                self._display_document_stats(self.answers.document_stats)
            else:
                self.console.print(f"[yellow]Path not found: {doc_path}. Skipping analysis.[/]")
                self.answers.document_stats = DocumentStats()
        else:
            self.console.print("[dim]No document path provided. Using defaults.[/]")
            self.answers.document_stats = DocumentStats()

        # Step 3: Future languages and content type
        self.console.print()
        self.console.print(f"[bold]Step 2/{self.TOTAL_STEPS}:[/] Language planning")
        self.answers.future_languages = Confirm.ask(
            "Will you add more languages in the future?",
            default=False,
        )

        if self.answers.future_languages:
            self.console.print(
                Panel(
                    "[bold yellow]Important:[/] Since you plan to add languages later,\n"
                    "a multilingual embedding model will be recommended from the start.\n"
                    "Changing the model later requires re-embedding ALL documents.",
                    border_style="yellow",
                )
            )

        self.console.print()
        self.console.print(f"[bold]Step 3/{self.TOTAL_STEPS}:[/] Content type")
        content_choices = {
            ContentType.PROSE.value: "General prose (articles, wikis, docs)",
            ContentType.CODE.value: "Technical docs / Code",
            ContentType.LEGAL.value: "Legal / Contracts",
            ContentType.CHAT.value: "Chat logs / Transcripts",
            ContentType.TABULAR.value: "Tabular / Structured data",
            ContentType.SCIENTIFIC.value: "Scientific / Academic papers",
            ContentType.MIXED.value: "Mixed content types",
        }

        detected_default = (
            self.answers.document_stats.detected_content_type.value
            if self.answers.document_stats
            else ContentType.PROSE.value
        )

        selected = _select(
            self.console,
            f"What is the primary content type? (detected: {detected_default})",
            content_choices,
            default=detected_default,
        )
        if selected != detected_default:
            self.answers.content_type_override = ContentType(selected)

    def _display_document_stats(self, stats: DocumentStats) -> None:
        """Display analysis results in a Rich table."""
        self.console.print()
        table = Table(title="Document Analysis Results", show_header=True, title_style="bold green")
        table.add_column("Property", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Total files", str(stats.total_files))
        table.add_row("Total size", f"{stats.total_size_bytes / 1_048_576:.1f} MB")
        table.add_row("Primary language", stats.primary_language)
        table.add_row("CJK content detected", "Yes" if stats.has_cjk else "No")
        table.add_row("Content type", stats.detected_content_type.value)
        table.add_row("Avg tokens/doc", f"{stats.avg_tokens_per_doc:.0f}")
        table.add_row("Total tokens", f"{stats.total_tokens:,}")

        if stats.languages_detected:
            langs = ", ".join(
                f"{lang} ({pct:.0%})"
                for lang, pct in sorted(
                    stats.languages_detected.items(), key=lambda x: -x[1]
                )
            )
            table.add_row("Languages", langs)

        if stats.file_types:
            types = ", ".join(
                f"{ext} ({count})"
                for ext, count in sorted(stats.file_types.items())
            )
            table.add_row("File types", types)

        self.console.print(table)

    # ── Phase 2: Use Case & Constraints ────────────────────────────────────

    def _phase_2_use_case_constraints(self) -> None:
        """Steps 4-8: Use case, deployment, hardware, privacy, models."""
        self.console.print()
        self.console.print(Rule("[bold blue]Phase 2: Use Case & Constraints[/]", style="blue"))
        self.console.print()

        # Step 4: Use case
        self.console.print(f"[bold]Step 4/{self.TOTAL_STEPS}:[/] Primary use case")
        use_case_choices = {
            UseCase.QA.value: "Q&A over documentation",
            UseCase.SEARCH.value: "Semantic search (no generation)",
            UseCase.SUMMARIZATION.value: "Summarization of results",
            UseCase.CODE.value: "Code assistance",
            UseCase.LEGAL.value: "Legal / contract analysis",
        }
        selected_uc = _select(
            self.console, "What is the main use case?", use_case_choices, default=UseCase.QA.value
        )
        self.answers.use_case = UseCase(selected_uc)

        # Step 5: Deployment
        self.console.print()
        self.console.print(f"[bold]Step 5/{self.TOTAL_STEPS}:[/] Deployment environment")
        deploy_choices = {
            DeploymentTarget.CLOUD.value: "Cloud (AWS, GCP, Azure)",
            DeploymentTarget.ON_PREM.value: "On-premises server",
            DeploymentTarget.EDGE.value: "Edge device (IoT, mobile)",
            DeploymentTarget.LOCAL.value: "Local development machine",
        }
        selected_deploy = _select(
            self.console, "Where will this be deployed?", deploy_choices,
            default=DeploymentTarget.LOCAL.value,
        )
        self.answers.constraints.environment = DeploymentTarget(selected_deploy)

        # Step 6: Hardware
        self.console.print()
        self.console.print(f"[bold]Step 6/{self.TOTAL_STEPS}:[/] Hardware constraints")

        latency_choices = {
            LatencyBudget.FAST.value: "Fast (<500ms) — real-time applications",
            LatencyBudget.MODERATE.value: "Moderate (<2s) — typical web apps",
            LatencyBudget.BATCH.value: "Batch — no latency requirements",
        }
        selected_latency = _select(
            self.console, "Latency budget?", latency_choices, default=LatencyBudget.MODERATE.value
        )
        self.answers.constraints.latency_budget = LatencyBudget(selected_latency)

        hardware_choices = {
            HardwareProfile.CPU_ONLY.value: "CPU only",
            HardwareProfile.GPU_AVAILABLE.value: "GPU available",
            HardwareProfile.LIMITED_RAM.value: "Limited RAM (<8GB)",
        }
        selected_hw = _select(
            self.console, "Available hardware?", hardware_choices,
            default=HardwareProfile.CPU_ONLY.value,
        )
        self.answers.constraints.hardware = HardwareProfile(selected_hw)

        self.answers.constraints.ram_gb = FloatPrompt.ask("Available RAM (GB)", default=16.0)

        if self.answers.constraints.hardware == HardwareProfile.GPU_AVAILABLE:
            self.answers.constraints.vram_gb = FloatPrompt.ask("Available VRAM (GB)", default=8.0)

        # Step 7: Budget and privacy
        self.console.print()
        self.console.print(f"[bold]Step 7/{self.TOTAL_STEPS}:[/] Budget & Privacy")

        budget_choices = {
            BudgetTier.FREE.value: "Free / open-source only",
            BudgetTier.PAID_API.value: "Paid API (OpenAI, Cohere, etc.)",
            BudgetTier.SELF_HOSTED.value: "Self-hosted (own infrastructure)",
        }
        selected_budget = _select(
            self.console, "Budget tier?", budget_choices, default=BudgetTier.FREE.value
        )
        self.answers.constraints.budget = BudgetTier(selected_budget)

        privacy_choices = {
            PrivacyLevel.NONE.value: "No restrictions — data can be sent externally",
            PrivacyLevel.MODERATE.value: "Moderate — no raw data to external APIs",
            PrivacyLevel.STRICT.value: "Strict — all processing must be local",
            PrivacyLevel.AIR_GAPPED.value: "Air-gapped — zero network access",
        }
        selected_privacy = _select(
            self.console, "Privacy requirements?", privacy_choices, default=PrivacyLevel.NONE.value
        )
        self.answers.constraints.privacy = PrivacyLevel(selected_privacy)

        # Step 8: Model preferences and libraries
        self.console.print()
        self.console.print(f"[bold]Step 8/{self.TOTAL_STEPS}:[/] Model & Library preferences")

        self.answers.embedding_provider = Prompt.ask(
            "Preferred embedding provider (openai/huggingface/cohere/none)",
            default="huggingface",
        )

        self.answers.llm_provider = Prompt.ask(
            "LLM for generation (gpt-4/llama-3/mistral/none)",
            default="none",
        )

        self.answers.implementation_lang = Prompt.ask(
            "Implementation language",
            default="python",
        )

        libs_str = Prompt.ask(
            "Preferred libraries (comma-separated, or press Enter for none)",
            default="",
        )
        if libs_str.strip():
            self.answers.preferred_libraries = [
                lib.strip() for lib in libs_str.split(",") if lib.strip()
            ]

    # ── Approach Gate ──────────────────────────────────────────────────────

    def _approach_gate(self) -> None:
        """Check if RAG is the right approach before continuing."""
        from rag_adviser.analyzers.approach_analyzer import ApproachAnalyzer

        analyzer = ApproachAnalyzer()
        assessment = analyzer.assess(self.answers)

        if not assessment.proceed_with_rag:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold yellow]Approach Assessment[/]\n\n"
                    f"{assessment.reasoning}\n\n"
                    f"[bold]Recommended alternative:[/] "
                    f"{assessment.recommended_approach.value.replace('_', ' ').title()}\n\n"
                    f"{assessment.alternative_description}",
                    title="RAG May Not Be the Best Approach",
                    border_style="yellow",
                )
            )
            self.console.print()
            proceed = Confirm.ask(
                "Continue with RAG recommendations anyway?",
                default=False,
            )
            if not proceed:
                self._skip_rag = True
                self.console.print()
                self.console.print(
                    "[bold green]Understood.[/] The report will include the "
                    "recommended alternative approach instead of RAG."
                )

    # ── Phase 3: Query & Operational Patterns ──────────────────────────────

    def _phase_3_query_patterns(self) -> None:
        """Steps 9-13: Query type, complexity, answer type, update freq, ground truth."""
        self.console.print()
        self.console.print(
            Rule("[bold blue]Phase 3: Query & Operational Patterns[/]", style="blue")
        )
        self.console.print()

        # Step 9: Query type
        self.console.print(f"[bold]Step 9/{self.TOTAL_STEPS}:[/] Expected query patterns")
        query_choices = {
            QueryType.SHORT_KEYWORDS.value: 'Short keywords ("invoice 2024")',
            QueryType.NATURAL_QUESTIONS.value: 'Natural questions ("What was our Q3 revenue?")',
            QueryType.MULTI_TURN.value: "Multi-turn conversational queries",
        }
        selected_query = _select(
            self.console,
            "What will user queries look like?",
            query_choices,
            default=QueryType.NATURAL_QUESTIONS.value,
        )
        self.answers.query_type = QueryType(selected_query)

        # Step 10: Query complexity
        self.console.print()
        self.console.print(f"[bold]Step 10/{self.TOTAL_STEPS}:[/] Query complexity")
        complexity_choices = {
            QueryComplexity.SIMPLE_FACTUAL.value: 'Simple factual ("What is the return policy?")',
            QueryComplexity.COMPARATIVE.value: 'Comparative ("How does product A compare to B?")',
            QueryComplexity.AGGREGATIVE.value: 'Aggregative ("Summarize all findings on topic X")',
            QueryComplexity.MULTI_HOP.value: 'Multi-hop ("What did the author of X say about Y?")',
        }
        selected_complexity = _select(
            self.console,
            "How complex are the typical queries?",
            complexity_choices,
            default=QueryComplexity.SIMPLE_FACTUAL.value,
        )
        self.answers.query_complexity = QueryComplexity(selected_complexity)

        # Step 11: Expected answer type
        self.console.print()
        self.console.print(f"[bold]Step 11/{self.TOTAL_STEPS}:[/] Expected answer type")
        answer_choices = {
            AnswerType.EXACT_PASSAGE.value: "Exact passage retrieval (find the right paragraph)",
            AnswerType.SYNTHESIZED.value: "Synthesized answer (combine info from multiple chunks)",
            AnswerType.YES_NO_WITH_EVIDENCE.value: "Yes/no with evidence",
            AnswerType.LIST_ENUMERATION.value: "List or enumeration of items",
        }
        selected_answer = _select(
            self.console,
            "What kind of answers do users expect?",
            answer_choices,
            default=AnswerType.EXACT_PASSAGE.value,
        )
        self.answers.expected_answer_type = AnswerType(selected_answer)

        # Optional sample queries
        self.console.print()
        self.console.print(
            "[dim]Optionally provide 2-3 example queries (helps tune recommendations).[/]"
        )
        sample_str = Prompt.ask(
            "Sample queries (semicolon-separated, or press Enter to skip)",
            default="",
        )
        if sample_str.strip():
            self.answers.sample_queries = [
                q.strip() for q in sample_str.split(";") if q.strip()
            ]

        # Step 12: Update frequency
        self.console.print()
        self.console.print(f"[bold]Step 12/{self.TOTAL_STEPS}:[/] Document update frequency")
        update_choices = {
            UpdateFrequency.NEVER.value: "Never / Rarely — static corpus",
            UpdateFrequency.WEEKLY.value: "Weekly updates",
            UpdateFrequency.DAILY.value: "Daily updates",
            UpdateFrequency.REALTIME.value: "Real-time streaming",
        }
        selected_update = _select(
            self.console,
            "How often do documents change?",
            update_choices,
            default=UpdateFrequency.NEVER.value,
        )
        self.answers.update_frequency = UpdateFrequency(selected_update)

        qpd = Prompt.ask(
            "Expected queries per day (drives cost/capacity estimates)",
            default=str(self.answers.expected_queries_per_day),
        )
        try:
            self.answers.expected_queries_per_day = max(int(qpd.replace(",", "").strip()), 0)
        except ValueError:
            self.console.print("[dim]Not a number; keeping the default.[/]")

        # Step 13: Ground truth
        self.console.print()
        self.console.print(f"[bold]Step 13/{self.TOTAL_STEPS}:[/] Evaluation data")
        self.answers.has_ground_truth = Confirm.ask(
            "Do you have ground truth data (queries + correct answers) for evaluation?",
            default=False,
        )

        if self.answers.has_ground_truth:
            gt_path = Prompt.ask("Path to ground truth file (or press Enter to skip)", default="")
            if gt_path.strip():
                self.answers.ground_truth_path = Path(gt_path.strip())
                if self.answers.document_path:
                    self.answers.run_validation = Confirm.ask(
                        "Validate the recommended configuration against it now? "
                        "(embeds your corpus locally; needs ragadvisor[eval])",
                        default=False,
                    )
                if self.answers.run_validation:
                    models_n = Prompt.ask(
                        "How many of the recommended embedding models should be compared? (1-3)",
                        default="1",
                    )
                    try:
                        self.answers.validate_models = min(max(int(models_n), 1), 3)
                    except ValueError:
                        self.console.print("[dim]Not a number; comparing 1 model.[/]")
                    sizes = Prompt.ask(
                        "Extra chunk sizes to try, in tokens (comma-separated, Enter to skip)",
                        default="",
                    )
                    if sizes.strip():
                        try:
                            self.answers.validate_chunk_sizes = sorted({
                                int(s) for s in sizes.split(",") if s.strip()
                            })
                        except ValueError:
                            self.console.print("[dim]Could not parse sizes; skipping the sweep.[/]")

        # LLM verification option
        self.console.print()
        self.answers.use_llm_verification = Confirm.ask(
            "Use LLM for final verification of recommendations?",
            default=False,
        )

    # ── Summary ────────────────────────────────────────────────────────────

    def _show_summary(self) -> None:
        """Display a summary of collected answers before proceeding."""
        self.console.print()
        self.console.print(Rule("[bold green]Input Summary[/]", style="green"))
        self.console.print()

        table = Table(show_header=True, title_style="bold")
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value", style="white")

        if self.answers.document_path:
            table.add_row("Document path", str(self.answers.document_path))
        if self.answers.document_stats:
            table.add_row("Files analyzed", str(self.answers.document_stats.total_files))
            table.add_row("Primary language", self.answers.document_stats.primary_language)

        table.add_row("Future languages", "Yes" if self.answers.future_languages else "No")
        table.add_row("Use case", self.answers.use_case.value)
        table.add_row("Deployment", self.answers.constraints.environment.value)
        table.add_row("Hardware", self.answers.constraints.hardware.value)
        table.add_row("RAM", f"{self.answers.constraints.ram_gb}GB")
        table.add_row("Latency", self.answers.constraints.latency_budget.value)
        table.add_row("Privacy", self.answers.constraints.privacy.value)
        table.add_row("Budget", self.answers.constraints.budget.value)

        if not self._skip_rag:
            table.add_row("Query type", self.answers.query_type.value)
            table.add_row("Query complexity", self.answers.query_complexity.value)
            table.add_row("Answer type", self.answers.expected_answer_type.value)
            if self.answers.sample_queries:
                table.add_row("Sample queries", "; ".join(self.answers.sample_queries))
            table.add_row("Update frequency", self.answers.update_frequency.value)
            table.add_row("Ground truth", "Yes" if self.answers.has_ground_truth else "No")
            if self.answers.run_validation:
                extras = f"{self.answers.validate_models} model(s)"
                if self.answers.validate_chunk_sizes:
                    sizes = ", ".join(str(s) for s in self.answers.validate_chunk_sizes)
                    extras += f", chunk sizes {sizes}"
                table.add_row("Validation", f"Yes ({extras})")
            table.add_row("Queries per day", f"{self.answers.expected_queries_per_day:,}")
        else:
            table.add_row("Approach", "[yellow]Alternative (non-RAG) recommended[/]")

        self.console.print(table)
        self.console.print()

        if not Confirm.ask("Proceed with these settings?", default=True):
            self.console.print("[yellow]Aborted by user.[/]")
            raise SystemExit(0)

        # Offer to save as custom preset
        if Confirm.ask(
            "Save these answers as a custom preset for future use?",
            default=False,
        ):
            name = Prompt.ask("Preset name (e.g., my-project)")
            desc = Prompt.ask("Description (optional)", default="")
            fmt = _select(
                self.console,
                "Save format?",
                {"yaml": "YAML", "json": "JSON"},
                default="yaml",
            )
            try:
                from rag_adviser.presets.manager import PresetManager

                path = PresetManager().save_preset(
                    name, self.answers, desc, fmt=fmt
                )
                self.console.print(f"[bold green]Preset saved:[/] {path}")
            except Exception as e:
                self.console.print(f"[yellow]Failed to save preset: {e}[/]")

        self.console.print()
        self.console.print("[bold green]Generating recommendations...[/]")
        self.console.print()
