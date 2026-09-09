"""Research assistant — mini-RAG that answers questions about RAG configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from rag_adviser.research.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = """\
You are a RAG systems expert. Answer the user's question using ONLY the \
provided context passages. Cite sources by name when possible. If the context \
does not contain enough information, say so — do not fabricate information.

Keep your answer concise, practical, and actionable. Use markdown formatting."""


@dataclass
class AnswerResult:
    """Result of a research assistant query."""

    query: str = ""
    answer_chunks: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    synthesized_answer: str = ""


class ResearchAssistant:
    """Interactive research assistant — retrieves relevant knowledge about RAG.

    By default, retrieval-only (no LLM generation). When `use_llm=True`,
    retrieved passages are sent to an LLM for synthesis into a coherent answer.
    """

    def __init__(
        self,
        console: Console | None = None,
        papers_dir: Path | None = None,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_llm: bool = False,
    ) -> None:
        self.console = console or Console()
        self._kb = KnowledgeBase(papers_dir=papers_dir)
        self._model_id = model_id
        self._use_llm = use_llm
        self._llm_client = None
        self._ready = False

    def initialize(self) -> None:
        """Build the knowledge base (downloads model on first run)."""
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Building research knowledge base...", total=None
            )
            self._kb.build(model_id=self._model_id)
            progress.remove_task(task)

        self.console.print(
            f"[dim]Knowledge base ready: {self._kb.chunk_count} chunks indexed[/]"
        )

        # Initialize LLM client if requested
        if self._use_llm:
            self._init_llm()

        self._ready = True

    def _init_llm(self) -> None:
        """Try to set up the LLM client."""
        try:
            from rag_adviser.llm.client import LLMClient, detect_llm_config

            config = detect_llm_config()
            if config:
                self._llm_client = LLMClient(config)
                self.console.print(
                    f"[dim]LLM synthesis enabled: {config.provider.value} ({config.model})[/]"
                )
            else:
                self.console.print(
                    "[yellow]LLM synthesis requested but no API key found. "
                    "Set ANTHROPIC_API_KEY or OPENAI_API_KEY. "
                    "Falling back to retrieval-only mode.[/]"
                )
                self._use_llm = False
        except Exception as e:
            self.console.print(
                f"[yellow]LLM initialization failed: {e}. "
                f"Falling back to retrieval-only mode.[/]"
            )
            self._use_llm = False

    def ask(self, query: str, top_k: int = 5) -> AnswerResult:
        """Ask a question and get relevant passages (+ optional LLM synthesis)."""
        if not self._ready:
            self.initialize()

        results = self._kb.search(query, top_k=top_k)

        answer = AnswerResult(query=query)
        seen_sources = set()

        for chunk, score in results:
            answer.answer_chunks.append({
                "text": chunk.text,
                "source": chunk.source_file,
                "score": score,
            })
            if chunk.source_file not in seen_sources:
                answer.sources.append(chunk.source_file)
                seen_sources.add(chunk.source_file)

        # LLM synthesis
        if self._use_llm and self._llm_client and answer.answer_chunks:
            answer.synthesized_answer = self._synthesize(query, answer.answer_chunks)

        return answer

    def _synthesize(self, query: str, chunks: list[dict]) -> str:
        """Use an LLM to synthesize a coherent answer from retrieved passages."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = Path(chunk["source"]).stem.replace("_", " ").title()
            context_parts.append(
                f"[Passage {i} — {source} (relevance: {chunk['score']:.0%})]\n"
                f"{chunk['text']}"
            )

        user_prompt = (
            "## Context\n\n"
            + "\n\n---\n\n".join(context_parts)
            + f"\n\n## Question\n\n{query}"
        )

        try:
            return self._llm_client.complete(_SYNTHESIS_SYSTEM, user_prompt)
        except Exception as e:
            logger.warning("LLM synthesis failed: %s", e)
            return ""

    def display_answer(self, result: AnswerResult) -> None:
        """Display the answer with Rich formatting."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{result.query}[/]",
                title="Question",
                border_style="cyan",
            )
        )

        # Show synthesized answer first if available
        if result.synthesized_answer:
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(result.synthesized_answer),
                    title="Answer (LLM-synthesized)",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            self.console.print()
            self.console.print("[dim]Source passages below:[/]")

        for i, chunk in enumerate(result.answer_chunks, 1):
            score = chunk["score"]
            source = Path(chunk["source"]).stem.replace("_", " ").title()

            # Color based on relevance score
            if score >= 0.6:
                score_style = "bold green"
            elif score >= 0.4:
                score_style = "bold yellow"
            else:
                score_style = "dim"

            header = Text()
            header.append(f"Result {i}", style="bold")
            header.append(f"  [{score_style}]{score:.0%} match[/]", style=score_style)
            header.append("  from ", style="dim")
            header.append(source, style="bold dim")

            self.console.print()
            self.console.print(header)
            # Render as markdown for proper formatting
            self.console.print(
                Panel(
                    Markdown(chunk["text"]),
                    border_style="blue" if score >= 0.4 else "dim",
                    padding=(0, 1),
                )
            )

        # Sources
        self.console.print()
        sources_text = ", ".join(
            Path(s).stem.replace("_", " ").title() for s in result.sources
        )
        self.console.print(f"[dim]Sources: {sources_text}[/]")

    def run_interactive(self) -> None:
        """Run the interactive Q&A loop."""
        from rich.prompt import Prompt
        from rich.rule import Rule

        mode_desc = (
            "Answers are synthesized by an LLM from retrieved passages."
            if self._use_llm
            else "No LLM generation — just direct passages from the knowledge base."
        )

        self.console.print()
        self.console.print(
            Panel(
                "[bold]RAG Research Assistant[/]\n\n"
                "Ask questions about RAG configuration, chunking strategies,\n"
                "embedding models, vector databases, and best practices.\n\n"
                f"{mode_desc}\n\n"
                "[dim]Type 'quit' or 'exit' to leave. Press Ctrl+C to abort.[/]",
                border_style="#b06040",
            )
        )

        self.initialize()

        while True:
            self.console.print()
            self.console.print(Rule(style="dim"))

            try:
                query = Prompt.ask("\n[bold cyan]Your question[/]")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye![/]")
                break

            query = query.strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                self.console.print("[dim]Goodbye![/]")
                break

            result = self.ask(query)
            self.display_answer(result)
