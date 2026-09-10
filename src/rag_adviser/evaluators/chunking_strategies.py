"""Dynamic chunking strategies for evaluation — applies multiple strategies to a corpus."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag_adviser.models import RagAdvisorError


class ChunkingError(RagAdvisorError):
    """Failed to chunk documents."""


@dataclass
class Chunk:
    """A single chunk produced by a chunking strategy."""

    text: str = ""
    source_file: str = ""
    chunk_index: int = 0
    strategy: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkedCorpus:
    """Result of applying a chunking strategy to a corpus."""

    strategy_name: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    description: str = ""

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


def load_documents(corpus_path: Path) -> list[tuple[str, str]]:
    """Load all text documents from a directory.

    Returns list of (filename, text_content) tuples.
    """
    corpus_path = Path(corpus_path)
    docs: list[tuple[str, str]] = []

    text_extensions = {
        ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl",
        ".xml", ".yaml", ".yml", ".py", ".js", ".ts", ".java", ".go",
        ".rs", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".sql",
        ".html", ".css", ".sh", ".bat", ".ps1", ".r", ".scala",
    }

    if corpus_path.is_file():
        text = _read_file(corpus_path)
        if text:
            docs.append((corpus_path.name, text))
    else:
        for file_path in sorted(corpus_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                text = _read_file(file_path)
                if text:
                    # Use relative path as identifier
                    rel = str(file_path.relative_to(corpus_path))
                    docs.append((rel, text))

    return docs


def _read_file(path: Path) -> str:
    """Read a text file, handling encoding errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""


# ── Chunking Strategy Implementations ─────────────────────────────────


def chunk_recursive(
    documents: list[tuple[str, str]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> ChunkedCorpus:
    """Recursive character splitting — baseline strategy.

    Splits on paragraph boundaries, then sentences, then words.
    """
    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[Chunk] = []

    for filename, text in documents:
        parts = _recursive_split(text, separators, chunk_size, chunk_overlap)
        for i, part in enumerate(parts):
            chunks.append(Chunk(
                text=part,
                source_file=filename,
                chunk_index=i,
                strategy="recursive",
                metadata={"source": filename, "chunk_index": i},
            ))

    return ChunkedCorpus(
        strategy_name="recursive",
        chunks=chunks,
        description=f"Recursive splitting (size={chunk_size}, overlap={chunk_overlap})",
    )


def chunk_semantic(
    documents: list[tuple[str, str]],
    embedding_fn: callable,
    threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1500,
) -> ChunkedCorpus:
    """Semantic chunking — split at topic boundaries using embedding similarity.

    Groups consecutive sentences until the cosine similarity between adjacent
    sentence groups drops below the threshold.
    """
    chunks: list[Chunk] = []

    for filename, text in documents:
        sentences = _split_into_sentences(text)
        if not sentences:
            continue

        if len(sentences) <= 2:
            chunks.append(Chunk(
                text=text.strip(),
                source_file=filename,
                chunk_index=0,
                strategy="semantic",
                metadata={"source": filename, "chunk_index": 0},
            ))
            continue

        # Embed all sentences
        embeddings = embedding_fn(sentences)

        # Find breakpoints where similarity drops
        current_group: list[str] = [sentences[0]]
        chunk_idx = 0

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            current_text = " ".join(current_group)

            if sim < threshold and len(current_text) >= min_chunk_size:
                chunks.append(Chunk(
                    text=current_text,
                    source_file=filename,
                    chunk_index=chunk_idx,
                    strategy="semantic",
                    metadata={"source": filename, "chunk_index": chunk_idx},
                ))
                current_group = [sentences[i]]
                chunk_idx += 1
            elif len(current_text) >= max_chunk_size:
                # Force split if chunk is getting too large
                chunks.append(Chunk(
                    text=current_text,
                    source_file=filename,
                    chunk_index=chunk_idx,
                    strategy="semantic",
                    metadata={"source": filename, "chunk_index": chunk_idx},
                ))
                current_group = [sentences[i]]
                chunk_idx += 1
            else:
                current_group.append(sentences[i])

        # Flush remaining
        if current_group:
            remaining = " ".join(current_group)
            if remaining.strip():
                chunks.append(Chunk(
                    text=remaining,
                    source_file=filename,
                    chunk_index=chunk_idx,
                    strategy="semantic",
                    metadata={"source": filename, "chunk_index": chunk_idx},
                ))

    return ChunkedCorpus(
        strategy_name="semantic",
        chunks=chunks,
        description=f"Semantic splitting (similarity threshold={threshold})",
    )


def chunk_hierarchical(
    documents: list[tuple[str, str]],
    parent_size: int = 1024,
    child_size: int = 256,
    child_overlap: int = 25,
) -> ChunkedCorpus:
    """Hierarchical parent-child chunking.

    Creates small child chunks for precise retrieval, each linked to a
    larger parent chunk for context. The retriever searches child chunks
    but returns parent chunks.
    """
    chunks: list[Chunk] = []

    for filename, text in documents:
        # Create parent chunks
        parent_parts = _recursive_split(
            text, ["\n\n", "\n", ". "], parent_size, overlap=0
        )

        for parent_idx, parent_text in enumerate(parent_parts):
            # Create child chunks within this parent
            child_parts = _recursive_split(
                parent_text, ["\n", ". ", " "], child_size, child_overlap
            )

            for child_idx, child_text in enumerate(child_parts):
                chunks.append(Chunk(
                    text=child_text,
                    source_file=filename,
                    chunk_index=parent_idx * 100 + child_idx,
                    strategy="hierarchical",
                    metadata={
                        "source": filename,
                        "parent_index": parent_idx,
                        "child_index": child_idx,
                        "parent_text": parent_text,
                    },
                ))

    return ChunkedCorpus(
        strategy_name="hierarchical",
        chunks=chunks,
        description=(
            f"Hierarchical parent-child (parent={parent_size}, "
            f"child={child_size}, overlap={child_overlap})"
        ),
    )


def chunk_per_file_adaptive(
    documents: list[tuple[str, str]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> ChunkedCorpus:
    """Per-file adaptive chunking — detects content type per file and
    applies the best strategy for each.
    """
    chunks: list[Chunk] = []

    for filename, text in documents:
        file_type = _detect_file_type(filename, text)

        if file_type == "code":
            parts = _split_code(text, chunk_size)
        elif file_type == "markdown":
            parts = _split_markdown(text, chunk_size, chunk_overlap)
        else:
            parts = _recursive_split(
                text, ["\n\n", "\n", ". ", " "], chunk_size, chunk_overlap
            )

        for i, part in enumerate(parts):
            chunks.append(Chunk(
                text=part,
                source_file=filename,
                chunk_index=i,
                strategy="adaptive",
                metadata={
                    "source": filename,
                    "chunk_index": i,
                    "detected_type": file_type,
                },
            ))

    return ChunkedCorpus(
        strategy_name="adaptive",
        chunks=chunks,
        description=f"Per-file adaptive (base size={chunk_size})",
    )


# ── Helper functions ──────────────────────────────────────────────────


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int = 0,
) -> list[str]:
    """Split text recursively using separator hierarchy."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try each separator
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            result: list[str] = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) > chunk_size and current:
                    result.append(current.strip())
                    # Apply overlap
                    if overlap > 0 and len(current) > overlap:
                        current = current[-overlap:] + sep + part
                    else:
                        current = part
                else:
                    current = candidate

            if current.strip():
                result.append(current.strip())

            return [r for r in result if r]

    # No separator worked — force character split
    result = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            result.append(chunk)
    return result


def _split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter using regex."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _detect_file_type(filename: str, text: str) -> str:
    """Detect content type of a single file."""
    ext = Path(filename).suffix.lower()

    code_extensions = {
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp",
        ".h", ".hpp", ".rb", ".php", ".sql", ".sh", ".bat", ".r",
        ".scala", ".swift", ".kt",
    }
    if ext in code_extensions:
        return "code"

    if ext in (".md", ".rst"):
        return "markdown"

    # Heuristic: detect code content even in .txt files
    code_patterns = ["def ", "function ", "class ", "import ", "from ", "#include"]
    code_matches = sum(1 for p in code_patterns if p in text[:2000])
    if code_matches >= 3:
        return "code"

    if text[:2000].count("#") >= 3 and text[:2000].count("\n") >= 5:
        return "markdown"

    return "prose"


def _split_code(text: str, chunk_size: int) -> list[str]:
    """Split code at function/class boundaries."""
    # Split at top-level definitions
    pattern = r'\n(?=(?:def |class |function |async def |export |public |private ))'
    parts = re.split(pattern, text)

    result: list[str] = []
    current = ""

    for part in parts:
        if len(current) + len(part) > chunk_size and current:
            result.append(current.strip())
            current = part
        else:
            current += part

    if current.strip():
        result.append(current.strip())

    return result or [text.strip()]


def _split_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split markdown at heading boundaries."""
    # Split at headings
    parts = re.split(r'\n(?=#{1,4} )', text)

    result: list[str] = []
    current = ""

    for part in parts:
        if len(current) + len(part) > chunk_size and current:
            result.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n" + part
            else:
                current = part
        else:
            current = (current + "\n" + part) if current else part

    if current.strip():
        result.append(current.strip())

    return result or [text.strip()]


# ── Speaker-turn chunking (chat logs, transcripts) ──────────────────────────

_SPEAKER_TURN_RE = re.compile(
    r"^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*)?"          # optional [12:34] timestamp
    r"(?:<[^>]{1,40}>|[A-Za-z][\w .'\-]{0,40}:)\s",       # "<alice>" or "Alice:" / "user:"
    re.MULTILINE,
)


def _split_turns(text: str) -> list[str]:
    """Split a transcript into speaker turns; [] when no speaker markers are found."""
    starts = [m.start() for m in _SPEAKER_TURN_RE.finditer(text)]
    if len(starts) < 2:
        return []
    if starts[0] > 0:
        starts.insert(0, 0)
    turns = [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)], strict=False)]
    return [t for t in turns if t]


def chunk_speaker_turns(
    documents: list[tuple[str, str]],
    chunk_size: int = 512,
) -> ChunkedCorpus:
    """Speaker-turn chunking for chat logs and transcripts.

    Consecutive turns are grouped until ``chunk_size`` characters is reached;
    a turn is never split. Files without detectable speaker markers fall back
    to recursive splitting so mixed corpora still work.
    """
    chunks: list[Chunk] = []
    for filename, text in documents:
        turns = _split_turns(text)
        if not turns:
            parts = _recursive_split(text, ["\n\n", "\n", ". ", " "], chunk_size, 0)
            fallback = True
        else:
            parts, current = [], ""
            for turn in turns:
                if current and len(current) + len(turn) + 1 > chunk_size:
                    parts.append(current)
                    current = turn
                else:
                    current = f"{current}\n{turn}" if current else turn
            if current:
                parts.append(current)
            fallback = False
        for i, part in enumerate(parts):
            chunks.append(Chunk(
                text=part,
                source_file=filename,
                chunk_index=i,
                strategy="speaker_split",
                metadata={"source": filename, "chunk_index": i,
                          "speaker_turns": not fallback},
            ))
    return ChunkedCorpus(
        strategy_name="speaker_split",
        chunks=chunks,
        description=f"Speaker-turn grouping (max {chunk_size} chars per chunk)",
    )


# ── Row-based chunking (CSV/TSV and other line-oriented tables) ─────────────


def _detect_delimiter(lines: list[str]) -> str | None:
    """Pick the delimiter most lines agree on; None when the text is not tabular."""
    for delim in ("\t", "|", ",", ";"):
        counts = [ln.count(delim) for ln in lines]
        with_delim = [c for c in counts if c >= 1]
        if len(with_delim) < max(3, int(0.6 * len(lines))):
            continue
        common = Counter(with_delim).most_common(1)[0][1]
        if common / len(with_delim) >= 0.5:
            return delim
    return None


def chunk_rows(
    documents: list[tuple[str, str]],
    chunk_size: int = 512,
) -> ChunkedCorpus:
    """Row-based chunking for tabular files.

    Rows are grouped so that header + rows stays under ``chunk_size``
    characters, and every chunk starts with the header row so each is
    self-describing. Non-tabular files fall back to recursive splitting.
    """
    chunks: list[Chunk] = []
    for filename, text in documents:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        delim = _detect_delimiter(lines) if len(lines) >= 3 else None
        if delim is None:
            parts = _recursive_split(text, ["\n\n", "\n", ". ", " "], chunk_size, 0)
            meta = {"tabular": False}
        else:
            header, rows = lines[0], lines[1:]
            parts, current = [], []
            budget = max(chunk_size - len(header) - 1, len(header))
            for row in rows:
                if current and sum(len(r) + 1 for r in current) + len(row) > budget:
                    parts.append("\n".join([header, *current]))
                    current = [row]
                else:
                    current.append(row)
            if current:
                parts.append("\n".join([header, *current]))
            meta = {"tabular": True, "delimiter": delim, "header": header[:200]}
        for i, part in enumerate(parts):
            chunks.append(Chunk(
                text=part,
                source_file=filename,
                chunk_index=i,
                strategy="row_based",
                metadata={"source": filename, "chunk_index": i, **meta},
            ))
    return ChunkedCorpus(
        strategy_name="row_based",
        chunks=chunks,
        description=f"Header + row groups (max {chunk_size} chars per chunk)",
    )
