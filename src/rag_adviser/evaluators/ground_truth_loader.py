"""Ground truth file parser — supports JSONL and CSV formats."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_adviser.models import RagAdvisorError


class GroundTruthError(RagAdvisorError):
    """Failed to load or parse ground truth file."""


@dataclass
class GroundTruthEntry:
    """A single ground truth query with expected results."""

    query: str = ""
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_passages: list[str] = field(default_factory=list)
    expected_answer: str = ""


@dataclass
class GroundTruthSet:
    """Complete ground truth dataset for evaluation."""

    entries: list[GroundTruthEntry] = field(default_factory=list)
    source_path: str = ""

    @property
    def query_count(self) -> int:
        return len(self.entries)

    @property
    def has_passages(self) -> bool:
        return any(e.relevant_passages for e in self.entries)

    @property
    def has_doc_ids(self) -> bool:
        return any(e.relevant_doc_ids for e in self.entries)


class GroundTruthLoader:
    """Load ground truth from JSONL or CSV files.

    JSONL format (one JSON object per line):
        {"query": "...", "relevant_docs": ["doc1.txt", ...], "passages": ["..."], "answer": "..."}

    CSV format:
        query,relevant_docs,passages,answer
        "What is X?","doc1.txt;doc2.txt","passage text","expected answer"

    All fields except 'query' are optional.
    """

    def load(self, path: Path) -> GroundTruthSet:
        """Load ground truth from file, auto-detecting format."""
        path = Path(path)
        if not path.exists():
            raise GroundTruthError(f"Ground truth file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".json", ".ndjson"):
            entries = self._load_jsonl(path)
        elif suffix in (".csv", ".tsv"):
            entries = self._load_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
        else:
            # Try JSONL first, fall back to CSV
            try:
                entries = self._load_jsonl(path)
            except (json.JSONDecodeError, GroundTruthError):
                entries = self._load_csv(path)

        if not entries:
            raise GroundTruthError(f"No valid entries found in {path}")

        return GroundTruthSet(entries=entries, source_path=str(path))

    def _load_jsonl(self, path: Path) -> list[GroundTruthEntry]:
        """Parse JSONL ground truth file."""
        entries: list[GroundTruthEntry] = []
        text = path.read_text(encoding="utf-8")

        for line_num, line in enumerate(text.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise GroundTruthError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

            if not isinstance(obj, dict) or "query" not in obj:
                raise GroundTruthError(
                    f"Line {line_num}: each entry must have a 'query' field"
                )

            entry = GroundTruthEntry(query=str(obj["query"]))

            if "relevant_docs" in obj:
                docs = obj["relevant_docs"]
                entry.relevant_doc_ids = docs if isinstance(docs, list) else [str(docs)]

            if "passages" in obj:
                passages = obj["passages"]
                entry.relevant_passages = (
                    passages if isinstance(passages, list) else [str(passages)]
                )

            if "answer" in obj:
                entry.expected_answer = str(obj["answer"])

            entries.append(entry)

        return entries

    def _load_csv(self, path: Path, delimiter: str = ",") -> list[GroundTruthEntry]:
        """Parse CSV ground truth file."""
        entries: list[GroundTruthEntry] = []

        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            if not reader.fieldnames or "query" not in reader.fieldnames:
                raise GroundTruthError(
                    f"CSV must have a 'query' column header. Found: {reader.fieldnames}"
                )

            for row in reader:
                query = row.get("query", "").strip()
                if not query:
                    continue

                entry = GroundTruthEntry(query=query)

                if "relevant_docs" in row and row["relevant_docs"]:
                    entry.relevant_doc_ids = [
                        d.strip() for d in row["relevant_docs"].split(";") if d.strip()
                    ]

                if "passages" in row and row["passages"]:
                    entry.relevant_passages = [
                        p.strip() for p in row["passages"].split("|||") if p.strip()
                    ]

                if "answer" in row and row["answer"]:
                    entry.expected_answer = row["answer"].strip()

                entries.append(entry)

        return entries
