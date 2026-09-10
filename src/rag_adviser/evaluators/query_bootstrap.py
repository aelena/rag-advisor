"""Bootstrap a synthetic evaluation set from a corpus using an LLM.

Writing evaluation queries is the main barrier to measuring a RAG setup.
This module samples chunks spread evenly across the corpus, asks an LLM to
write one specific question (and short answer) per chunk, and writes a
ground-truth JSONL file the evaluator understands. Every entry is marked
``synthetic: true`` so reports can say that the metrics are indicative, not a
substitute for real user questions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag_adviser.evaluators.chunking_strategies import Chunk, chunk_recursive, load_documents
from rag_adviser.llm.client import LLMClient, LLMError
from rag_adviser.models import RagAdvisorError

logger = logging.getLogger(__name__)


class BootstrapError(RagAdvisorError):
    """Could not generate synthetic queries."""


SYSTEM_PROMPT = """\
You write evaluation questions for a document retrieval system.

Given a passage, write ONE question that a real user of this document
collection might ask, and that can be answered from the passage alone.

Rules:
- Be specific: mention the concrete entity, term, number or name the passage is about.
- Do not refer to "the passage", "the text" or "the document".
- Do not copy a sentence verbatim as the question.
- Vary the form: how / why / what / which / when questions, not only "what is".
- The answer must be short (one sentence) and grounded in the passage.

Respond ONLY with JSON: {"question": "...", "answer": "..."}"""

_MIN_CHUNK_CHARS = 300
_PASSAGE_CHARS = 200


@dataclass
class SyntheticQuery:
    query: str
    relevant_docs: list[str]
    passages: list[str]
    answer: str
    synthetic: bool = True

    def to_json(self) -> str:
        return json.dumps(
            {
                "query": self.query,
                "relevant_docs": self.relevant_docs,
                "passages": self.passages,
                "answer": self.answer,
                "synthetic": True,
            },
            ensure_ascii=False,
        )


@dataclass
class BootstrapResult:
    queries: list[SyntheticQuery] = field(default_factory=list)
    sampled_chunks: int = 0
    failures: int = 0
    output_path: Path | None = None


def sample_chunks(
    corpus_path: Path, n: int, chunk_chars: int = 1200, min_chars: int = _MIN_CHUNK_CHARS
) -> list[Chunk]:
    """Pick up to ``n`` chunks spread evenly across the corpus (by file order)."""
    documents = load_documents(corpus_path)
    if not documents:
        raise BootstrapError(f"No documents found in {corpus_path}")
    chunked = chunk_recursive(documents, chunk_size=chunk_chars, chunk_overlap=0)
    usable = [c for c in chunked.chunks if len(c.text.strip()) >= min_chars]
    if not usable:
        raise BootstrapError("Corpus has no chunks long enough to write questions about")
    if len(usable) <= n:
        return usable
    step = len(usable) / n
    return [usable[int(i * step)] for i in range(n)]


def _first_sentence(text: str, limit: int = _PASSAGE_CHARS) -> str:
    flat = " ".join(text.split())
    m = re.search(r"^(.{10,}?[.!?])(?=\s|$)", flat)
    sentence = m.group(1) if m else flat
    return sentence[:limit]


def _parse_llm_json(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    data = json.loads(text[start : end + 1])
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    if not question:
        raise ValueError("empty question")
    return question, answer


def generate_queries(
    client: LLMClient,
    chunks: list[Chunk],
    progress_cb=None,
) -> BootstrapResult:
    """Ask the LLM for one question per chunk. Failures are counted, not fatal."""
    result = BootstrapResult(sampled_chunks=len(chunks))
    seen: set[str] = set()
    for i, chunk in enumerate(chunks):
        try:
            prompt = f"Passage (from {chunk.source_file}):\n\n{chunk.text}"
            question, answer = _parse_llm_json(client.complete(SYSTEM_PROMPT, prompt))
        except (LLMError, ValueError, json.JSONDecodeError) as e:
            logger.debug(
                "Bootstrap failed for chunk %s#%s: %s", chunk.source_file, chunk.chunk_index, e
            )
            result.failures += 1
            continue
        key = question.lower()
        if key in seen:
            result.failures += 1
            continue
        seen.add(key)
        result.queries.append(
            SyntheticQuery(
                query=question,
                relevant_docs=[chunk.source_file],
                passages=[_first_sentence(chunk.text)],
                answer=answer,
            )
        )
        if progress_cb:
            progress_cb(i + 1, len(chunks))
    return result


def write_jsonl(queries: list[SyntheticQuery], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(q.to_json() + "\n" for q in queries), encoding="utf-8")
    return path
