"""Document corpus analyzer: language detection, stats, content type classification."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import chardet
import tiktoken
from langdetect import DetectorFactory, detect

from rag_adviser.models import ContentType, DocumentAnalysisError, DocumentStats

# Reproducible language detection
DetectorFactory.seed = 0

# CJK Unicode ranges
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
]

# File extensions we can extract text from
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml",
}
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".h",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".r", ".sql", ".sh", ".bash",
}
_TABULAR_EXTENSIONS = {".csv", ".tsv"}
_PDF_EXTENSIONS = {".pdf"}
_DOCX_EXTENSIONS = {".docx"}

_MAX_SAMPLE_DOCS = 50
_MAX_CHARS_PER_DOC = 5000
_MAX_PDF_PAGES = 20
_MAX_READ_BYTES = 4_000_000  # never pull more than this from a single file

# ── Content classification patterns ────────────────────────────────────────
# Each pattern list is scored per document, then normalised per 1,000 chars so
# long and short documents contribute comparably.
_CODE_PATTERNS = [
    r"\bdef\s+\w+\s*\(", r"\bfunction\s+\w+\s*\(", r"\bclass\s+\w+[\s:({]",
    r"^\s*(?:import|from)\s+[\w.]+", r"\b(?:const|let|var)\s+\w+\s*=",
    r"\bif\s*\(.*\)\s*\{", r"^\s*return\b", r"^\s*#include\s*<",
    r"^\s*(?:public|private)\s+\w+", r"=>", r";\s*$",
]
_LEGAL_PATTERNS = [
    r"\bwhereas\b", r"\bherein(?:after)?\b", r"\bthereof\b", r"\bplaintiff\b",
    r"\bdefendant\b", r"\bclause\s+\d+", r"\barticle\s+\d+", r"\bsection\s+\d+(?:\.\d+)*",
    r"\bterms\s+and\s+conditions\b", r"\bpursuant\s+to\b", r"\bindemnif", r"\bshall\s+not\b",
    r"\bnotwithstanding\b", r"\bparty\s+of\s+the\s+(?:first|second)\s+part\b",
]
_CHAT_PATTERNS = [
    r"^\s*[a-z][\w .'-]{0,30}:\s+\S",          # "alice: hello" (text is lower-cased)
    r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s",   # [12:34] timestamps
    r"^\s*(?:user|assistant|human|bot|agent|customer|system)\s*:",
    r"^\s*<(?:user|assistant|human|bot)>",
]
_SCIENTIFIC_PATTERNS = [
    r"\babstract\b", r"\bmethodolog(?:y|ies)\b", r"\bhypothes[ie]s\b", r"\bet\s+al\.",
    r"\bdoi:\s*10\.\d+", r"\barxiv\b", r"\breferences\b", r"\bfig(?:ure)?\.?\s*\d+",
    r"\btable\s+\d+\b", r"\bp\s*[<=]\s*0?\.\d+", r"\bconclusions?\b", r"\bexperiments?\b",
    r"\bin\s+this\s+paper\b", r"\bwe\s+propose\b", r"\bbaselines?\b",
]

# A document counts as tabular when most of its non-empty lines contain
# at least this many delimiter characters.
_TABULAR_MIN_DELIMS = 2
_TABULAR_MIN_LINE_RATIO = 0.6
_TABULAR_MIN_LINES = 3
_TABULAR_MAX_CELL_CHARS = 30


class DocumentAnalyzer:
    """Analyze a document corpus directory for language, content type, and statistics."""

    def __init__(self) -> None:
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def analyze(self, corpus_path: Path) -> DocumentStats:
        """Analyze all documents in the given directory.

        Only a bounded sample of files is opened and tokenized; corpus-wide
        totals (``total_tokens``, ``avg_tokens_per_doc``) are extrapolated from
        that sample to the full file count. ``sampled_files`` records how many
        files were actually inspected.

        Args:
            corpus_path: Path to a directory of documents or a single file.

        Returns:
            DocumentStats with aggregated analysis results.

        Raises:
            DocumentAnalysisError: If the path doesn't exist or no documents found.
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise DocumentAnalysisError(f"Path does not exist: {corpus_path}")

        stats = DocumentStats()
        texts: list[str] = []
        sample_exts: list[str] = []
        token_estimates: list[float] = []   # estimated full-document tokens
        sentence_counts: list[int] = []

        files = list(self._walk_files(corpus_path))
        if not files:
            raise DocumentAnalysisError(f"No supported documents found in: {corpus_path}")

        for file_path in files:
            stats.total_files += 1
            try:
                stats.total_size_bytes += file_path.stat().st_size
            except OSError:
                continue
            ext = file_path.suffix.lower()
            stats.file_types[ext] = stats.file_types.get(ext, 0) + 1

        # Sample evenly across the sorted file list so that one large
        # sub-directory does not dominate the sample.
        for file_path in self._sample(files, _MAX_SAMPLE_DOCS):
            extracted = self._extract_text(file_path)
            if not extracted:
                continue
            full_text, scale = extracted
            if not full_text.strip():
                continue

            text = full_text[:_MAX_CHARS_PER_DOC]
            texts.append(text)
            sample_exts.append(file_path.suffix.lower())

            tokens = len(self._tokenizer.encode(text))
            # Scale the sample token count up to the full extracted text, and
            # further by `scale` when the reader itself truncated the source.
            ratio = len(full_text) / len(text) if text else 1.0
            token_estimates.append(tokens * ratio * scale)

            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
            sentence_counts.append(len(sentences))

        stats.sampled_files = len(texts)

        # Compute stats from sampled texts
        if texts:
            stats.languages_detected = self._detect_languages(texts)
            stats.primary_language = max(
                stats.languages_detected, key=stats.languages_detected.get
            )
            stats.has_cjk = self._check_cjk("".join(texts[:10]))
            stats.detected_content_type, stats.content_type_confidence = (
                self._classify_corpus(texts, sample_exts)
            )
            stats.sample_texts = texts[:3]

        if token_estimates:
            avg = sum(token_estimates) / len(token_estimates)
            stats.avg_tokens_per_doc = avg
            stats.max_tokens = int(max(token_estimates))
            stats.min_tokens = int(min(token_estimates))
            # Extrapolate from the sample to the whole corpus.
            stats.total_tokens = int(avg * stats.total_files)

        if sentence_counts:
            stats.avg_sentences_per_doc = sum(sentence_counts) / len(sentence_counts)

        return stats

    # ── File discovery ─────────────────────────────────────────────────────

    def _walk_files(self, path: Path) -> Iterator[Path]:
        """Yield all supported files recursively, skipping hidden dirs."""
        all_extensions = _TEXT_EXTENSIONS | _CODE_EXTENSIONS | _PDF_EXTENSIONS | _DOCX_EXTENSIONS

        if path.is_file():
            if path.suffix.lower() in all_extensions:
                yield path
            return

        for item in sorted(path.rglob("*")):
            if (
                item.is_file()
                and item.suffix.lower() in all_extensions
                and not any(p.startswith(".") for p in item.relative_to(path).parts)
            ):
                yield item

    @staticmethod
    def _sample(files: list[Path], limit: int) -> list[Path]:
        """Pick up to ``limit`` files spread evenly across the sorted list."""
        if len(files) <= limit:
            return files
        step = len(files) / limit
        return [files[int(i * step)] for i in range(limit)]

    # ── Text extraction ────────────────────────────────────────────────────

    def _extract_text(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text using the appropriate reader.

        Returns ``(text, scale)`` where ``scale`` >= 1.0 is how much larger the
        full document is than the returned text (e.g. a 60-page PDF read with a
        20-page cap yields scale 3.0). ``None`` if the file could not be read.
        """
        ext = file_path.suffix.lower()

        if ext in _TEXT_EXTENSIONS | _CODE_EXTENSIONS:
            return self._read_text_file(file_path)
        elif ext in _PDF_EXTENSIONS:
            return self._read_pdf(file_path)
        elif ext in _DOCX_EXTENSIONS:
            return self._read_docx(file_path)
        return None

    def _read_text_file(self, file_path: Path) -> tuple[str, float] | None:
        """Read a plain text file with encoding detection."""
        try:
            size = file_path.stat().st_size
            with open(file_path, "rb") as f:
                raw = f.read(min(size, _MAX_READ_BYTES))
            detected = chardet.detect(raw[:10_000])
            encoding = detected.get("encoding") or "utf-8"
            text = raw.decode(encoding, errors="replace")
            scale = size / len(raw) if raw and size > len(raw) else 1.0
            return text, scale
        except (OSError, LookupError):
            return None

    def _read_pdf(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text from a PDF file (first ``_MAX_PDF_PAGES`` pages)."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            n_pages = len(reader.pages)
            texts = []
            for page in reader.pages[:_MAX_PDF_PAGES]:
                text = page.extract_text()
                if text:
                    texts.append(text)
            if not texts:
                return None
            scale = n_pages / min(n_pages, _MAX_PDF_PAGES) if n_pages else 1.0
            return "\n".join(texts), scale
        except Exception:
            return None

    def _read_docx(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text from a DOCX file."""
        try:
            from docx import Document

            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return ("\n".join(paragraphs), 1.0) if paragraphs else None
        except Exception:
            return None

    # ── Language ───────────────────────────────────────────────────────────

    def _detect_languages(self, texts: list[str]) -> dict[str, float]:
        """Detect language distribution across document samples."""
        lang_counts: Counter[str] = Counter()

        for text in texts:
            if len(text.strip()) < 20:
                continue
            try:
                lang = detect(text)
                lang_counts[lang] += 1
            except Exception:
                lang_counts["unknown"] += 1

        total = sum(lang_counts.values())
        if total == 0:
            return {"en": 1.0}

        return {
            lang: round(count / total, 3)
            for lang, count in lang_counts.most_common()
        }

    def _check_cjk(self, text: str) -> bool:
        """Check if text contains CJK characters."""
        for ch in text[:10_000]:
            code = ord(ch)
            for start, end in _CJK_RANGES:
                if start <= code <= end:
                    return True
        return False

    # ── Content type ───────────────────────────────────────────────────────

    def _classify_corpus(
        self, texts: list[str], exts: list[str]
    ) -> tuple[ContentType, float]:
        """Classify the corpus by voting over per-document classifications.

        Returns ``(content_type, confidence)`` where confidence is the share of
        sampled documents that agree with the winner. If no type reaches a
        simple majority the corpus is reported as MIXED.
        """
        votes: Counter[ContentType] = Counter()
        for text, ext in zip(texts, exts, strict=True):
            votes[self._classify_document(text, ext)] += 1

        if not votes:
            return ContentType.PROSE, 0.0

        winner, count = votes.most_common(1)[0]
        confidence = count / sum(votes.values())
        if confidence < 0.5 and len(votes) > 1:
            return ContentType.MIXED, confidence
        return winner, confidence

    def _classify_document(self, text: str, ext: str) -> ContentType:
        """Classify a single document using its extension as a strong prior."""
        if ext in _CODE_EXTENSIONS:
            return ContentType.CODE
        if ext in _TABULAR_EXTENSIONS or self._looks_tabular(text):
            return ContentType.TABULAR

        lower = text.lower()
        per_kchars = max(len(text), 1) / 1000.0

        def density(patterns: list[str]) -> float:
            return sum(
                len(re.findall(p, lower, re.MULTILINE)) for p in patterns
            ) / per_kchars

        scores = {
            ContentType.CODE: density(_CODE_PATTERNS) * 0.6,
            ContentType.LEGAL: density(_LEGAL_PATTERNS) * 1.5,
            ContentType.CHAT: density(_CHAT_PATTERNS) * 1.0,
            ContentType.SCIENTIFIC: density(_SCIENTIFIC_PATTERNS) * 1.2,
        }
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        # Require a minimum signal density (matches per 1k chars) before
        # leaving the PROSE default.
        if scores[best] < 1.5:
            return ContentType.PROSE
        return best

    @staticmethod
    def _looks_tabular(text: str) -> bool:
        """True when most non-empty lines share a delimiter structure."""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < _TABULAR_MIN_LINES:
            return False
        for delim in ("\t", "|", ","):
            counts = [ln.count(delim) for ln in lines]
            delimited = [c for c in counts if c >= _TABULAR_MIN_DELIMS]
            if len(delimited) / len(lines) < _TABULAR_MIN_LINE_RATIO:
                continue
            # Real tables have a consistent column count; prose with commas
            # does not.
            common = Counter(delimited).most_common(1)[0][1]
            if common / len(delimited) < 0.5:
                continue
            # Table cells are short; comma-separated prose clauses are not.
            cells = [c for ln in lines for c in ln.split(delim) if c.strip()]
            mean_cell = sum(len(c.strip()) for c in cells) / max(len(cells), 1)
            if mean_cell <= _TABULAR_MAX_CELL_CHARS:
                return True
        return False
