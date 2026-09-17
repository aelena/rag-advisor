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

# File extensions we can extract text from directly (plain-text / code /
# lightweight structured formats).
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

# Rich text formats — first-class documents that need a format-specific
# extractor. Files count as text documents even when the extractor
# library is missing (they just contribute no tokens to the sample and
# get a hint in the report). Bugs from putting these in "other":
# EPUBs and MOBIs are ebooks, DOC/RTF are Word variants, HTML/CHM/DJVU
# are common technical corpora, OPF is EPUB metadata.
_TEXT_RICH_EXTENSIONS = {
    ".epub", ".mobi",
    ".doc", ".rtf",
    ".html", ".htm",
    ".djvu", ".djv",
    ".chm",
    ".opf",
}
_DOC_EXTENSIONS = (
    _TEXT_EXTENSIONS
    | _CODE_EXTENSIONS
    | _PDF_EXTENSIONS
    | _DOCX_EXTENSIONS
    | _TEXT_RICH_EXTENSIONS
)

# Formats we recognise but cannot yet turn into text (binary music
# notation etc.). Counted and reported so the user knows they exist
# and can decide per-format whether to convert them.
_UNSUPPORTED_EXTENSIONS = {".gp", ".ptb"}

# Non-text modalities: counted and reported, never opened.
_MODALITY_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg", ".heic"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"},
    "audio": {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"},
    "spreadsheet": {".xlsx", ".xlsm", ".xls", ".ods", ".numbers"},
    "presentation": {".pptx", ".ppt", ".odp", ".key"},
    "cad": {
        ".dwg", ".dxf", ".dgn", ".ifc", ".rvt", ".step", ".stp", ".iges", ".igs", ".skp", ".3dm",
    },
    "unsupported": _UNSUPPORTED_EXTENSIONS,
}
# Files nobody would want indexed: build artefacts, archives, binaries,
# system junk, ML checkpoints, fonts, installer/disc images.
_IGNORED_EXTENSIONS = {
    # Compiled binaries
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe",
    # Archives
    ".zip", ".tar", ".gz", ".7z", ".rar",
    # Build / runtime artefacts
    ".lock", ".log", ".tmp", ".bak",
    # OS / editor metadata
    ".ds_store", ".ini", ".cfg", ".toml", ".env",
    # System / temp — download partials, shortcuts, installers, DB files
    ".crdownload", ".lnk", ".msi", ".db",
    # Generic binary payloads
    ".bin", ".dat",
    # Disc images
    ".iso", ".img", ".dmg",
    # Serialised objects / ML checkpoints
    ".pkl", ".pickle", ".npy", ".npz", ".pt", ".pth", ".safetensors", ".ckpt",
    # Fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}

# Anything that survives the taxonomy above but has a "suffix" that is
# clearly not an extension. Windows filenames like
# ``Foo & Bar - Machine Learning-Springer (2 files)``
# report a suffix of ``. Springer (2 files)`` — treating that as an
# extension leaks into the file_types map and creates false modalities.
_MAX_EXTENSION_LEN = 6  # length not counting the leading dot


def _looks_like_filename_fragment(ext: str) -> bool:
    """Heuristic: is ``ext`` really an extension, or a filename fragment?"""
    if not ext or not ext.startswith("."):
        return False
    body = ext[1:]
    return bool(
        " " in body
        or len(body) > _MAX_EXTENSION_LEN
        or ext.startswith("..")
        or not body.isascii()
    )

# Sampling: opening every file in a 5000-file corpus is wasteful, but
# the pre-0.5.0 fixed cap of 50 gave misleading confidence intervals on
# any corpus larger than a few hundred files. We now sample 5% of the
# document count with a floor of 50 (small corpora still get inspected)
# and a ceiling of 500 (large corpora stay fast).
_MIN_SAMPLE_DOCS = 50
_MAX_SAMPLE_DOCS = 500
_SAMPLE_SHARE = 0.05

_MAX_CHARS_PER_DOC = 5000
_MAX_PDF_PAGES = 20
_MAX_READ_BYTES = 4_000_000  # never pull more than this from a single file


def _target_sample_size(n_files: int) -> int:
    """Number of files to sample from a corpus of ``n_files`` documents."""
    if n_files <= _MIN_SAMPLE_DOCS:
        return n_files
    return max(_MIN_SAMPLE_DOCS, min(_MAX_SAMPLE_DOCS, int(n_files * _SAMPLE_SHARE)))


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on a pre-sorted list. ``q`` in ``[0, 1]``."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion.

    Preferred over the normal approximation for small samples with rates
    near 0 or 1 — which is exactly the scanned-PDF regime (5 of 50).
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    spread = (z / denom) * ((p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - spread), min(1.0, centre + spread))

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

        all_files = list(self._walk_all(corpus_path))
        if not all_files:
            raise DocumentAnalysisError(f"No supported documents found in: {corpus_path}")

        files: list[Path] = []
        for file_path in all_files:
            ext = file_path.suffix.lower()
            if _looks_like_filename_fragment(ext):
                # Record the offending path once (capped so a corpus of
                # 500 broken filenames does not blow up the report).
                if len(stats.filename_fragment_paths) < 25:
                    stats.filename_fragment_paths.append(str(file_path))
                continue
            modality = self._modality_of(ext)
            if modality is None:
                continue  # build artefacts, archives, system junk, binaries
            stats.total_files_all += 1
            stats.modalities[modality] = stats.modalities.get(modality, 0) + 1
            stats.file_types[ext] = stats.file_types.get(ext, 0) + 1
            if modality == "document":
                files.append(file_path)
                stats.total_files += 1
                try:
                    stats.total_size_bytes += file_path.stat().st_size
                except OSError:
                    continue

        if not files and stats.total_files_all == 0:
            raise DocumentAnalysisError(f"No supported documents found in: {corpus_path}")

        # Sample stratified by extension so a corpus of 5000 PDFs and
        # 800 EPUBs contributes proportionally to the token estimate,
        # rather than the sample being 99% whichever extension sorts
        # first. Fixed 50-file caps hid this on heterogeneous corpora.
        sample_target = _target_sample_size(len(files))
        for file_path in self._stratified_sample(files, sample_target):
            is_pdf = file_path.suffix.lower() in _PDF_EXTENSIONS
            if is_pdf:
                stats.sampled_pdfs += 1
            extracted = self._extract_text(file_path)
            if not extracted or not extracted[0].strip():
                if is_pdf:
                    stats.scanned_pdfs += 1  # no text layer: needs OCR
                continue
            full_text, scale = extracted

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
            # Distributional stats — corpora are heavy-tailed and the
            # mean alone lies. The report can now say "median 800
            # tokens, p99 60,000" and the reader spots the tail.
            sorted_tokens = sorted(token_estimates)
            stats.tokens_p50 = int(_percentile(sorted_tokens, 0.50))
            stats.tokens_p75 = int(_percentile(sorted_tokens, 0.75))
            stats.tokens_p90 = int(_percentile(sorted_tokens, 0.90))
            stats.tokens_p95 = int(_percentile(sorted_tokens, 0.95))
            stats.tokens_p99 = int(_percentile(sorted_tokens, 0.99))
            # Extrapolate from the sample to the whole corpus.
            stats.total_tokens = int(avg * stats.total_files)

        # Wilson 95% CI on the scanned-PDF rate: a fixed "5 of 50 -> 10%"
        # figure hides that the true rate could plausibly be anywhere
        # between ~3% and ~22%. The interval belongs alongside the point
        # estimate whenever the sample is small.
        stats.scanned_pdf_rate_ci = _wilson_interval(stats.scanned_pdfs, stats.sampled_pdfs)

        if sentence_counts:
            stats.avg_sentences_per_doc = sum(sentence_counts) / len(sentence_counts)

        return stats

    # ── File discovery ─────────────────────────────────────────────────────

    def _walk_all(self, path: Path) -> Iterator[Path]:
        """Yield every file recursively, skipping hidden dirs and files."""
        if path.is_file():
            yield path
            return

        for item in sorted(path.rglob("*")):
            if item.is_file() and not any(
                p.startswith(".") for p in item.relative_to(path).parts
            ):
                yield item

    def _walk_files(self, path: Path) -> Iterator[Path]:
        """Yield text-extractable documents only (kept for callers that need just those)."""
        for item in self._walk_all(path):
            if item.suffix.lower() in _DOC_EXTENSIONS:
                yield item

    @staticmethod
    def _modality_of(ext: str) -> str | None:
        """Classify an extension: 'document', a modality bucket, 'other', or None to skip.

        Returns ``None`` when the file is a build artefact, system/temp
        file, binary payload, or a filename fragment mistaken for an
        extension — in every case the file is excluded from token
        counts and from the file_types map entirely.
        """
        if _looks_like_filename_fragment(ext):
            return None
        if ext in _DOC_EXTENSIONS:
            return "document"
        for modality, exts in _MODALITY_EXTENSIONS.items():
            if ext in exts:
                return modality
        if ext in _IGNORED_EXTENSIONS or not ext:
            return None
        return "other"

    @staticmethod
    def _sample(files: list[Path], limit: int) -> list[Path]:
        """Pick up to ``limit`` files spread evenly across the sorted list."""
        if len(files) <= limit:
            return files
        step = len(files) / limit
        return [files[int(i * step)] for i in range(limit)]

    @classmethod
    def _stratified_sample(cls, files: list[Path], target: int) -> list[Path]:
        """Sample ``target`` files stratified proportionally by extension.

        Groups files by their (lower-cased) suffix, then draws from each
        group in proportion to that group's share of the corpus. Within
        each group we pick evenly-spaced items so directory layout does
        not bias the pick. Small groups always contribute at least one
        file, which keeps rare formats visible in the token estimate.
        """
        if len(files) <= target:
            return files
        buckets: dict[str, list[Path]] = {}
        for f in files:
            buckets.setdefault(f.suffix.lower(), []).append(f)
        total = len(files)
        chosen: list[Path] = []
        # First pass: allocate integer quotas proportional to share.
        allocations: dict[str, int] = {}
        for ext, group in buckets.items():
            share = len(group) / total
            allocations[ext] = max(1, int(share * target))
        # Trim / grow the total allocation to match ``target``.
        overflow = sum(allocations.values()) - target
        if overflow > 0:
            # Reduce the largest allocations first, but never below 1.
            for ext in sorted(allocations, key=lambda e: -allocations[e]):
                if overflow == 0:
                    break
                room = allocations[ext] - 1
                if room > 0:
                    trim = min(room, overflow)
                    allocations[ext] -= trim
                    overflow -= trim
        elif overflow < 0:
            # Give the extra picks to the largest groups.
            for ext in sorted(allocations, key=lambda e: -len(buckets[e])):
                if overflow == 0:
                    break
                headroom = len(buckets[ext]) - allocations[ext]
                if headroom > 0:
                    add = min(headroom, -overflow)
                    allocations[ext] += add
                    overflow += add
        for ext, group in buckets.items():
            chosen.extend(cls._sample(group, allocations[ext]))
        return chosen

    # ── Text extraction ────────────────────────────────────────────────────

    def _extract_text(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text using the appropriate reader.

        Returns ``(text, scale)`` where ``scale`` >= 1.0 is how much larger the
        full document is than the returned text (e.g. a 60-page PDF read with a
        20-page cap yields scale 3.0). ``None`` if the file could not be read
        — including because an optional extractor library is missing. Callers
        interpret ``None`` as "file counted but not sampled for tokens", so a
        missing library degrades the token estimate rather than crashing.
        """
        ext = file_path.suffix.lower()

        if ext in _TEXT_EXTENSIONS | _CODE_EXTENSIONS:
            return self._read_text_file(file_path)
        if ext in _PDF_EXTENSIONS:
            return self._read_pdf(file_path)
        if ext in _DOCX_EXTENSIONS:
            return self._read_docx(file_path)
        if ext == ".epub":
            return self._read_epub(file_path)
        if ext in {".html", ".htm"}:
            return self._read_html(file_path)
        if ext == ".rtf":
            return self._read_rtf(file_path)
        if ext == ".opf":
            # OPF is EPUB metadata — plain XML with textual descriptions.
            return self._read_text_file(file_path)
        # .doc / .mobi / .djvu / .djv / .chm need external tools
        # (LibreOffice, calibre, DjVuLibre, chmlib). We count them as
        # documents but don't sample tokens; the token estimate is
        # extrapolated from the formats we did open.
        return None

    def _read_epub(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text from an EPUB via ``ebooklib`` when available."""
        try:
            from ebooklib import ITEM_DOCUMENT, epub
        except ImportError:
            return None
        try:
            book = epub.read_epub(str(file_path))
        except Exception:
            return None
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            BeautifulSoup = None  # noqa: N806 — aliasing the class, not a variable
        chunks: list[str] = []
        total_chars = 0
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            try:
                content = item.get_content()
            except Exception:
                continue
            if BeautifulSoup is not None:
                try:
                    text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
                except Exception:
                    text = ""
            else:
                # Cheap fallback: strip HTML tags with a regex.
                text = re.sub(r"<[^>]+>", " ", content.decode("utf-8", errors="replace"))
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            chunks.append(text)
            total_chars += len(text)
            if total_chars >= _MAX_READ_BYTES:
                break
        return ("\n\n".join(chunks), 1.0) if chunks else None

    def _read_html(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text from a static HTML file."""
        raw = self._read_text_file(file_path)
        if raw is None:
            return None
        html, scale = raw
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return (text, scale) if text else None

    def _read_rtf(self, file_path: Path) -> tuple[str, float] | None:
        """Extract text from an RTF file via ``striprtf`` when available."""
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            return None
        raw = self._read_text_file(file_path)
        if raw is None:
            return None
        try:
            text = rtf_to_text(raw[0])
        except Exception:
            return None
        return (text.strip(), raw[1]) if text.strip() else None

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
