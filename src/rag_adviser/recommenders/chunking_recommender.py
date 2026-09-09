"""Content-type and language-aware chunking strategy recommender."""

from __future__ import annotations

from rag_adviser.config import load_defaults
from rag_adviser.models import ChunkingRecommendation, ContentType, UseCase

# Base configurations by content type
_BASE_CONFIGS: dict[ContentType, dict] = {
    ContentType.PROSE: {
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "count_by": "tokens",
    },
    ContentType.CODE: {
        "strategy": "language_aware",
        "chunk_size": 256,
        "chunk_overlap": 25,
        "count_by": "tokens",
    },
    ContentType.LEGAL: {
        "strategy": "hierarchical",
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "count_by": "tokens",
    },
    ContentType.CHAT: {
        "strategy": "speaker_split",
        "chunk_size": 300,
        "chunk_overlap": 50,
        "count_by": "tokens",
    },
    ContentType.TABULAR: {
        "strategy": "row_based",
        "chunk_size": 400,
        "chunk_overlap": 0,
        "count_by": "tokens",
    },
    ContentType.SCIENTIFIC: {
        "strategy": "recursive",
        "chunk_size": 800,
        "chunk_overlap": 80,
        "count_by": "tokens",
    },
    ContentType.MIXED: {
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "count_by": "tokens",
    },
}


class ChunkingRecommender:
    """Recommend chunking strategy based on content type, languages, and model limits."""

    def __init__(self) -> None:
        defaults = load_defaults()
        self._cjk_languages: list[str] = defaults.get("cjk_languages", ["zh", "ja", "ko"])
        self._cjk_multiplier: float = defaults.get("cjk_chunk_size_multiplier", 0.7)
        self._lang_tokenizers: dict[str, str] = defaults.get("lang_tokenizers", {})
        self._spacy_models: dict[str, str] = defaults.get("spacy_models", {})

    def recommend(
        self,
        content_type: ContentType | None = None,
        languages: list[str] | None = None,
        embedding_max_tokens: int = 512,
        use_case: UseCase | None = None,
    ) -> ChunkingRecommendation:
        """Generate a chunking recommendation.

        Args:
            content_type: Detected or overridden content type.
            languages: Detected languages in the corpus.
            embedding_max_tokens: Max token limit of the chosen embedding model.
            use_case: Primary use case (affects chunk size tuning).

        Returns:
            ChunkingRecommendation with strategy, sizes, and language overrides.
        """
        ct = content_type or ContentType.PROSE
        langs = languages or ["en"]

        # Start from base config
        base = dict(_BASE_CONFIGS.get(ct, _BASE_CONFIGS[ContentType.PROSE]))
        notes: list[str] = []

        # Adjust for embedding model token limit
        if embedding_max_tokens < base["chunk_size"]:
            base["chunk_size"] = int(embedding_max_tokens * 0.8)
            notes.append(
                f"Chunk size reduced to fit embedding model limit ({embedding_max_tokens} tokens)"
            )

        # Use case adjustments
        if use_case == UseCase.QA:
            base["chunk_size"] = min(base["chunk_size"], 512)
            notes.append("Smaller chunks preferred for precise Q&A retrieval")
        elif use_case == UseCase.SUMMARIZATION:
            base["chunk_size"] = int(base["chunk_size"] * 1.5)
            base["chunk_overlap"] = int(base["chunk_overlap"] * 1.5)
            notes.append("Larger chunks preserve context for summarization")
        elif use_case == UseCase.CODE:
            notes.append("Code-aware splitting preserves function/class boundaries")
        elif use_case == UseCase.LEGAL:
            notes.append("Hierarchical chunking preserves section/clause structure")

        # Language-specific overrides
        language_overrides: dict[str, dict] = {}
        for lang in langs:
            lang_code = lang.lower().split("-")[0]  # Normalize zh-cn -> zh

            if lang_code in self._cjk_languages:
                # CJK: use characters, not tokens
                language_overrides[lang_code] = {
                    "chunk_size": int(base["chunk_size"] * self._cjk_multiplier),
                    "overlap": int(base["chunk_overlap"] * 0.8),
                    "count_by": "characters",
                    "tokenizer": self._lang_tokenizers.get(lang_code, "jieba"),
                }
                notes.append(
                    f"CJK language ({lang_code}): chunk by characters with "
                    f"{self._cjk_multiplier:.0%} size reduction"
                )
            elif lang_code in self._spacy_models:
                language_overrides[lang_code] = {
                    "sentence_model": self._spacy_models[lang_code],
                    "tokenizer": "spacy",
                }

        # Generate code snippet
        code_snippet = self._generate_code_snippet(base, language_overrides)

        return ChunkingRecommendation(
            strategy=base["strategy"],
            chunk_size=base["chunk_size"],
            chunk_overlap=base["chunk_overlap"],
            count_by=base["count_by"],
            content_type=ct.value,
            language_overrides=language_overrides,
            notes=notes,
            code_snippet=code_snippet,
        )

    def _generate_code_snippet(
        self, config: dict, language_overrides: dict[str, dict]
    ) -> str:
        """Generate a ready-to-use Python code snippet for the recommended strategy."""
        strategy = config["strategy"]
        chunk_size = config["chunk_size"]
        overlap = config["chunk_overlap"]

        # Sizes are expressed in tokens, so use the tiktoken-aware constructor;
        # the plain constructor counts characters.
        if strategy == "recursive":
            snippet = f"""# pip install langchain-text-splitters tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size={chunk_size},      # tokens
    chunk_overlap={overlap},   # tokens
    separators=["\\n\\n", "\\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)"""

        elif strategy == "language_aware":
            snippet = f"""# pip install langchain-text-splitters
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # Adjust for your code language
    chunk_size={chunk_size * 4},  # characters (~{chunk_size} tokens)
    chunk_overlap={overlap * 4},
)
chunks = splitter.split_documents(documents)"""

        elif strategy == "hierarchical":
            snippet = f"""# pip install langchain-text-splitters tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Large parent chunks for context
parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size={chunk_size},
    chunk_overlap={overlap},
    separators=["\\nArticle ", "\\nSection ", "\\n\\n", "\\n"],
)

# Smaller child chunks for precise retrieval (index children, return parents)
child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size={chunk_size // 3},
    chunk_overlap={overlap // 3},
)"""

        elif strategy == "speaker_split":
            snippet = f"""# For chat/transcript data, split by speaker turns
import re

def split_by_speaker(text: str, max_chunk_size: int = {chunk_size}) -> list[str]:
    turns = re.split(r'(?=\\n\\w+:)', text)
    chunks = []
    current = ""
    for turn in turns:
        if len(current) + len(turn) > max_chunk_size and current:
            chunks.append(current.strip())
            current = turn
        else:
            current += turn
    if current.strip():
        chunks.append(current.strip())
    return chunks"""

        elif strategy == "row_based":
            snippet = f"""# For tabular data, keep whole rows together and prepend the header
import csv

def chunk_rows(path: str, rows_per_chunk: int = 20) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    chunks = []
    for i in range(0, len(rows), rows_per_chunk):
        block = rows[i : i + rows_per_chunk]
        lines = [", ".join(header)] + [", ".join(r) for r in block]
        chunks.append("\\n".join(lines))
    return chunks  # target ~{chunk_size} tokens per chunk; tune rows_per_chunk"""

        else:
            snippet = f"""# pip install langchain-text-splitters tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size={chunk_size},
    chunk_overlap={overlap},
)
chunks = splitter.split_documents(documents)"""

        # Add CJK-specific note if applicable
        if language_overrides:
            cjk_langs = [
                lang for lang, cfg in language_overrides.items()
                if cfg.get("count_by") == "characters"
            ]
            if cjk_langs:
                tokenizers = [
                    language_overrides[lang].get("tokenizer", "jieba")
                    for lang in cjk_langs
                ]
                snippet += f"\n\n# CJK languages ({', '.join(cjk_langs)}): install tokenizers"
                snippet += f"\n# pip install {' '.join(set(tokenizers))}"

        return snippet
