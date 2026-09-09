# RAG Advisor

**Rule-based recommendations for building Retrieval-Augmented Generation systems.**

RAG Advisor is a CLI tool that analyzes your document corpus, collects your infrastructure constraints, and generates a complete RAG configuration — embedding model, chunking strategy, vector database, and retrieval settings — tailored to your exact requirements. No guesswork, no trial-and-error.

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Profile Presets](#profile-presets)
- [LLM Verification](#llm-verification)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Research Assistant](#research-assistant)
- [CLI Reference](#cli-reference)
- [Document Analysis](#document-analysis)
- [External API Integration](#external-api-integration)
- [Recommendation Engine](#recommendation-engine)
- [Output & Reports](#output--reports)
- [Architecture](#architecture)
- [Configuration](#configuration)

## How It Works

RAG Advisor runs an 11-step pipeline:

```
 Input Collection ──► Document Analysis ──► Constraint Validation
         │
         ▼
 Approach Assessment ──► HuggingFace API Query ──► Model Scoring
         │
         ▼
 Chunking Strategy ──► Vector DB Selection ──► Retrieval Settings
         │
         ▼
 Query Pipeline ──► LLM Verification (optional) ──► Report Generation
```

1. **Collect user inputs** — corpus path, use case, deployment constraints, hardware limits, privacy requirements
2. **Analyze documents** — detect languages, classify content types, count tokens, sample texts
3. **Assess approach** — determine if RAG is the right approach; suggest alternatives if not
4. **Validate constraints** — flag conflicts (e.g., air-gapped + paid API, edge deployment + large models)
5. **Find embedding models** — query the HuggingFace Hub API for models from trusted organizations, score and rank them
6. **Recommend chunking** — select a strategy (recursive, language-aware, code-based, speaker-split, row-based) with appropriate sizes
7. **Select vector database** — score 8 database profiles against your scale, deployment, and privacy requirements
8. **Configure retrieval** — set top-K, reranking, similarity threshold, and prompt strategy based on use case
9. **Design query pipeline** — recommend query transformation techniques (HyDE, expansion, multi-query, condensation)
10. **LLM verification** (optional) — send recommendations to an LLM for expert review
11. **Generate reports** — produce professional reports in up to 3 formats with code snippets and implementation checklists

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

### Optional dependencies

```bash
# Evaluation pipeline (chunking + embedding + vector search)
pip install -e ".[eval]"          # ChromaDB backend
pip install -e ".[eval-faiss]"    # FAISS backend
pip install -e ".[eval-pgvector]" # pgvector backend
pip install -e ".[eval-sqlite]"   # SQLite + sqlite-vec backend

# Language-aware sentence splitting (spaCy)
pip install -e ".[spacy]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Interactive mode (default)

```bash
ragadvisor run
```

Launches a guided 13-step questionnaire with a Rich terminal UI, organized into 3 phases:

- **Phase 1** — Document discovery (corpus path, languages, content type)
- **Phase 2** — Use case & constraints (deployment, hardware, privacy, budget)
- **Approach Gate** — Checks if RAG is the right approach before continuing
- **Phase 3** — Query & operational patterns (query type, complexity, answer type, update frequency)

At the end, you can optionally save your answers as a custom preset for future use.

### Non-interactive mode

```bash
ragadvisor run --no-interactive \
  -d ./my_documents \
  -u question_answering \
  -e local \
  --hardware cpu_only \
  --ram-gb 8 \
  --privacy strict \
  --budget free \
  -f all \
  -o ./my_report
```

### Using a preset

```bash
ragadvisor run --preset legal-discovery
ragadvisor run --preset customer-support --no-interactive
```

### XML input mode

```bash
ragadvisor run --from-xml answers.xml
```

### Quick document analysis only

```bash
ragadvisor analyze ./my_documents
```

## Profile Presets

Presets are pre-configured profiles for common RAG scenarios. They pre-fill the questionnaire so you can get started quickly.

### Built-in presets

| Preset | Description |
|--------|-------------|
| `legal-discovery` | E-discovery and legal document analysis — strict privacy, precise retrieval |
| `customer-support` | Customer support chatbot — fast responses, multi-turn conversations |
| `code-assistant` | Code documentation search — precise, low-latency |
| `internal-wiki` | Internal knowledge base / wiki — moderate scale, mixed content |
| `research-papers` | Academic paper analysis — multi-hop queries, multilingual |
| `air-gapped` | Fully offline deployment — zero network access |

### List available presets

```bash
ragadvisor presets
```

### Create custom presets

At the end of the interactive questionnaire, you'll be asked if you want to save your answers as a custom preset. Custom presets are stored in `~/.ragadvisor/presets/` as YAML or JSON files.

## LLM Verification

The `--use-llm` flag sends your rule-based recommendations to an LLM for expert review. The LLM assesses each recommendation, notes agreements, suggests refinements, and flags additional considerations.

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# Run with LLM verification
ragadvisor run --use-llm
```

**Supported providers:** Anthropic (preferred) and OpenAI. The tool auto-detects which API key is available.

**Environment variables:**
- `ANTHROPIC_API_KEY` — Anthropic API key (checked first)
- `OPENAI_API_KEY` — OpenAI API key (fallback)
- `RAGADVISOR_LLM_MODEL` — Override the default model (defaults: `claude-opus-5` for Anthropic, `gpt-4o-mini` for OpenAI)
- `OPENAI_BASE_URL` — Custom OpenAI-compatible endpoint (Ollama, vLLM, Azure, ...)
- `ANTHROPIC_BASE_URL` — Custom Anthropic-compatible endpoint (proxies, gateways)

The LLM verification section appears in all report formats (Markdown, HTML, YAML) and in the terminal summary.

## Evaluation Pipeline

Compare chunking strategies against ground truth data with real embeddings and vector search.

```bash
# Basic evaluation
ragadvisor evaluate ./docs ./queries.jsonl

# Compare specific strategies
ragadvisor evaluate ./docs ./queries.jsonl --strategy recursive --strategy semantic

# Use FAISS backend
ragadvisor evaluate ./docs ./queries.jsonl --backend faiss

# Use pgvector
ragadvisor evaluate ./docs ./queries.jsonl --backend pgvector --db-connection "postgresql://localhost/mydb"
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Hit Rate@k** | Fraction of queries with at least one relevant result in top-k |
| **MRR** | Mean Reciprocal Rank — how high is the first relevant result |
| **Precision@k** | Fraction of top-k results that are relevant |
| **Recall@k** | Fraction of all relevant documents found in top-k |
| **NDCG@k** | Normalized Discounted Cumulative Gain — position-weighted relevance |

### Chunking strategies

- **recursive** — Recursive character splitting with configurable size/overlap
- **semantic** — Embedding-based boundary detection (splits at topic changes)
- **hierarchical** — Parent-child chunks (index children, return parents)
- **adaptive** — Per-file strategy selection (detects code/markdown/prose)

### Vector backends

- **chroma** — ChromaDB in-memory (zero setup)
- **faiss** — FAISS IndexFlatIP (fastest similarity search)
- **pgvector** — PostgreSQL + pgvector (production-realistic)
- **sqlite** — SQLite + sqlite-vec (lightweight, file-based)

### CI/Regression Mode

Save evaluation results as a baseline and compare future runs against it. Exits with code 1 if any metric regresses beyond the threshold — perfect for CI pipelines.

```bash
# Save a baseline
ragadvisor evaluate ./docs ./queries.jsonl --save-baseline baseline.json

# Compare against baseline (CI mode)
ragadvisor evaluate ./docs ./queries.jsonl --baseline baseline.json

# Custom regression threshold (default: 2%)
ragadvisor evaluate ./docs ./queries.jsonl --baseline baseline.json --regression-threshold 0.05
```

The comparison displays a Rich diff table showing baseline vs. current metrics, with regressions highlighted in red and improvements in green.

## Research Assistant

Ask questions about RAG configuration using a curated knowledge base of 14 research papers.

```bash
# Interactive mode
ragadvisor ask

# Single question
ragadvisor ask "What chunking strategy should I use for legal documents?"

# With LLM synthesis (requires API key)
ragadvisor ask --use-llm "How does HyDE work?"
```

### Retrieval-only mode (default)

Returns the most relevant passages from the knowledge base. No LLM involved — no hallucination possible.

### LLM synthesis mode (`--use-llm`)

Retrieves passages and sends them to an LLM to synthesize a coherent answer with citations. Source passages are always shown for transparency.

### Knowledge base

14 curated papers covering:
- RAG foundations (Lewis et al. 2020)
- Comprehensive RAG survey (Gao et al. 2024)
- Chunking strategies and practical analysis
- Embedding models (E5, BGE, BGE-M3)
- Vector database selection
- Query transformation techniques (HyDE, expansion, multi-query)
- Self-reflective RAG (Self-RAG, REPLUG, CRAG)
- RAG vs. long context trade-offs
- Evaluation (RAGAS, RGB benchmark)
- Advanced RAG patterns and best practices

## CLI Reference

### `ragadvisor run`

| Flag | Short | Description |
|------|-------|-------------|
| `--preset NAME` | | Use a built-in or custom preset |
| `--from-xml PATH` | `-x` | Load answers from an XML file |
| `--interactive / --no-interactive` | | Toggle interactive mode (default: on) |
| `--document-path PATH` | `-d` | Path to document corpus |
| `--content-type TYPE` | `-t` | Override: `prose`, `code`, `legal`, `chat`, `tabular` |
| `--use-case CASE` | `-u` | `question_answering`, `semantic_search`, `summarization`, `code_assistance`, `legal_analysis` |
| `--environment ENV` | `-e` | `cloud`, `on_prem`, `edge`, `local` |
| `--latency BUDGET` | | `<500ms`, `<2s`, `batch` |
| `--hardware PROFILE` | | `cpu_only`, `gpu_available`, `limited_ram` |
| `--ram-gb N` | | Available RAM in GB |
| `--vram-gb N` | | Available VRAM in GB |
| `--budget TIER` | | `free`, `paid_api`, `self_hosted` |
| `--privacy LEVEL` | | `none`, `moderate`, `strict`, `air_gapped` |
| `--embedding-provider NAME` | | Preferred embedding provider |
| `--llm-provider NAME` | | Preferred LLM provider (or `none`) |
| `--preferred-lib NAME` | | Preferred libraries (repeatable) |
| `--query-type TYPE` | | `short_keywords`, `natural_questions`, `multi_turn` |
| `--query-complexity TYPE` | | `simple_factual`, `comparative`, `aggregative`, `multi_hop` |
| `--answer-type TYPE` | | `exact_passage`, `synthesized`, `yes_no_with_evidence`, `list_enumeration` |
| `--sample-query TEXT` | | Sample queries for tuning (repeatable) |
| `--update-frequency FREQ` | | `never`, `weekly`, `daily`, `realtime` |
| `--ground-truth / --no-ground-truth` | | Have evaluation data? |
| `--format FORMAT` | `-f` | `markdown`, `html`, `yaml`, `all` (default: `all`) |
| `--output DIR` | `-o` | Output directory (default: `./rag_report`) |
| `--use-llm / --no-llm` | | Send recommendations to an LLM for verification |

### `ragadvisor evaluate CORPUS GROUND_TRUTH`

| Flag | Short | Description |
|------|-------|-------------|
| `--model NAME` | `-m` | Embedding model (default: `all-MiniLM-L6-v2`) |
| `--strategy NAME` | `-s` | Chunking strategies to compare (repeatable) |
| `--chunk-size N` | | Base chunk size in characters (default: 512) |
| `--chunk-overlap N` | | Chunk overlap in characters (default: 50) |
| `--top-k N` | `-k` | Results to retrieve per query (default: 5) |
| `--backend NAME` | `-b` | Vector backend: `chroma`, `faiss`, `pgvector`, `sqlite` |
| `--db-connection STR` | | Connection string for pgvector or sqlite |
| `--baseline PATH` | | Compare against baseline JSON (CI mode) |
| `--save-baseline PATH` | | Save current results as baseline JSON |
| `--regression-threshold N` | | Metric drop threshold for regression (default: 0.02) |
| `--output DIR` | `-o` | Output directory (default: `./eval_results`) |

### `ragadvisor ask [QUERY]`

| Flag | Short | Description |
|------|-------|-------------|
| `--papers-dir PATH` | `-p` | Custom research papers directory |
| `--model NAME` | `-m` | Embedding model for retrieval |
| `--top-k N` | `-k` | Number of results to retrieve (default: 5) |
| `--use-llm / --no-llm` | | Synthesize answers with an LLM |

### `ragadvisor presets`

Lists all available built-in and custom presets.

### `ragadvisor analyze PATH`

Runs document analysis only and displays results as a Rich table.

### `ragadvisor version`

Prints the current version.

## Document Analysis

The `DocumentAnalyzer` examines your corpus using **rule-based heuristics** — no LLMs are involved in the analysis step. It performs:

### Language Detection
Uses `langdetect` to identify languages across all documents. Detects CJK content (Chinese, Japanese, Korean) which triggers specialized chunking strategies and multilingual model selection.

### Content Type Classification
Each sampled document is classified individually, then the corpus type is decided by majority vote. The file extension is a strong prior (`.py`/`.js`/... → code, `.csv`/`.tsv` → tabular); everything else is scored by pattern density per 1,000 characters so long and short documents count equally.

| Type | Detection |
|------|-----------|
| **Code** | Code extensions, or `def`/`function`/`class`/`import`/`=>`/`;` density |
| **Legal** | `whereas`, `plaintiff`, `pursuant to`, `section N`, `article N`, `indemnif...` |
| **Chat** | Speaker-turn lines (`alice: ...`, `user:`), `[HH:MM]` timestamps |
| **Tabular** | `.csv`/`.tsv`, or most lines sharing a consistent delimiter count with short cells |
| **Scientific** | `abstract`, `et al.`, `doi:`, `arxiv`, `fig. N`, `p < 0.05`, `we propose` |
| **Prose** | Default when no signal reaches the threshold |
| **Mixed** | No single type wins a majority of sampled documents |

The vote share is reported as `content_type_confidence`; below 60% the report carries a warning suggesting `--content-type` to override.

### Token Counting
Uses `tiktoken` with the `cl100k_base` encoding. Only a sample of documents is opened (50 files, spread evenly across the corpus, first 5,000 characters each). Per-document token counts are scaled to the full document length and the corpus total is extrapolated to the full file count, so `total_tokens` is an **estimate**. The report says how many files were actually sampled. These statistics drive the approach gate (RAG vs. direct context), chunk sizing and vector DB capacity planning.

### Supported File Formats

- **Text**: `.txt`, `.md`, `.rst`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.xml`, `.yaml`, `.yml`
- **Code**: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.go`, `.rs`, `.cpp`, `.c`, `.h`, `.cs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.r`, `.sql`, `.sh`, `.bash`
- **Documents**: `.pdf` (via `pypdf`), `.docx` (via `python-docx`)

### Character Encoding
Automatic encoding detection via `chardet` ensures files in UTF-8, Latin-1, Shift-JIS, and other encodings are read correctly.

## External API Integration

### HuggingFace Hub API

The only external API call in the core recommendation pipeline is to the **HuggingFace Hub** for embedding model discovery.

**Endpoint**: `huggingface_hub.list_models()` (Python SDK)

**What it queries**:
- Models from 8 trusted organizations: `BAAI`, `intfloat`, `sentence-transformers`, `thenlper`, `mixedbread-ai`, `Alibaba-NLP`, `nomic-ai`, `jinaai`
- Filtered by the `sentence-transformers` tag
- Sorted by community likes (descending)
- Limited to 10 models per organization

Hub queries run concurrently under a 20-second overall budget; a slow or unreachable Hub yields partial results instead of hanging the CLI. Set `HF_HUB_OFFLINE=1` to skip the Hub entirely.

**Offline / air-gapped fallback**: When privacy is set to `strict` or `air_gapped`, or when the API is unreachable, the tool falls back to a **curated list of 14 open-weight models** defined in `src/rag_adviser/config/defaults.yaml`. Each carries an approximate MTEB retrieval quality score, license, dimension and context length.

**Hosted embedding APIs**: When `--budget paid_api` and privacy is `none` or `moderate`, four hosted endpoints (OpenAI `text-embedding-3-large/small`, Cohere `embed-v4.0`, Voyage `voyage-3-large`) join the candidate pool with indicative per-million-token pricing. They are never suggested for strict or air-gapped setups.

### LLM APIs (optional)

When `--use-llm` is enabled, the tool calls either:
- **Anthropic Messages API** (`api.anthropic.com/v1/messages`)
- **OpenAI Chat Completions API** (`api.openai.com/v1/chat/completions`)

This is used for recommendation verification (`ragadvisor run --use-llm`) and research answer synthesis (`ragadvisor ask --use-llm`). The tool works fully offline without API keys.

## Recommendation Engine

### Embedding Model Scoring

Each candidate model receives a composite fitness score (0.0–1.0) based on weighted factors:

| Factor | Weight | Criteria |
|--------|--------|----------|
| **Retrieval quality** | up to +0.30 | Approximate MTEB nDCG@10 (40 → 0, 60 → +0.30); unknown quality is flagged |
| **Hardware fit** | +0.15 / -0.40 | fp32 size within limits (CPU cap 3 GB, 30% RAM, 70% VRAM); hosted APIs get +0.10 |
| **Language coverage** | +0.25 / -0.30 | Multilingual model when the corpus is non-English, mixed, CJK or growing |
| **Use case** | +0.20 / +0.05 | Code-specialized embedders for code; long inputs for legal/summarization |
| **Latency** | +0.10 / -0.15 | Small models for <500 ms on CPU; APIs penalised for network round-trip |
| **License** | -0.10 | Non-commercial licenses (e.g. `cc-by-nc`) are flagged and penalised |
| **Popularity** | +0.05 | Community validation (>1000 likes or >500K downloads) |
| **Long context** | +0.05 | Bonus for 8192+ token inputs |

The top 5 models (deduplicated, sorted by score) are returned with explanations and warnings.

### Chunking Strategy Selection

Strategies are selected based on content type and language:

| Content Type | Strategy | Notes |
|-------------|----------|-------|
| Code | `language_aware` | AST-aware splitting by function/class boundaries |
| Chat | `speaker_split` | Split by speaker turns |
| Tabular | `row_based` | Preserve row integrity |
| Legal | `hierarchical` | Section/subsection structure |
| Prose / Scientific | `recursive` | Default recursive character splitting |

CJK languages get a 30% reduction in chunk size (character-density adjustment). Language-specific tokenizers are recommended: `jieba` (Chinese), `sudachi` (Japanese), `mecab` (Korean), `spaCy` (European languages).

### Vector Database Selection

8 database profiles are scored against your requirements:

| Database | Category | Max Capacity | Key Strength |
|----------|----------|-------------|--------------|
| ChromaDB | Embedded | 1M docs | Simple local setup |
| LanceDB | Embedded | 10M docs | Serverless, columnar |
| FAISS | In-memory | 10M docs | Fastest similarity search |
| Qdrant | Client-server | 100M docs | Hybrid search, filtering |
| Weaviate | Client-server | 100M docs | GraphQL API |
| Milvus | Client-server | 1B docs | Distributed scale |
| pgvector | PostgreSQL ext. | 10M docs | Existing Postgres infra |
| Pinecone | Managed | 1B docs | Zero-ops, API-based |

### Retrieval Settings

| Use Case | top_k | Temperature | Max Tokens | Strategy | Threshold |
|----------|-------|-------------|------------|----------|-----------|
| Q&A | 5 | 0.1 | 500 | stuff | 0.7 |
| Summarization | 10 | 0.3 | 1500 | map_reduce | 0.5 |
| Code | 3 | 0.0 | 1000 | stuff | 0.8 |
| Semantic search | 10 | — | — | none | 0.6 |
| Legal | 5 | 0.0 | 1000 | refine | 0.75 |

Retrieval settings are further adjusted based on query complexity and expected answer type.

## Output & Reports

Reports are generated in up to 3 formats, written to the output directory (default: `./rag_report/`):

### Markdown (`rag_report.md`)
Human-readable report with all recommendations, reasoning, warnings, LLM verification (if enabled), and an implementation checklist.

### HTML (`rag_report.html`)
Styled report generated from a Jinja2 template with inline CSS. Features responsive tables, color-coded score badges (green/amber/red), and collapsible warning panels.

### YAML (`rag_config.yaml`)
Machine-readable configuration file with all recommendations structured for programmatic consumption.

## Architecture

```
src/rag_adviser/
├── cli.py                          # Typer CLI (run, evaluate, analyze, ask, presets, version)
├── main.py                         # RAGAdviser orchestrator — coordinates the pipeline
├── models.py                       # Data models, enums, exceptions
├── analyzers/
│   ├── document_analyzer.py        # Corpus analysis (language, tokens, content type)
│   ├── constraint_analyzer.py      # Hardware/privacy constraint validation
│   └── approach_analyzer.py        # Determines if RAG is the right approach
├── config/
│   ├── __init__.py                 # YAML config loader
│   └── defaults.yaml               # Scoring weights, fallback models, DB profiles
├── evaluators/
│   ├── pipeline_runner.py          # Full eval orchestrator (chunk → embed → index → query)
│   ├── chunking_strategies.py      # 4 chunking strategies (recursive, semantic, hierarchical, adaptive)
│   ├── vector_store.py             # Abstract store + 4 backends (Chroma, FAISS, pgvector, sqlite-vec)
│   ├── metrics.py                  # Hit Rate, MRR, Precision, Recall, NDCG
│   ├── ground_truth_loader.py      # JSONL/CSV ground truth parser
│   └── baseline.py                 # Baseline save/load/compare for CI regression
├── input_modes/
│   ├── interactive.py              # Rich-powered 13-step guided flow + preset saving
│   ├── cli_params.py               # CLI flag → UserAnswers converter
│   └── xml_input.py                # XML file parser with schema validation
├── llm/
│   ├── client.py                   # Lightweight LLM client (OpenAI + Anthropic via httpx)
│   └── verifier.py                 # LLM verification of recommendations
├── presets/
│   ├── manager.py                  # Load/save/apply presets
│   └── builtin/                    # 6 built-in preset YAML files
├── recommenders/
│   ├── model_finder.py             # HuggingFace Hub API + offline fallback + scoring
│   ├── chunking_recommender.py     # Content-type-aware chunking strategy selection
│   ├── vector_db_recommender.py    # Vector DB scoring and selection
│   └── query_recommender.py        # Query transformation pipeline recommendations
├── reporters/
│   ├── markdown_renderer.py        # Markdown report renderer
│   ├── html_renderer.py            # Jinja2-based HTML renderer
│   ├── yaml_renderer.py            # Machine-readable YAML config
│   └── report_generator.py         # Multi-format report orchestrator
├── research/
│   ├── knowledge_base.py           # Chunk, embed, and index research papers
│   ├── assistant.py                # Interactive Q&A with optional LLM synthesis
│   └── papers/                     # 14 curated knowledge base files
└── templates/
    ├── report.html.j2              # Jinja2 HTML template
    └── report.css                  # Inline stylesheet for HTML/PDF
```

## Known Limitations

This is a heuristic advisor, not an evaluator. Be aware of what it does **not** do:

- **No measurement of your data.** Every recommendation is a rule applied to metadata (languages, size, content type, constraints). Quality scores are public benchmark averages, not results on your corpus. Use `ragadvisor evaluate` with real queries before committing.
- **Corpus statistics are sampled.** 50 files, 5,000 characters each, extrapolated. Highly heterogeneous corpora (a few huge PDFs among thousands of notes) will be estimated poorly; the report states the sample size.
- **Content-type detection is regex-based.** It works for clear-cut corpora and falls back to `mixed` when unsure. Override with `--content-type` when you know better.
- **Similarity thresholds are model-dependent.** The suggested cosine thresholds (0.5–0.8) are starting points; calibrate them against your embedding model.
- **Benchmark figures drift.** Model quality scores and API prices in `defaults.yaml` are snapshots; re-check leaderboards and vendor pricing pages.
- **LLM verification is advisory.** The optional `--use-llm` review is a second opinion generated from the same inputs; it cannot inspect your documents.

## Configuration

All defaults, scoring weights, fallback models, and database profiles are defined in `src/rag_adviser/config/defaults.yaml`.

### Key configurable parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_sample_docs` | 50 | Max documents sampled for analysis |
| `max_chars_per_doc` | 5000 | Characters sampled per document |
| `min_language_confidence` | 0.1 | Minimum confidence for language detection |
| `default_overlap_ratio` | 0.1 | Chunk overlap as fraction of chunk size |
| `cjk_chunk_size_multiplier` | 0.7 | Chunk size reduction for CJK languages |

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
