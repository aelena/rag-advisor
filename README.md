# RAG Advisor

[![CI](https://github.com/aelena/rag-advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/aelena/rag-advisor/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ragadvisor?label=PyPI)](https://pypi.org/project/ragadvisor/)
[![Version](https://img.shields.io/badge/version-0.6.2-informational)](CHANGELOG.md)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typer](https://img.shields.io/badge/CLI-Typer-009485?logo=fastapi&logoColor=white)](https://typer.tiangolo.com/)
[![Works offline](https://img.shields.io/badge/works-offline-8A2BE2)](#external-api-integration)

---

```plaintext

██████   █████   ██████   █████  ██████  ██    ██ ██ ███████  ██████  ██████  
██   ██ ██   ██ ██       ██   ██ ██   ██ ██    ██ ██ ██      ██    ██ ██   ██ 
██████  ███████ ██   ███ ███████ ██   ██ ██    ██ ██ ███████ ██    ██ ██████  
██   ██ ██   ██ ██    ██ ██   ██ ██   ██  ██  ██  ██      ██ ██    ██ ██   ██ 
██   ██ ██   ██  ██████  ██   ██ ██████    ████   ██ ███████  ██████  ██   ██ 
                                                                              
                                                                              
```

---


**Rule-based recommendations for building Retrieval-Augmented Generation systems.**

RAG Advisor analyses your document corpus, collects your infrastructure constraints, and generates a complete RAG configuration — embedding model, chunking strategy, vector database, retrieval settings and physical sizing — tailored to your exact requirements. Ships as three consumption surfaces:

- **CLI** — `ragadvisor …` (`pip install ragadvisor`).
- **Claude Code skill** — `/rag-advisor <folder>` inside Claude Code; drives the CLI, collects constraints from your prompt, and explains the report.
- **Python library** — `from rag_adviser import RAGAdviser, UserAnswers, …`; plain-dataclass inputs and outputs, no framework lock-in.

Informed defaults, measurable hypotheses, targeted experiments: the recommendation is a starting point, and `--validate` measures whether it actually works on your queries before you commit to indexing.

## Table of Contents

- [Reading the report](#reading-the-report)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Claude Code Skill](#claude-code-skill)
- [Python API](#python-api)
- [Profile Presets](#profile-presets)
- [Cost, Footprint & Latency Estimates](#cost-footprint--latency-estimates)
- [Multimodal Corpora](#multimodal-corpora)
- [Validate Before You Build](#validate-before-you-build)
- [BYOK LLM Integration](#byok-llm-integration)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Research Assistant](#research-assistant)
- [CLI Reference](#cli-reference)
- [Document Analysis](#document-analysis)
- [External API Integration](#external-api-integration)
- [Recommendation Engine](#recommendation-engine)
- [Output & Reports](#output--reports)
- [Architecture](#architecture)
- [Known Limitations](#known-limitations)
- [Configuration](#configuration)
- [Releasing](#releasing)
- [Changelog](CHANGELOG.md)

## Reading the report

The generated `rag_report.md` / `.html` / `.yaml` / `.json` quartet reports one recommendation and a lot of surrounding evidence. A few conventions worth internalising before you scan the tables:

- **Every scalar is a starting point, not a measurement.** `chunk_size = 512`, `top_k = 5`, `similarity_threshold = 0.75` and their siblings encode reasonable defaults for the profile you described. They are hypotheses to test with `--validate`, not conclusions. Numbers are pinned in the report so they can be compared across runs, not because the tool measured them on your data.
- **Confidence figures are bands, not probabilities.** `Approach: RAG (confidence 85%)` means "five of five typical RAG signals matched"; it does not mean the tool ran an experiment. The `Approaches considered` table shows the runner-up so you can see how narrow the margin was — a shift in corpus size or use case may change which candidate wins.
- **Estimates that come from a sample are marked with `~`.** A token total prefixed with `~` was extrapolated from an adaptive sample; the report says how many files were opened alongside a Wilson 95% CI on rates such as scanned-PDF share.
- **The physical footprint depends on the sizing preset.** Index RAM and disk figures reflect the `--sizing-preset` you picked (or the `cpu-balanced` default). fp16 halves index RAM vs fp32; scalar quantization compresses further; `on-disk-mmap` keeps only the HNSW graph resident.
- **The four output formats are semantically equivalent.** Anything you see in the Markdown report is present in the YAML/JSON config too. `rag_config.yaml` is the machine-readable hand-off — a downstream builder should only need that file.
- **When something is fragile, the report says so.** Warnings are surfaced verbatim: reranker/language mismatches, multimodal share, filename fragments, corpus-token extrapolation from a small sample, non-commercial licences. Read them before you invest CPU-days indexing.

Before shipping, run `ragadvisor run --validate --ground-truth-path queries.jsonl` on your own queries. The report is a starting point; validation is how you find out whether it was a good one for your workload.

## How It Works

RAG Advisor runs a pipeline of analysis, recommendation and (optionally) measurement steps:

```
 Input Collection ──► Document & Modality Analysis ──► Approach Gate ──► Constraint Checks
         │
         ▼
 Embedding Model Ranking ──► Chunking ──► Hybrid Decision ──► Vector DB ──► Retrieval Settings
         │
         ▼
 Reranker Selection ──► Query Pipeline ──► Cost & Latency Estimates ──► Implementation Steps
         │
         ▼
 Validation on your ground truth (optional) ──► LLM Verification (optional) ──► Reports
```

1. **Collect user inputs** — corpus path, use case, deployment constraints, hardware limits, privacy, expected traffic
2. **Analyze the corpus** — languages, content type with confidence, extrapolated token counts, and an inventory of non-text modalities (images, video, spreadsheets, CAD, scanned PDFs) with ingestion plans
3. **Assess the approach** — decide whether RAG is right at all, or whether direct context, Text-to-SQL, full-text search or structured extraction fits better
4. **Check constraints** — flag conflicts (air-gapped + paid API, edge + large models, sub-second latency on CPU)
5. **Rank embedding models** — curated open models plus live HuggingFace Hub results, and hosted APIs when budget and privacy allow, scored by benchmark quality, hardware fit, languages, latency and license
6. **Recommend chunking** — strategy and token sizes from content type, use case and the model's input window
7. **Decide on hybrid retrieval** — BM25 + dense with reciprocal rank fusion when keyword queries or lexical content call for it
8. **Select the vector database** — 8 profiles scored on scale, deployment, privacy, update frequency and native hybrid support
9. **Configure retrieval** — top-K, similarity threshold, prompt strategy and generation settings per use case
10. **Choose a reranker** — whether to rerank and which cross-encoder fits the latency budget, languages and chunk length
11. **Design the query pipeline** — condensation, expansion, HyDE, multi-query, step-back, with LCEL snippets
12. **Estimate cost and latency** — chunk count, index footprint, indexing cost or time, per-query latency by stage against the budget, monthly API spend
13. **Validate** (optional) — run the recommended configuration on your corpus and ground-truth queries, compare models and chunk sizes, measure hybrid and reranking against a dense baseline
14. **LLM verification** (optional) — send the recommendations (never your documents) to an LLM for a second opinion
15. **Generate reports** — Markdown, HTML and YAML with code snippets, warnings and an implementation checklist

## Installation

Requires Python 3.10+.

```bash
pip install ragadvisor            # from PyPI
pip install -e .                  # from a clone, for development
```

See the [changelog](CHANGELOG.md) for what each release added.

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

### Try it in two commands

```bash
pip install "ragadvisor[eval]"
ragadvisor example-corpus ./ragadvisor-example
ragadvisor run --no-interactive -d ./ragadvisor-example/corpus -u question_answering --privacy strict \
    --ground-truth-path ./ragadvisor-example/queries.jsonl --validate
```

The example is a small corpus of 14 research notes on RAG with 40 hand-written questions. The second command analyses it, recommends a configuration, then measures that recommendation on those questions and reports hit rate, MRR and a verdict. Add `--validate-models 2` or `--validate-chunk-sizes 256,512,1024` to compare options; every extra model or size re-embeds the corpus and, when a reranker is recommended, reranks all queries, so allow several minutes per combination on a CPU.

### Interactive mode (default)

```bash
ragadvisor run
```

Launches a guided 13-step questionnaire with a Rich terminal UI, organized into 3 phases:

- **Phase 1** — Document discovery (corpus path, languages, content type)
- **Phase 2** — Use case & constraints (deployment, hardware, privacy, budget)
- **Approach Gate** — Checks if RAG is the right approach before continuing
- **Phase 3** — Query & operational patterns (query type, complexity, answer type, update frequency)

If you give a ground-truth file, the flow offers to validate the recommendation right away and asks how many embedding models to compare and which extra chunk sizes to try. At the end, you can optionally save your answers as a custom preset for future use.

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

For scripted or repeatable runs, put every answer in one file and version it next to your project:

```bash
ragadvisor run --from-xml examples/answers.example.xml -f all -o ./rag_report
```

The schema mirrors the questionnaire in three sections: `document_discovery` (corpus path, content type, future languages), `use_case_constraints` (use case, deployment, latency, hardware, budget, privacy, preferred libraries) and `query_patterns` (query type and complexity, expected answer type, sample queries, update frequency, expected queries per day, ground truth path, `validate`, `validate_models`, LLM verification). Every element is optional. [`examples/answers.example.xml`](examples/answers.example.xml) is a fully commented template; flags passed alongside `--from-xml` override the file.

### Quick document analysis only

```bash
ragadvisor analyze ./my_documents
```

## Claude Code Skill

The repository ships a project skill at `.claude/skills/rag-advisor/SKILL.md`. With this repo open in Claude Code, invoke it as `/rag-advisor <folder> [constraints]`. The skill runs `analyze` on the folder, maps your stated constraints ("must stay on-prem", "chatbot over docs", "fast", …) to CLI flags, runs the adviser, reads the generated Markdown report and explains the verdict, warnings and stack in plain language — then offers `--validate` when you have ground-truth queries.

### Example invocations

```
/rag-advisor ./docs
/rag-advisor ~/Corpora/handbooks "must stay on-prem, fast responses"
/rag-advisor ./contracts "legal analysis, privacy strict, GPU 16 GB"
/rag-advisor ./corpus "chatbot over docs" --validate ./queries.jsonl
```

The skill handles the constraint → flag mapping (see `.claude/skills/rag-advisor/SKILL.md` for the full table), asks only for details that would actually change the recommendation, and warns loudly when the recommendation is fragile (scanned PDFs, multilingual reranker mismatch, RAM budget breached, filename fragments, low content-type confidence).

Nothing leaves your machine: the skill invokes the local CLI. The `--use-llm` step is opt-in and uses your key, exactly as it does at the CLI (see [BYOK LLM Integration](#byok-llm-integration)).

## Python API

RAG Advisor is a library first and a CLI on top. Every load-bearing type is re-exported from the top of the package so you can drive the recommender from a notebook, a test suite, or another tool:

```python
from pathlib import Path

from rag_adviser import (
    RAGAdviser,
    UserAnswers, HardwareConstraints,
    UseCase, LatencyBudget, HardwareProfile, PrivacyLevel,
    ReportFormat,
    Recommendations,
)

answers = UserAnswers(
    document_path=Path("./my-corpus"),
    use_case=UseCase.QA,
    constraints=HardwareConstraints(
        hardware=HardwareProfile.CPU_ONLY,
        ram_gb=32,
        latency_budget=LatencyBudget.MODERATE,
        privacy=PrivacyLevel.STRICT,
    ),
)

recs: Recommendations = RAGAdviser().run(
    answers,
    output_dir=Path("./out"),
    formats=[ReportFormat.MARKDOWN, ReportFormat.YAML],
)

# recs is a plain dataclass — inspect it programmatically
top_model = recs.embedding_models[0]
print(f"embedding: {top_model.model_id} (dim={top_model.dimension})")
print(f"chunks: {recs.estimates.chunk_count:,}")
print(f"index RAM: {recs.estimates.index_memory_mb:.0f} MB")
for w in recs.warnings:
    print(f"! {w}")
```

Every field is a plain Python value (str / int / float / list / dict / enum / dataclass) — serialise with `dataclasses.asdict`, diff in tests, feed into another tool.

### What's in `Recommendations`

The returned `Recommendations` object holds the entire result tree:

| Attribute | Type | Contents |
|---|---|---|
| `approach` | `ApproachAssessment` | Top pick + ranked `candidates` (RAG vs Direct-Context vs Long-Context vs Text-to-SQL vs …) with evidence |
| `embedding_models` | `list[EmbeddingModelRecommendation]` | Top pick at index 0, alternatives with scores and reasons |
| `chunking` | `ChunkingRecommendation` | Strategy, size, overlap, language overrides, code snippet |
| `vector_db` | `VectorDBRecommendation` | Provider, category, library, reason |
| `retrieval` | `RetrievalRecommendation` | top_k, threshold, hybrid, rerank, calibration_target, abstention_policy |
| `reranker` | `RerankerRecommendation` | Picked model, coherence warnings, alternatives |
| `query_transformation` | `QueryTransformationRecommendation` | HyDE / step-back / multi-query with per-technique latency |
| `modalities` | `list[ModalityRecommendation]` | Per-modality ingestion advice (scanned PDF, image, audio, video, …) |
| `estimates` | `CostEstimate` | Chunks, index size, RAM, indexing time, retrieval-latency scenarios |
| `validation` | `ValidationResult \| None` | Populated when `run_validation=True` |
| `llm_verification` | `LLMVerification \| None` | Populated when `use_llm_verification=True` |
| `warnings` | `list[str]` | CRITICAL / LATENCY / FILENAME FRAGMENTS / SCANNED PDFs / etc. |
| `implementation_steps` | `list[str]` | The Implementation Checklist rendered in the report |

### Composing sub-recommenders

You don't have to run the full pipeline — every stage is usable in isolation:

```python
from pathlib import Path

from rag_adviser.analyzers.document_analyzer import DocumentAnalyzer
from rag_adviser.analyzers.approach_analyzer import ApproachAnalyzer
from rag_adviser.recommenders.model_finder import HFModelFinder
from rag_adviser.recommenders.cost_estimator import CostEstimator
from rag_adviser.presets.manager import load_sizing_preset

# 1. Analyse the corpus in isolation
stats = DocumentAnalyzer().analyze(Path("./my-corpus"))
print(f"{stats.total_files:,} text docs, ~{stats.total_tokens:,} tokens")
print(f"p50={stats.tokens_p50:,}, p99={stats.tokens_p99:,}")

# 2. Ask the approach analyser what it would recommend
from rag_adviser import HardwareConstraints, UseCase, UserAnswers
answers = UserAnswers(document_stats=stats, use_case=UseCase.QA,
                     constraints=HardwareConstraints())
decision = ApproachAnalyzer().assess(answers)
for c in decision.candidates:
    print(f"  {c.approach.value:20s} {c.confidence:.0%}  {c.reasoning}")

# 3. Just find embedding models — offline mode skips the HF Hub call
models = HFModelFinder(offline=True).find_models(
    doc_stats=stats, constraints=answers.constraints, use_case=UseCase.QA,
)
print(models[0].model_id)

# 4. Load a sizing preset by name for use elsewhere
sizing = load_sizing_preset("gpu-fp16-quantized")
```

### Using it from tests

Because everything is dataclass-in / dataclass-out, RAG Advisor is easy to test-double:

```python
def test_our_pipeline_matches_advisor_recommendation():
    recs = RAGAdviser().run(
        UserAnswers(
            document_path=Path("./fixtures/corpus"),
            use_case=UseCase.QA,
        ),
        output_dir=Path("./out"),
        formats=[ReportFormat.YAML],
    )
    assert recs.embedding_models[0].model_id.startswith("BAAI/bge")
    assert recs.chunking.chunk_size == 512
    assert not any(w.startswith("CRITICAL:") for w in recs.warnings)
```

### Stable-vs-internal boundary

- **Stable** — everything in `rag_adviser.__all__`. Renames go through a deprecation cycle.
- **Internal** — anything imported from a sub-module (`rag_adviser.recommenders.model_finder.HFModelFinder`, etc.). Usable, but expect signature changes without a semver bump.

If you find yourself relying on an internal name, open an issue — we'll consider promoting it.

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

## Cost, Footprint & Latency Estimates

Every report ends the recommendation with numbers you can plan against: chunk count and tokens to embed, index size on disk and vectors in RAM, one-off embedding cost (hosted APIs) or compute time (local models), monthly re-indexing cost from the update frequency, per-query retrieval latency broken down by stage (query embedding, vector search, BM25 fusion, reranking, LLM query transformations) checked against the latency budget, and monthly query-side API spend at the assumed volume (`--queries-per-day`, default 1,000).

These are order-of-magnitude estimates built from public throughput and price figures; the assumptions are listed under the table. A retrieval stack that blows the latency budget raises a warning naming the heaviest stage.

## Multimodal Corpora

Real corpora are rarely text only. The analyzer inventories every file it finds and the report gets a **Non-Text Modalities** section with an ingestion plan per kind:

| Modality | Detected from | Recommended treatment |
|----------|---------------|------------------------|
| Scanned PDFs | PDFs in the sample with no text layer | OCR + layout-aware parsing (docling, marker, PaddleOCR; Azure Document Intelligence / Textract hosted), chunk by section |
| Images | `.png .jpg .tiff .webp .svg ...` | Caption with a vision-language model or OCR document-like images; optional CLIP/SigLIP image index; joint multimodal embeddings (Cohere embed-v4, Voyage multimodal) when APIs are allowed |
| Spreadsheets | `.xlsx .xls .ods ...` | Fact tables to DuckDB/SQL + Text-to-SQL; small lookup tables serialized row-wise with headers |
| Presentations | `.pptx .ppt .odp .key` | One chunk per slide (title + body + notes), captions for diagram slides |
| Video | `.mp4 .mov .mkv ...` | Transcribe (faster-whisper), 30-60 s timestamped chunks, key-frame captions |
| Audio | `.mp3 .wav .m4a ...` | Transcription + speaker diarization, speaker-turn chunks |
| CAD / blueprints | `.dwg .dxf .ifc .rvt .step ...` | Structured metadata first (ezdxf, ifcopenshell), render sheets to images for captions, metadata filters over similarity |

Hosted services are only proposed when privacy and budget allow; otherwise they are listed for reference and the plan stays local. When 30% or more of the files are not text, the report leads with a multimodal warning because the text pipeline covers only part of the corpus.

## Validate Before You Build

`--validate` closes the loop between advice and measurement. It takes the recommended chunking strategy, chunk size, embedding model and top-k, runs them through the evaluation pipeline on **your** corpus and **your** queries, and attaches the retrieval metrics to the report with a verdict and concrete next steps.

```bash
ragadvisor run --no-interactive -d ./docs --use-case question_answering \
  --ground-truth-path ./queries.jsonl --validate
```

- Requires `pip install ragadvisor[eval]` (embeds the corpus locally; nothing leaves your machine).
- Ground truth is the same JSONL/CSV format as `ragadvisor evaluate`: `{"query": "...", "relevant_docs": ["file.txt"]}` per line.
- If the top embedding recommendation is a hosted API, the best local alternative is evaluated instead and the report says so. If a model fails to load (missing extra package, gated repo), the next recommended local model is tried and the report lists what was skipped.
- With no vector database installed, an exact numpy search (`--backend memory`) is used, so only `sentence-transformers` is strictly required.
- `--validate-models 2` (or more) evaluates the top recommended local embedding models side by side and reports which one actually retrieves best on your queries.
- `--validate-chunk-sizes 256,512,1024` sweeps chunk sizes around the recommendation and says whether a different size wins on your data.
- Hybrid retrieval and a local reranker, when recommended, are part of what gets measured. The same index is also queried dense-only, so the report states whether those stages actually helped on your data.
- Verdicts: **strong** (hit rate ≥ 80%), **acceptable** (≥ 60%), **weak**. Weak results come with suggestions: raise top-k, add a reranker, enable hybrid search, or compare strategies with `ragadvisor evaluate`.
- The interactive flow offers validation as soon as you provide a ground-truth path.

### No evaluation queries yet? Bootstrap a synthetic set

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
ragadvisor bootstrap-queries ./docs -o ./queries.synthetic.jsonl --n 30
ragadvisor run --no-interactive -d ./docs --ground-truth-path ./queries.synthetic.jsonl --validate
```

`bootstrap-queries` samples passages evenly across the corpus and asks the LLM to write one specific question and short answer per passage. Only the sampled passages are sent to the API. Every entry is marked `"synthetic": true`, and the validator and evaluation report say so, because LLM-written questions are easier than real user questions and tend to overstate retrieval quality. Use them to get the loop running, then replace them with real questions as they come in. `--dry-run` shows the sampled passages without any API call.

## BYOK LLM Integration

RAG Advisor's core recommendation pipeline is **local-first**: it runs to completion with no API keys, no telemetry, no phone-home. LLM calls are strictly **opt-in** — you enable them per-command, using **your own API key** from your own environment. The tool never stores your key, never uploads your documents unless you explicitly ask it to via the one subcommand that does, and works fully offline for every other feature.

### The three opt-in LLM features

| Feature | CLI flag | What is sent | What data leaves your machine |
|---|---|---|---|
| **Recommendation verification** | `ragadvisor run --use-llm` | The generated recommendation (embedding model, chunking, retrieval, sizing, warnings) | **Recommendations only — never your documents.** Corpus stats (counts, sizes, language mix) are included so the LLM can reason about scale. |
| **Research assistant synthesis** | `ragadvisor ask --use-llm "..."` | Your question + top-K passages from **the bundled research knowledge base**, not your corpus | Your question text and passages from RAG Advisor's own shipped research notes. |
| **Ground-truth bootstrap** | `ragadvisor bootstrap-queries CORPUS` | **Sampled passages from your corpus**, one at a time, asking the LLM to write a question for each | Your document text (sampled evenly). This is the only path that sends your corpus content — the subcommand is separate and explicit. |

Everything else (analyze, run without `--use-llm`, evaluate, validate, presets, HuggingFace embedding-model catalogue refresh) is local-only.

### Configuring your key

RAG Advisor auto-detects the first available key from environment variables:

```bash
# Anthropic (checked first, preferred)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (fallback if no Anthropic key)
export OPENAI_API_KEY=sk-...

# Neither set? Recommendation still runs; --use-llm features print
# "LLM not available" and continue with the rest of the report.
```

The env vars are read once when `--use-llm` or `bootstrap-queries` runs. Nothing writes them to disk.

### Model, endpoint and provider overrides

Everything is overridable via env vars — no config file to edit.

```bash
# Pick a specific model
export RAGADVISOR_LLM_MODEL=claude-opus-4-7        # Anthropic
export RAGADVISOR_LLM_MODEL=gpt-4o-mini            # OpenAI

# Route through a proxy, gateway or self-hosted endpoint
export ANTHROPIC_BASE_URL=https://your-proxy.example.com
export OPENAI_BASE_URL=https://your-litellm.example.com/v1

# Common OpenAI-compatible endpoints work as-is
export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama
export OPENAI_BASE_URL=https://your-azure-openai.openai.azure.com/openai/deployments/<name>
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=<the-endpoint's-key>
```

Environment variable reference:

- `ANTHROPIC_API_KEY` — Anthropic key (checked first)
- `OPENAI_API_KEY` — OpenAI or OpenAI-compatible key (fallback)
- `RAGADVISOR_LLM_MODEL` — override the default model
- `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` — override the API endpoint
- `HF_HUB_OFFLINE=1` — skip the HuggingFace Hub call at model-discovery time (falls back to the curated model catalogue)

### End-to-end examples

**1) Recommendation with a second opinion from Claude:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
ragadvisor run -d ./my-corpus --use-case question_answering --use-llm
```

The `LLMVerification` section appears in every rendered format (Markdown, HTML, YAML, JSON). The LLM sees the recommendation table, not your documents.

**2) Recommendation without any LLM call (default):**

```bash
# No env vars needed. Nothing leaves your machine.
ragadvisor run -d ./my-corpus --use-case question_answering
```

**3) Ask a research question against the bundled knowledge base:**

```bash
# Retrieval-only (default) — no LLM required
ragadvisor ask "when should I use HyDE?"

# LLM synthesis — uses your key to write a coherent answer
export ANTHROPIC_API_KEY=sk-ant-...
ragadvisor ask --use-llm "when should I use HyDE?"
```

**4) Generate synthetic ground-truth queries from your corpus** *(this one sends corpus passages — use only if the corpus can leave your machine):*

```bash
export OPENAI_API_KEY=sk-...
ragadvisor bootstrap-queries ./my-corpus --output ./queries.jsonl

# Dry-run — shows the passages that would be sent, without any API calls
ragadvisor bootstrap-queries ./my-corpus --dry-run
```

**5) Route Claude through a local Ollama server (no cloud call at all):**

```bash
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export RAGADVISOR_LLM_MODEL=llama3.1
ragadvisor run -d ./my-corpus --use-llm
```

### Privacy interactions

RAG Advisor also prunes recommendations based on your `--privacy` setting, independent of BYOK:

- `--privacy strict` or `--privacy air_gapped` removes hosted embedding APIs (OpenAI, Cohere, Voyage) from the candidate pool and marks any `hosted` modality tools as reference-only.
- With `--privacy strict`, `--use-llm` still works — it's your key, your choice — but the surrounding recommendation is built assuming nothing else leaves your machine.
- `ragadvisor bootstrap-queries` refuses to run silently on a strict-privacy setup; you have to opt in explicitly by re-running without `--privacy strict`.

### What the LLM never sees

- Your document contents (unless you explicitly run `bootstrap-queries`).
- File paths, filenames or directory layout.
- Any credentials other than the one API key you set in your own environment.
- Persistent state across runs (each `--use-llm` invocation is stateless).

## Evaluation Pipeline

Compare chunking strategies against ground truth data with real embeddings and vector search.

```bash
# Basic evaluation
ragadvisor evaluate ./docs ./queries.jsonl

# Compare specific strategies
ragadvisor evaluate ./docs ./queries.jsonl --strategy recursive --strategy semantic

# Sweep chunk sizes (overlap scales with size); the report shows MRR per size per strategy
ragadvisor evaluate ./docs ./queries.jsonl --chunk-size 256 --chunk-size 512 --chunk-size 1024

# Compare embedding models (models that fail to load are skipped, not fatal)
ragadvisor evaluate ./docs ./queries.jsonl -m BAAI/bge-small-en-v1.5 -m BAAI/bge-base-en-v1.5

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
- **speaker_split** — Whole speaker turns grouped up to the chunk size (chat logs, transcripts); files without speaker markers fall back to recursive
- **row_based** — Header + row groups for CSV/TSV and other delimited tables; non-tabular files fall back to recursive

### Retrieval modes

Every strategy can be evaluated as plain dense retrieval, **hybrid** (a dependency-free BM25 index fused with the dense results by reciprocal rank fusion), **dense + rerank** (a sentence-transformers cross-encoder over the top `--fetch-k` candidates), or **hybrid + rerank**. With `--dense-baseline` (default) the dense-only numbers for the same index are reported alongside, so the value of each stage is visible.

```bash
ragadvisor evaluate ./docs ./queries.jsonl --hybrid --rerank cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Vector backends

- **chroma** — ChromaDB in-memory (zero setup)
- **faiss** — FAISS IndexFlatIP (fastest similarity search)
- **pgvector** — PostgreSQL + pgvector (production-realistic)
- **sqlite** — SQLite + sqlite-vec (lightweight, file-based)
- **memory** — exact numpy search, no extra dependency (default for `--validate` when nothing else is installed)

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
| `--queries-per-day N` | | Expected query volume for cost/capacity estimates (default: 1000) |
| `--ground-truth-path PATH` | | Ground truth file (JSONL/CSV) for `--validate` |
| `--validate / --no-validate` | | Run the recommended configuration against the ground truth and report metrics |
| `--validate-models N` | | With `--validate`, compare the top N recommended local embedding models on your data |
| `--validate-chunk-sizes LIST` | | With `--validate`, also try these chunk sizes in tokens (e.g. `256,512,1024`) |
| `--format FORMAT` | `-f` | `markdown`, `html`, `yaml`, `json`, `all` (default: `all`) |
| `--output DIR` | `-o` | Output directory (default: `./rag_report`) |
| `--use-llm / --no-llm` | | Send recommendations to an LLM for verification |

### `ragadvisor evaluate CORPUS GROUND_TRUTH`

| Flag | Short | Description |
|------|-------|-------------|
| `--model NAME` | `-m` | Embedding model(s) to compare, repeatable (default: `all-MiniLM-L6-v2`) |
| `--strategy NAME` | `-s` | Chunking strategies to compare (repeatable): `recursive`, `semantic`, `hierarchical`, `adaptive`, `speaker_split`, `row_based` |
| `--chunk-size N` | | Chunk size in characters; repeat to sweep several sizes (default: 512) |
| `--chunk-overlap N` | | Overlap in characters for a single size (default: 50) |
| `--overlap-ratio X` | | When sweeping, overlap = size × ratio (default: 0.1) |
| `--top-k N` | `-k` | Results to retrieve per query (default: 5) |
| `--backend NAME` | `-b` | Vector backend: `chroma`, `faiss`, `pgvector`, `sqlite` |
| `--db-connection STR` | | Connection string for pgvector or sqlite |
| `--hybrid / --no-hybrid` | | Fuse BM25 with dense results (reciprocal rank fusion) |
| `--rerank MODEL` | | Cross-encoder to rerank fetched candidates |
| `--fetch-k N` | | Candidates fetched before fusion/reranking (default: 20) |
| `--dense-baseline / --no-dense-baseline` | | Also evaluate dense-only on the same index (default: on) |
| `--trust-remote-code` | | Allow models that ship custom code |
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

### `ragadvisor example-corpus [DIR]`

Exports the bundled example corpus (14 research notes on RAG) and 40 hand-written ground-truth queries to `DIR` (default `./ragadvisor-example`) and prints the `--validate` command to run against them.

### `ragadvisor bootstrap-queries CORPUS`

| Flag | Short | Description |
|------|-------|-------------|
| `--output PATH` | `-o` | JSONL file to write (default: `./queries.synthetic.jsonl`) |
| `--n N` | | Questions to generate (default: 30) |
| `--chunk-chars N` | | Passage size shown to the LLM (default: 1200) |
| `--dry-run` | | Show sampled passages only, no LLM calls |

Needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; see [BYOK LLM Integration](#byok-llm-integration) for model overrides.

### `ragadvisor refresh-catalogue [--write]`

Recomputes the embedding-model `quality_score` values in `defaults.yaml` from the MTEB results published in each model's Hugging Face model card (average English retrieval nDCG@10, CQADupstack subsets counted once). Dry run by default; `--write` rewrites only the score lines and keeps every comment. Models without a model-index (e.g. the older `sentence-transformers/*` checkpoints), hosted APIs and rerankers are reported but left for manual maintenance. A monthly GitHub Actions workflow runs it and opens a pull request when anything moved.

| Flag | Description |
|------|-------------|
| `--write / --dry-run` | Apply changes (default: dry run) |
| `--min-datasets N` | Retrieval datasets a card must report to be trusted (default: 10) |
| `--min-delta X` | Ignore changes smaller than X points (default: 0.5) |

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

### Reranking

Reranking is decided and sized rather than switched on blindly. A cross-encoder stage is recommended when the evidence calls for precision: batch latency budgets, multi-hop, aggregative or comparative queries, hybrid fusion (a mixed candidate list needs consistent re-scoring), exact-passage answers, or legal/code use cases.

The model is then chosen from a catalogue of 8 open cross-encoders and 2 hosted rerankers (Cohere `rerank-v3.5`, Voyage `rerank-2`) using:

| Factor | Effect |
|--------|--------|
| **Latency** | Reranking gets ~150 ms of a <500 ms budget, ~800 ms of a <2 s budget, unlimited for batch. Cost = candidates × per-pair CPU cost (÷8 on GPU); models that do not fit are excluded |
| **Quality** | Relative quality score (approx., from public BEIR/MIRACL-style results) |
| **Languages** | Multilingual cross-encoder required for non-English or mixed corpora |
| **Input window** | Must fit query + recommended chunk size, else a truncation warning |
| **Hardware / privacy / budget** | fp32 size vs. memory limit; hosted rerankers only with `--budget paid_api` and non-strict privacy |
| **License** | Non-commercial licenses flagged |

The report states the pipeline shape (retrieve top N → rerank → keep K), the estimated latency, up to three alternatives, and a ready-to-run snippet. When reranking is not recommended, it still names the model to reach for if precision turns out to be a problem.

### Hybrid Retrieval (BM25 + dense)

Dense embeddings blur exact terms: product codes, function names, statute numbers, citations. The advisor recommends a BM25 sparse retriever run alongside the vector search and merged with reciprocal rank fusion when the evidence adds up:

| Signal | Weight |
|--------|--------|
| Short keyword queries | strong |
| Code, legal, tabular or scientific content | strong |
| Code assistance, legal analysis or semantic search use case | medium |
| Exact-passage answers expected | medium |
| Aggregative queries | medium |

One strong or two medium signals trigger the recommendation. When hybrid is recommended, vector databases with native hybrid search (Qdrant, Weaviate, pgvector, LanceDB, Pinecone, Milvus) score higher, the report includes a ready-to-run `rank-bm25` + RRF snippet plus the native equivalent for the chosen database, and CJK corpora get a word-segmentation note.

## Output & Reports

Reports are generated in up to 4 formats, written to the output directory (default: `./rag_report/`):

### Markdown (`rag_report.md`)
Human-readable report with all recommendations, reasoning, warnings, LLM verification (if enabled), and an implementation checklist.

### HTML (`rag_report.html`)
Styled report generated from a Jinja2 template with inline CSS. Features responsive tables, color-coded score badges (green/amber/red), and collapsible warning panels.

### YAML (`rag_config.yaml`)
Machine-readable configuration file with all recommendations structured for programmatic consumption.

### JSON (`rag_config.json`)
The same content as the YAML config, for tools that prefer JSON.

## Architecture

```
src/rag_adviser/
├── cli.py                          # Typer CLI (run, evaluate, analyze, ask, presets, example-corpus, bootstrap-queries, refresh-catalogue, version)
├── catalogue_refresh.py            # MTEB score refresh from model cards
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
│   ├── ground_truth_loader.py      # JSONL/CSV ground truth parser (synthetic flag)
│   ├── query_bootstrap.py          # LLM-generated synthetic evaluation queries
│   ├── baseline.py                 # Baseline save/load/compare for CI regression
│   ├── retrieval_modes.py          # BM25 index, RRF fusion, cross-encoder reranking
│   └── validator.py                # --validate: measure recommendations on your data
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
│   ├── query_recommender.py        # Query transformation pipeline recommendations
│   ├── hybrid_recommender.py       # BM25 + dense hybrid retrieval decision
│   ├── reranker_recommender.py     # Cross-encoder selection under latency budgets
│   ├── modality_recommender.py     # Ingestion plans for non-text files
│   └── cost_estimator.py           # Footprint, latency and cost estimates
├── reporters/
│   ├── markdown_renderer.py        # Markdown report renderer
│   ├── html_renderer.py            # Jinja2-based HTML renderer
│   ├── yaml_renderer.py            # Machine-readable YAML config
│   ├── json_renderer.py            # Same config as JSON
│   └── report_generator.py         # Multi-format report orchestrator
├── research/
│   ├── knowledge_base.py           # Chunk, embed, and index research papers
│   ├── assistant.py                # Interactive Q&A with optional LLM synthesis
│   ├── papers/                     # 14 curated knowledge base files (also the example corpus)
│   └── ground_truth.jsonl          # 40 hand-written queries for the example corpus
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
- **Benchmark figures drift.** Open-model quality scores can be refreshed with `ragadvisor refresh-catalogue`; API prices and reranker scores in `defaults.yaml` are manual snapshots.
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

## Releasing

Releases are cut from tags. Bump `version` in `pyproject.toml` and `src/rag_adviser/__init__.py`, add a section to `CHANGELOG.md`, commit, then:

```bash
git tag v0.3.0 && git push origin v0.3.0
```

The release workflow builds the wheel and sdist, checks the tag matches the package version, publishes to PyPI via trusted publishing (configure the `pypi` environment and the trusted publisher on pypi.org once), and attaches the files to a GitHub release.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
