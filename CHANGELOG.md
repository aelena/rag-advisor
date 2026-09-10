# Changelog

All notable changes to RAG Advisor. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-09-10

### Added
- `ragadvisor example-corpus DIR`: exports the 14 bundled RAG research notes
  and 40 hand-written ground-truth queries so `--validate` can be tried in one
  command before preparing your own evaluation set.
- `ragadvisor bootstrap-queries CORPUS`: samples passages evenly across a
  corpus and asks the configured LLM to write one specific question and short
  answer per passage, producing a JSONL ground-truth file. Entries are marked
  `synthetic: true`; the validator and evaluation report flag such metrics as
  indicative. `--dry-run` shows the sampled passages without API calls.
- JSON report format (`-f json`, included in `-f all`): the same content as
  the YAML config as `rag_config.json`.

### Changed
- GitHub Actions workflows use the Node 24 action majors (checkout v7,
  setup-python v7, upload/download-artifact v7/v8, create-pull-request v8,
  action-gh-release v3), removing the Node 20 deprecation warnings.
- README: refreshed pipeline overview and table of contents, documented the
  new commands and the JSON format.

## [0.3.0] - 2026-09-09

The advice-to-measurement release. Every recommendation the advisor makes can
now be checked on the user's own corpus and queries.

### Added
- `ragadvisor run --validate`: runs the recommended chunking strategy, chunk
  size, embedding model, hybrid setting and reranker through the evaluation
  pipeline against `--ground-truth-path`, with a verdict, next steps and a
  dense-only baseline for comparison. `--validate-models N` compares the top N
  recommended local embedding models; `--validate-chunk-sizes 256,512,1024`
  sweeps sizes around the recommendation.
- Hybrid retrieval recommendations (BM25 + dense, reciprocal rank fusion) with
  native hints for Qdrant, Weaviate, pgvector, LanceDB, Pinecone and Milvus, and
  a ready-to-run `rank-bm25` snippet.
- Constraint-aware reranker recommendations: a catalogue of 8 open
  cross-encoders and 2 hosted rerankers, chosen by quality, latency budget,
  languages, input window, memory, privacy and license.
- Multimodal corpus inventory: images, video, audio, spreadsheets,
  presentations, CAD/BIM files and scanned PDFs are counted and each gets an
  ingestion plan (OCR, captioning, SQL-first tables, transcripts, metadata
  extraction) with local and hosted tool options.
- Cost, footprint and latency estimates in every report: chunk count, index
  size, one-off embedding cost or compute time, monthly re-indexing and query
  API spend (`--queries-per-day`), and per-query retrieval latency by stage
  checked against the latency budget.
- Evaluation pipeline: `--hybrid`, `--rerank MODEL`, `--fetch-k`,
  `--dense-baseline`, repeatable `--model`, repeatable `--chunk-size` with
  `--overlap-ratio`, a dependency-free `memory` vector backend, and
  `--trust-remote-code`.
- `ragadvisor refresh-catalogue`: recomputes embedding quality scores from the
  MTEB results in Hugging Face model cards; a monthly workflow opens a pull
  request when scores move.
- Hosted embedding APIs (OpenAI, Cohere, Voyage) as candidates when the budget
  allows paid APIs and privacy permits.
- Claude Code skill (`.claude/skills/rag-advisor`) that runs the adviser on a
  folder and explains the report.
- Fully commented XML template at `examples/answers.example.xml`; XML mode
  accepts every option the flags do.
- GitHub Actions CI (ruff, pytest on Python 3.10-3.13 and Windows, CLI smoke
  test) and a tag-triggered PyPI release workflow.

### Changed
- Embedding model ranking is now driven by an approximate MTEB retrieval
  quality score; the catalogue grew to 14 open models with licenses,
  `trust_remote_code` flags and long-context metadata.
- Corpus token totals are extrapolated from an evenly spaced sample instead of
  being capped by it; content type is decided per document with file-extension
  priors and reported with a confidence.
- Chunking, vector database and query-pipeline code snippets use current
  libraries (`langchain_text_splitters`, LCEL, `pinecone`, HNSW for pgvector)
  and the recommended model's embedding dimension.
- The LLM verification client targets current Anthropic models and no longer
  sends sampling parameters they reject.
- `--preset` combined with `--no-interactive` now lets explicit flags override
  preset values.

### Fixed
- A `--content-type` override was dropped when no corpus was analyzed.
- The use case never reached the chunking recommender.
- Prose with commas was classified as tabular and routed to Text-to-SQL.
- Large corpora were misjudged as fitting a single context window.
- Hugging Face Hub queries could hang the CLI; they now run concurrently under
  a 20-second budget.

## [0.2.0] - 2026-09-09

Review release: correctness fixes to corpus analysis, content-type detection,
preset handling and model ranking; lint-clean codebase; regression test suite.

## [0.1.0] - 2026-03-06

Initial rule-based adviser: interactive, flag and XML input; embedding model,
chunking, vector database, retrieval and query-pipeline recommendations;
Markdown, HTML and YAML reports; evaluation pipeline; presets; research
assistant.
