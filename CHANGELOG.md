# Changelog

All notable changes to RAG Advisor. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-09-17

The epistemic shift promised in the 2026-09-16 review: rules with
teeth, honesty about defaults, and separation of what the pipeline
actually costs from what it might cost if the user opts in to
experimental additions.

### Changed
- **Reranker coherence is now a hard gate, not a warning.** The
  recommender used to pick the highest-scored model even when its
  input window would truncate the chunks it was reranking or when it
  didn't cover the corpus languages — warnings sat next to the pick
  without altering it. Length + language coverage are now filtered
  first; the top score among coherent fits wins. When no coherent
  model fits the latency allowance the fallback still runs, but with
  an explicit "no reranker satisfies both length and language
  coverage" warning so the user knows what compromise they got.
- **Approach confidence is derived, not literal.** The default RAG
  branch used to report `confidence: 0.9` regardless of what the
  inputs actually said. Confidence is now a function of how many
  typical RAG signals (corpus size, token count, use case, content
  type, multi-document) the inputs match, and an `evidence` list on
  `ApproachAssessment` enumerates the concrete facts behind the
  decision. A blind assessment lands in the 0.5–0.6 band; a
  fully-populated one lands near 0.9. The evidence list renders in
  every output format.
- **Optional query transformations no longer inflate the headline
  latency.** HyDE marked as OPTIONAL used to be silently added to the
  "1400 ms retrieval latency" figure. The baseline scenario now
  includes only the retrieval + rerank + required transforms
  pipeline; optional/recommended transforms are reported as a second
  scenario with its own total. Per-technique latency is preserved so
  a reader can see "Baseline ~800 ms; Baseline + HyDE ~1400 ms" side
  by side.
- **Softer, less categorical wording** on the scalars the review
  flagged: `top_k = 5` is framed as a heuristic starting point and
  evaluation parameter, similarity thresholds are described as
  model- and corpus-dependent and needing calibration, "low
  temperature keeps answers factual and grounded" is corrected to
  "low temperature reduces output variability; grounding comes from
  retrieval quality, prompt constraints, citations and evaluation".
  The vector-DB catalogue capacity is now framed as an
  order-of-magnitude sanity check, not a measurement. The README's
  "no guesswork, no trial-and-error" claim is replaced with the more
  defensible "informed defaults, measurable hypotheses, targeted
  experiments".

### Added
- `RerankerRecommender._coherent()` — encapsulates the length +
  language hard checks and returns a list of blocker strings for
  reporting.
- `ApproachAssessment.evidence: list[str]` — the concrete facts that
  drove the decision, surfaced in Markdown, HTML and YAML/JSON.
- `CostEstimate.query_latency_scenarios: list[dict]` — the baseline
  vs. baseline+optional-transforms latency table, rendered in every
  format when a second scenario exists.
- Per-technique `latency_ms` field on `QueryTransformationRecommendation.techniques`,
  so the cost estimator can split required from optional cleanly.

### Deferred
- Full `Provenance` dataclass threaded through every numeric decision
  (Phase 2 item 1 of the plan) — the wording changes above cover the
  headline complaint without a schema-wide refactor. A structural
  provenance model can land alongside the Phase 4 physical sizing
  work, where the number of decisions to annotate is smaller and the
  need is stronger.

## [0.3.4] - 2026-09-17

### Fixed
- **Cross-format report drift.** The four output formats used to disagree
  about what the recommendation actually was: the approach decision and
  its confidence were hidden from Markdown and HTML whenever RAG was the
  recommendation (invisible exactly when a reader wants to see it);
  hardware, RAM, budget, latency budget and update frequency were absent
  from the machine-readable YAML/JSON; the embedding-model alternatives
  in YAML were bare model IDs with no scores or reasons; and the vector
  DB reasoning did not survive the trip to the config file. All of these
  are now present in every renderer.

### Added
- `Tokens to embed` row in the HTML estimates table (Markdown already
  had it; the human formats now agree).
- `metadata.hardware / ram_gb / vram_gb / budget / latency_budget /
  update_frequency / expected_queries_per_day / has_ground_truth /
  sample_queries` in `rag_config.yaml` and `rag_config.json`.
- `embedding.score`, `embedding.quality_score`, `embedding.reasons` and
  full alternative records (with `score`, `quality_score`, `dimension`,
  `max_tokens`, `estimated_size_gb`, `multilingual`, `reasons`) in the
  machine-readable config.
- `vector_db.reason` and `vector_db.estimated_capacity` in the machine
  config.
- `modalities[].notes` and `modalities[].code_snippet` in the machine
  config.
- `tests/test_report_equivalence.py` — a cross-format semantic
  equivalence suite that renders all four formats from the same
  `Recommendations` object and asserts every load-bearing field appears
  everywhere. When you add a new decision to the report, add it here so
  the formats can never silently disagree again.

## [0.3.3] - 2026-09-16

### Fixed
- **Legacy presentations were routed to a tool that can't open them.** The
  modality recommender pointed every `.ppt`, `.odp` and `.key` file at
  `python-pptx`, which only reads Office Open XML `.pptx`. The presentation
  recommendation now inspects the actual extensions in the corpus, prepends a
  "convert with LibreOffice headless first" step when a legacy format is
  present, and adds a warning that names the offending extensions.
- **Query preprocessing silently lowercased text and stripped punctuation.**
  `RetrievalRecommendation.query_preprocessing` used to default to
  `{"lowercase": True, "remove_punctuation": True}` — invisible in the
  Markdown/HTML reports, present in the YAML/JSON config, and actively
  harmful for transformer bi-encoders (BGE, E5, GTE, MiniLM…) because it
  destroys case-sensitive names, product codes and legal citations. The
  default is now empty, and any preprocessing that is applied surfaces in
  both the human-readable and machine-readable outputs.

## [0.3.2] - 2026-09-10

### Added
- Evaluation chunkers `speaker_split` (groups whole speaker turns of chat
  logs and transcripts up to the chunk size) and `row_based` (header plus row
  groups for CSV/TSV and other line-oriented tables). Files without speaker
  markers or a consistent delimiter fall back to recursive splitting, so
  mixed corpora still work. Available in `ragadvisor evaluate --strategy`.

### Changed
- `--validate` now evaluates the recommended `speaker_split` and `row_based`
  strategies as themselves instead of approximating them with recursive
  splitting.

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

### Fixed
- Recall@k could exceed 100% when several retrieved chunks came from the same
  relevant document; distinct documents (or passages) are now counted once.

### Changed
- Reranker CPU latency priors recalibrated after measuring real cross-encoder
  cost (the remote-code multilingual rerankers run at hundreds of ms per pair
  on CPU, not tens). Sub-second budgets now rerank a shorter candidate list
  (10-12) so a small cross-encoder still fits; heavier rerankers are steered
  to GPU or batch use.
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
