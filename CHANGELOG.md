# Changelog

All notable changes to RAG Advisor. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.6.2] - 2026-09-19

Positioning + documentation release. The recommender has been a
CLI + skill + library for a while, but the README and package
top-level pretended it was CLI-only. This release makes the Python
API a first-class contract and documents both the skill and the
library alongside the CLI.

### Added
- **Public Python API.** `rag_adviser/__init__.py` now defines
  `__all__` and re-exports the load-bearing types (`RAGAdviser`,
  `UserAnswers`, `HardwareConstraints`, `SizingProfile`,
  `Recommendations`, the enums for use case / hardware / latency /
  privacy / budget / query type / query complexity / answer type /
  citation granularity / error cost / update frequency / content
  type / deployment target / report format / recommended approach,
  the `Recommendations` sub-dataclasses, and the error hierarchy).
  A downstream caller can now write `from rag_adviser import
  RAGAdviser, UserAnswers, UseCase` without knowing the sub-module
  layout.
- **`tests/test_public_api.py`** — an import smoke test that walks
  `__all__` and asserts every name resolves, plus an end-to-end
  recommendation using only top-level imports. If a rename drops a
  symbol from the public surface, this fails so the docs and code
  stay in sync.
- **README "Claude Code Skill" section** — moved and expanded from
  the old "Use It From Claude Code" one-paragraph note. Shows the
  `/rag-advisor <folder> [constraints]` invocation, a handful of
  representative examples, and points at the SKILL.md for the full
  constraint-to-flag mapping. Notes the skill inherits the BYOK
  privacy semantics from the CLI.
- **README "Python API" section** — covers the import pattern, a
  full end-to-end example, a table of every attribute on
  `Recommendations`, sub-recommender composition (`DocumentAnalyzer`,
  `ApproachAnalyzer`, `HFModelFinder`, `CostEstimator`,
  `load_sizing_preset`), a "using it from tests" snippet, and an
  explicit stable-vs-internal boundary statement (top-level =
  stable, sub-modules = internal, use at own risk).

### Changed
- **Tagline / positioning.** Was "RAG Advisor is a CLI tool that
  analyzes …". Now framed as three consumption surfaces: CLI,
  Claude Code skill, Python library. Matches reality and matches
  what the follow-up reviewer noticed.

### Notes
- No behaviour change. This is a documentation + surface release —
  the recommender itself, sizing model, format taxonomy, cost
  estimator, presets, tests are unchanged from 0.6.1. Existing
  callers of the sub-module imports keep working (nothing was
  moved), and the CLI is unchanged.

## [0.6.1] - 2026-09-17

Follow-up review of 0.6.0 (`_memoria/rag_advisor_followup_review_v060.md`)
caught concrete bugs and one design regression that this patch closes.

### Fixed
- **HTML report was missing `citation_granularity` and `error_cost`**
  from its User Input Summary, while Markdown / YAML / JSON had them.
  The equivalence test used a hand-curated field list and did not
  catch this. Test now walks every enum on `UserAnswers` and asserts
  its string value appears in every rendered format, so any future
  questionnaire field must be threaded through all four renderers or
  fail loudly.
- **`pip install` line silently dropped packages after `python-pptx`.**
  0.3.3 wrote `"python-pptx  # applied after LibreOffice conversion"`
  into the pip_packages list; the checklist concatenates those into a
  shell command where `#` starts a comment, so `faster-whisper` and
  any following package would not be installed by a copy-paste. The
  explanatory text is now a separate note; pip_packages contain
  package names only.
- **"Other" modality's extension list included recognised text
  formats.** `_modality_of_extension` in the modality recommender
  defaulted to `"other"` for unknown extensions and did not know
  about the 0.5.0 text-format promotion, so `.epub` / `.docx` /
  `.mobi` / `.rtf` / `.html` / `.djvu` / `.chm` / `.opf` were listed
  as members of the Other bucket even when zero files actually landed
  there. Recognised text extensions now short-circuit to `"document"`.
- **`estimated_capacity: "~100,000,000 documents"`** removed from the
  machine config schema. The 0.4.0 release softened the *reason*
  string but forgot the field. Delete it: real capacity depends on
  vector dimension, HNSW parameters and storage mode, which is what
  `--sizing-preset` encodes.
- **Missing RAM feasibility warning.** The 0.6.0 Books report showed
  a 15.4 GB index on a 16 GB host with an empty warnings list. When
  `index_memory_mb >= 70%` of `ram_gb × 1024` and vectors aren't
  mmap'd on-disk, the cost estimator now raises a `CRITICAL:` warning
  naming the shortfall and pointing at `--sizing-preset
  gpu-fp16-quantized` or `--sizing-preset on-disk-mmap`. 50–70% share
  emits a softer note.
- **Chunking wording caught up with 0.4.0's other softening.**
  `"Smaller chunks preferred for precise Q&A retrieval"` (still
  categorical) is replaced with "heuristic starting point; sweep
  {256, 512, 1024} with `ragadvisor evaluate` on your own ground
  truth". Summarization's larger-chunk note gets the same treatment.

### Changed
- **`error_cost` sets calibration intent, not a raw threshold.** The
  0.6.0 wiring turned `wrong_worse` into `similarity_threshold >=
  0.80` and `no_answer_worse` into `similarity_threshold == 0.0`.
  The follow-up review correctly flagged this as the same
  false-precision problem the review had originally raised: similarity
  distributions are model- and corpus-dependent, so no universal
  cosine value encodes "precision-first". `RetrievalRecommendation`
  now carries `calibration_target` (`maximize_precision` /
  `maximize_recall` / `balanced`) and `abstention_policy`
  (`conservative` / `permissive`); the actual numerical threshold is
  the user's calibration output, not a hardcoded default. Both fields
  render in Markdown / HTML / YAML / JSON when set. `top_k` is still
  nudged upward for `no_answer_worse` — that's a knob a caller can
  legitimately preset ahead of evaluation.

### Not addressed here
- **`confidence: 85%` still reads probabilistic** — the reviewer
  suggests renaming to `heuristic_fit_score` or `signals_matched: 5/5`.
  Sensible, but deferred pending a decision on whether to keep the
  field name (breaking JSON/YAML consumers) or add a sibling field.
- **Reranker still surfaces an incoherent compromise** on CPU +
  moderate latency instead of proposing alternative coherent
  configurations (shorter chunks / larger reranker window / no
  rerank). The reviewer's "turn compatibility check into architecture
  search" is a real feature; scoping it for a later release.

## [0.6.0] - 2026-09-17

### Changed
- **Approach decision is now a comparator, not a single winner.**
  `ApproachAnalyzer.assess()` scores every applicable approach (RAG,
  Direct Context, Long-Context LLM, Text-to-SQL, Structured
  Extraction, Full-Text Search) and picks the highest-confidence one.
  The top pick still lives on `recommended_approach` for
  back-compatibility, but every candidate the tool considered is
  exposed on `ApproachAssessment.candidates` and rendered as an
  "Approaches considered" table in every output format. Downstream
  candidates are load-bearing: they let a reader see that RAG at 0.85
  beat Direct Context at 0.72 rather than "0.9 because the code
  literal said so".
  - The rule-specific confidences (previously 0.65 / 0.70 / 0.80 /
    0.85 / 0.90 depending on the branch) have been rebalanced to sit
    in a [0.86, 0.92] band that always beats "all RAG signals
    matched" (now capped at 0.85), so a specific diagnostic rule
    still wins when it fires without needing a first-match-wins fall-
    through.

### Added
- **Two new questionnaire fields with load-bearing downstream effects:**
  - `citation_granularity` (`document` / `page` / `span`) — declares
    how precisely answers must cite their source. Rendered in the
    user-input summary; ground-truth schema evolution to actually
    consume this ships in a later release once vector-store metadata
    is threaded through.
  - `error_cost` (`wrong_worse` / `equal` / `no_answer_worse`) —
    flips the precision/recall balance in retrieval defaults.
    `wrong_worse` tightens `similarity_threshold` to 0.80 and adds
    an abstention hint to the report; `no_answer_worse` drops the
    threshold to 0 and widens `top_k` to at least 8.
- **"Reading the report" section** in the README, covering the
  epistemic conventions (heuristic starting points vs measurements,
  what confidence bands mean, when to run `--validate`, how to
  interpret the `~` prefix on extrapolated numbers, format
  equivalence across MD/HTML/YAML/JSON).

### Deferred
- Passage-level ground-truth schema (`{query, document_id, page,
  section, relevant_spans}`) with matching passage-aware Hit@K/MRR.
  Adding the schema without the metric-side threading would be visual
  clutter; adding the metric-side needs vector-store metadata plumbing
  that belongs in its own release. Keep an eye on §11 of the 2026-09-16
  review — this remains the single largest evaluation gap.
- RAGAS / ARES faithfulness metrics (grounding, unsupported-assertion
  rate, citation correctness) — a whole new evaluator. Same reason:
  ships when it can ship coherently.
- The other ~17 questionnaire fields the review's §12 enumerates.
  Most only pay off when a downstream rule reads them; adding them
  without those rules is just extra prompts.

## [0.5.0] - 2026-09-17

The physical-truth release: corpus estimates that actually reflect the
corpus, footprint numbers that match the deployment, and a format
taxonomy that stops silently dropping ~15% of a Books folder into a
useless "other" bucket. Addresses phases 4, 5 and 6 of the 2026-09-16
review remediation plan.

### Added

- **Physical sizing presets.** `--sizing-preset {cpu-balanced,
  gpu-fp16, gpu-fp16-quantized, on-disk-mmap}` selects a coherent
  bundle of vector datatype, HNSW graph parameters, quantization mode
  and on-disk-vs-mmap mode. The cost estimator now uses these to
  compute the physical footprint: fp16 halves index RAM vs. fp32, int8
  scalar quantization compresses to ~1 byte per dimension, and mmap
  mode keeps only the HNSW graph resident (~44% RAM saving at M=16).
  Custom bundles live in `~/.ragadvisor/presets/sizing/*.yaml`.

  **Why preset-only, not individual flags.** These knobs interact —
  int8 on a low-M HNSW graph hurts recall differently from int8 on
  high-M, and an mmap-on-disk deployment flips the entire RAM/disk
  trade-off. Named bundles encode combinations known to be coherent
  and let users version-track experiments in a repo. Individual
  flags would trivially express bad combinations. Ship curated
  presets first; expose overrides later if the demand appears.

- **Rich text formats promoted to first-class documents.** `.epub`,
  `.mobi`, `.doc`, `.docx`, `.rtf`, `.html` / `.htm`, `.djvu` / `.djv`,
  `.chm`, `.opf` are extracted as text documents rather than dropped
  into an unrecognised "other" bucket. Lazy imports keep the base
  install lean; the optional `[formats]` extra pulls in `ebooklib`,
  `beautifulsoup4` and `striprtf` for pure-Python extractors. Files
  whose extractors aren't available are still counted; only their
  token estimate degrades.

- **Adaptive stratified sampling.** The fixed 50-file sample cap
  regardless of corpus size (the source of "45 of 5,963 files opened"
  on your Books scan) has been replaced with a sample that scales to
  5% of the corpus, floored at 50 and capped at 500. Sampling is
  stratified by file extension so a corpus of 5,000 PDFs and 800
  EPUBs contributes proportionally to the token estimate.

- **Distributional token stats.** The report now shows `p50 / p75 /
  p90 / p95 / p99 / max` alongside the mean, so heavy tails are
  visible instead of hiding inside "average 115K tokens/doc".

- **Wilson 95% confidence interval on the scanned-PDF rate.** The
  "5 of 50 sampled → 10%" figure becomes "10% (95% CI 3%–22%)" so
  extrapolating to a 592-scanned-PDF total no longer looks precise.

- **`unsupported` modality bucket** for formats we recognise but can't
  turn into text (`.gp`, `.ptb` guitar tabs — expand as needed). The
  recommender emits a short ingestion note ("convert with MuseScore
  or exclude") instead of dumping these into `other`.

- **Filename-fragment warnings.** Files like `Author.machine learning
  paradigms` used to leak into `file_types` and create false
  modalities. The analyzer now flags any suffix that contains a space,
  is longer than 6 characters, or starts with a double-dot, excludes
  them from the inventory, and lists the offending paths as a
  `FILENAME FRAGMENTS` warning.

### Changed

- **Index footprint formula** upgraded from a fixed `dim × 4 × 1.5`
  (fp32 vectors + fixed 50% overhead) to `raw_vectors_bytes +
  hnsw_overhead(M) × raw_vectors_bytes + payload`, where HNSW
  overhead scales with the graph degree (`0.3 + 0.06 × M`). Real
  deployments show ~130% overhead at M=16, not 50% — the pre-0.5.0
  RAM figures were optimistic. The two `test_cost_estimator` bounds
  that pinned the old formula have been updated accordingly.

- **Ignored extensions expanded** with `.crdownload`, `.lnk`, `.msi`,
  `.db`, `.bin`, `.dat`, `.iso`, `.img`, `.dmg`, `.pkl`, `.pickle`,
  `.npy`, `.npz`, `.pt`, `.pth`, `.safetensors`, `.ckpt`, `.woff`,
  `.woff2`, `.ttf`, `.otf`, `.eot`. These were all landing in the
  `other` bucket and inflating "N unrecognised files" counts. They
  are now silently excluded from token totals and modality reports —
  expect a small drop in the corpus-token estimate on existing users'
  next run.

### Migration

- **`--sizing-preset` is optional.** Users who don't pass it get the
  `cpu-balanced` defaults (fp32, M=16, no quantization, in-memory)
  which match pre-0.5.0 assumptions. Existing scripts keep working.
- **`[formats]` extra is optional.** `pip install "ragadvisor[formats]"`
  when you have EPUB / MOBI / RTF corpora and want token estimates
  that see through them. The base install continues to count these
  files as documents; only the sampled token count degrades.

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
