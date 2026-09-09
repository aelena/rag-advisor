---
name: rag-advisor
description: Analyze a document folder and recommend a complete RAG configuration (or advise against RAG) using the ragadvisor CLI in this repo. Use when the user points at a corpus folder and asks how to build RAG for it, which embedding model / chunking / vector DB / reranker to use, whether RAG is the right approach, or to validate a RAG setup against ground-truth queries. Invoked as /rag-advisor <folder> [constraints].
---

# RAG Advisor skill

Turn a folder of documents into a reviewed RAG recommendation by driving the
`ragadvisor` CLI, then explain the result in plain language.

## Prerequisites

- Run from the ragadvisor repository root, or anywhere after `pip install -e .`
  in that repo. Check with `ragadvisor version`; if missing, install it first.
- For `--validate`, the `[eval]` extra must be installed (`pip install -e ".[eval]"`).
- Never send the user's documents anywhere. The CLI only reads them locally; the
  optional `--use-llm` review sends *recommendations*, not documents.

## Workflow

1. **Inventory the corpus** (fast, read-only):

   ```bash
   ragadvisor analyze <folder>
   ```

   Report languages, content type (and its confidence), total/estimated tokens,
   and any non-text modalities (images, video, spreadsheets, CAD, scanned PDFs).

2. **Collect constraints** from what the user already said. Map them to flags;
   ask only for what is missing *and* changes the outcome (privacy, deployment,
   latency budget, use case). Sensible defaults otherwise.

   | User says | Flag |
   |-----------|------|
   | "must stay on-prem / no data leaves" | `--privacy strict` (or `air_gapped`) `--environment on_prem` |
   | "we can use OpenAI/Cohere" | `--budget paid_api --privacy none` |
   | "chatbot / Q&A over docs" | `--use-case question_answering` |
   | "search box" | `--use-case semantic_search` |
   | "codebase" | `--use-case code_assistance --content-type code` |
   | "contracts / legal" | `--use-case legal_analysis` |
   | "needs to feel instant" | `--latency "<500ms"` |
   | "nightly batch" | `--latency batch` |
   | "we have a GPU with N GB" | `--hardware gpu_available --vram-gb N` |
   | "users type keywords" | `--query-type short_keywords` |
   | "multi-turn chat" | `--query-type multi_turn` |
   | "documents change daily" | `--update-frequency daily` |
   | "we'll add languages later" | `--future-languages` |

   Presets (`ragadvisor presets`) are a shortcut: `--preset legal-discovery`
   etc.; explicit flags override preset values.

3. **Run the adviser** non-interactively and write reports to a scratch folder:

   ```bash
   ragadvisor run --no-interactive -d <folder> <flags> -f all -o <scratch>/rag_report
   ```

   If the user has ground-truth queries (JSONL `{"query": ..., "relevant_docs": [...]}`
   or CSV), add `--ground-truth-path <file> --validate` so the recommendation
   is measured on their data.

4. **Read `rag_report.md`** (not the terminal output) and summarize for the user:

   - Lead with the **approach verdict**. If RAG is not recommended (direct
     context, Text-to-SQL, full-text search, structured extraction), say so first
     and explain why; only then mention the RAG config as a fallback.
   - Then the concrete stack: embedding model (and why), chunking strategy and
     size, vector DB, retrieval settings, hybrid search, reranker, query pipeline.
   - Call out every **warning** in the report verbatim in spirit: multimodal
     share, scanned PDFs, low content-type confidence, sampled token estimates,
     license or trust_remote_code caveats.
   - If validation ran, quote the verdict and metrics and the suggested next steps.
   - Point to `rag_config.yaml` as the machine-readable hand-off.

5. **Offer next steps**, not more analysis: run `--validate` if it did not run,
   `ragadvisor evaluate` to compare chunking strategies, or `--use-llm` for a
   second opinion.

## Rules

- Do not paraphrase numbers from memory; copy them from the report.
- Do not invent flags. `ragadvisor run --help` is authoritative.
- If `analyze` shows a mostly non-text corpus, lead with the modality plan;
  the text pipeline is secondary there.
- Keep the user's documents and report inside their machine and scratch folder.
