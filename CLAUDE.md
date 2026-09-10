# RAG Advisor — working notes for coding agents

Rule-based CLI that recommends (and, with `--validate`, measures) a RAG configuration.
Package `rag_adviser` under `src/`, CLI entry `ragadvisor`. Python 3.10+.

## Commands

```bash
pip install -e ".[dev]"                 # dev deps (no numpy / sentence-transformers)
pip install -e ".[eval]"                # adds sentence-transformers, chromadb, einops
python -m ruff check src tests          # must be clean (E501 ignored only in modality_recommender.py)
python -m pytest -q                     # ~10 s; numpy tests skip when numpy is absent
ragadvisor example-corpus ./ex && ragadvisor run --no-interactive -d ./ex/corpus \
    --ground-truth-path ./ex/queries.jsonl --validate --privacy strict   # end-to-end smoke
```

## Layout

- `main.py` orchestrates: analysis -> modalities -> approach gate -> constraints -> models ->
  chunking -> hybrid -> vector DB -> retrieval -> reranker -> query pipeline -> estimates ->
  steps -> validation (outside the Rich progress block) -> LLM verification -> reports.
- `analyzers/`, `recommenders/` (model_finder, chunking, hybrid, vector_db, reranker, query,
  modality, cost_estimator), `evaluators/` (pipeline_runner, retrieval_modes, validator,
  chunking_strategies, vector_store, metrics, query_bootstrap), `reporters/` (md/html/yaml/json),
  `input_modes/` (interactive, cli_params, xml_input), `config/defaults.yaml` (catalogues).
- All inputs converge on `models.UserAnswers`; all outputs on `models.Recommendations`.

## Conventions

- Recommendations use tokens for chunk sizes; the evaluator uses characters (x4).
- `defaults.yaml`: `quality_score` = approx. MTEB retrieval nDCG@10 (refresh with
  `ragadvisor refresh-catalogue --write`); reranker `latency_ms_per_pair_cpu` are conservative
  measured priors. Never invent precise benchmark numbers; mark approximations as such.
- Never send user documents to an API; only recommendations (verification) or sampled
  passages (`bootstrap-queries`) with the user's own key.
- Tests: add a regression test for every bug fixed; pipeline tests patch
  `EvalPipelineRunner._load_embedding_model` with a fake model; CLI tests use
  `CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})`.
- XML template `examples/answers.example.xml` is parsed by a test; XML comments cannot contain `--`.
- Commit identity: `aelena <antonioelena@gmail.com>` (repo git config). Keep `CHANGELOG.md`
  and `README.md` current in the same commit as the feature.

## Releasing

Bump `version` in `pyproject.toml`, `src/rag_adviser/__init__.py`, the assertion in
`tests/test_cli.py` and the README badge; add a CHANGELOG section; commit; then
`git tag -a vX.Y.Z -m "RAG Advisor X.Y.Z" && git push origin main vX.Y.Z`.
`release.yml` builds, checks tag == version, publishes to PyPI (trusted publishing,
environment `pypi`) and creates the GitHub release. PyPI shows the README from the uploaded
package, so README-only changes need a new version.
