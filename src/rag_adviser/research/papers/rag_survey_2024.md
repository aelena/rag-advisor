# Retrieval-Augmented Generation for Large Language Models: A Survey

**Reference**: Gao et al., 2024 — arXiv:2312.10997

## Taxonomy of RAG Approaches

The survey categorizes RAG into three paradigms representing its evolution:

### 1. Naive RAG (Retrieve-Read)
- Pipeline: Index → Retrieve → Generate
- **Indexing**: Chunk documents, embed, store in vector DB
- **Retrieval**: Embed query, find top-k similar chunks
- **Generation**: Stuff chunks into prompt, generate answer

**Known problems with Naive RAG**:
- Retrieval: Low precision (irrelevant chunks), low recall (misses relevant ones)
- Generation: Hallucination even with correct context, irrelevance, toxicity
- Augmentation: Redundancy across chunks, importance ordering lost, context window overflow

### 2. Advanced RAG
Adds optimization before and after retrieval:

**Pre-retrieval optimizations**:
- Query rewriting and expansion
- Query routing to appropriate indices
- Query decomposition for complex questions

**Retrieval optimizations**:
- Hybrid search (dense + sparse)
- Metadata filtering
- Sentence-window retrieval (retrieve sentence, expand to surrounding window)
- Auto-merging retrieval (retrieve child, return parent if enough children match)

**Post-retrieval optimizations**:
- Reranking with cross-encoders
- Context compression (remove irrelevant sentences)
- Deduplication of overlapping chunks

### 3. Modular RAG
Treats each component as an interchangeable module:
- Search module (can query web, SQL, knowledge graphs, vector stores)
- Memory module (maintains conversation context and retrieved history)
- Routing module (decides which retrieval strategy to use)
- Predict module (generates hypothetical docs for retrieval — HyDE)
- Task adapter module (adjusts behavior per task type)

## Retrieval Source Comparison

| Source | Best For | Latency | Freshness |
|--------|----------|---------|-----------|
| Vector store | Semantic similarity over own corpus | Low | Manual re-index |
| Web search | Current events, broad knowledge | Medium | Real-time |
| Knowledge graph | Entity relationships, structured queries | Low | Manual update |
| SQL database | Structured data, aggregations | Low | Real-time |
| Code repository | Code search, API lookup | Medium | Git-synced |

## Retrieval Granularity Findings

The survey compiles findings on optimal retrieval granularity:

- **Token-level**: Too fine-grained, noisy. Used in kNN-LM research, not practical RAG.
- **Sentence-level**: High precision but lacks context. Good when combined with sentence-window expansion.
- **Chunk-level (100-512 tokens)**: The sweet spot for most applications. Balances precision and context.
- **Document-level**: Too coarse for retrieval but useful as a reranking/filtering stage.
- **Proposition-level**: Decompose documents into atomic facts. High precision, emerging approach.

**Recommendation from survey**: Use chunk-level retrieval (256-512 tokens) with sentence-window expansion or parent-document retrieval for context.

## When RAG Outperforms Fine-Tuning

The survey identifies scenarios where each approach wins:

**RAG preferred**:
- Knowledge changes frequently
- Need for source attribution and citations
- Domain has clear, retrievable documents
- Avoiding hallucination is critical
- Limited training data or compute for fine-tuning

**Fine-tuning preferred**:
- Task requires specific output format or style
- Domain knowledge is implicit (hard to retrieve)
- Latency requirements preclude retrieval
- Task is more about reasoning than knowledge

**Both together** (RAG + fine-tuned model):
- Best results in most benchmarks
- Fine-tune the model to be a better "reader" of retrieved context
- Fine-tune the retriever on domain-specific queries

## Evaluation Framework Summary

The survey consolidates RAG evaluation into:

### Retrieval Quality
- **Context Precision**: What fraction of retrieved chunks are relevant?
- **Context Recall**: What fraction of relevant chunks were retrieved?
- **MRR (Mean Reciprocal Rank)**: How high is the first relevant result?

### Generation Quality
- **Faithfulness**: Does the answer only use information from the context?
- **Answer Relevancy**: Does the answer address the question?
- **Answer Correctness**: Is the answer factually correct?

### End-to-End
- **Noise Robustness**: Can the system handle irrelevant retrieved context?
- **Negative Rejection**: Does the system refuse to answer when context is insufficient?
- **Counterfactual Robustness**: Does the system detect contradictions in context?
- **Information Integration**: Can the system synthesize across multiple chunks?

## Key Recommendations from the Survey

1. **Always implement hybrid search** — dense retrieval alone misses keyword matches; BM25 alone misses semantic similarity. Combining via RRF consistently outperforms either alone.

2. **Reranking is the highest-ROI optimization** — adding a cross-encoder reranker to top-20 results improves precision significantly with modest latency cost (~300ms).

3. **Query transformation matters more than people think** — the gap between user query and document content is a primary failure mode. Even simple query expansion helps.

4. **Chunk size should match your content** — don't use one size for everything. Technical docs need smaller chunks (256 tokens), narrative content can use larger ones (512-1024 tokens).

5. **Evaluate retrieval and generation separately** — a bad answer might be a retrieval problem or a generation problem. Separate evaluation tells you where to invest effort.
