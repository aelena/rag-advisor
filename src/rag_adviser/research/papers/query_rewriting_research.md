# Query Rewriting and HyDE: Bridging the Query-Document Gap

## Query Rewriting for Retrieval-Augmented Large Language Models

**Reference**: Ma et al., 2023 — arXiv:2305.14283

### The Problem

User queries are often:
- Ambiguous: "How do I fix it?" (fix what?)
- Under-specified: "best database" (for what use case?)
- Keyword-style: "RAG chunking overlap" (not a proper question)
- Conversational: "What about the pricing?" (needs conversation context)

These queries don't match the language used in documents. A document might say "configuring chunk overlap parameters for optimal retrieval" while the user searches "chunking overlap."

### Rewrite-Retrieve-Read Pipeline

Instead of the standard Retrieve-Read pipeline:
1. **Rewrite**: Use an LLM to transform the query into a better retrieval query
2. **Retrieve**: Search with the rewritten query
3. **Read**: Generate the answer from retrieved context

### Rewriting Strategies Evaluated

#### Strategy 1: Query Clarification
Rewrite the query to be more specific and self-contained.

**Prompt template**:
```
Given the user's query, rewrite it to be more specific and self-contained
for searching a document collection.

Original query: {query}
Rewritten query:
```

**Example**:
- Input: "how to split docs"
- Output: "What are the methods for splitting documents into chunks for retrieval-augmented generation?"

#### Strategy 2: Query Decomposition
Break complex queries into simpler sub-queries.

**Example**:
- Input: "Compare FAISS and ChromaDB performance for million-document collections with real-time updates"
- Output:
  1. "What is FAISS performance at million-document scale?"
  2. "What is ChromaDB performance at million-document scale?"
  3. "Does FAISS support real-time document updates?"
  4. "Does ChromaDB support real-time document updates?"

#### Strategy 3: Keyword Extraction
Extract key terms and concepts for hybrid search.

**Example**:
- Input: "I'm building a chatbot for our legal department and need to know what chunk size works best"
- Keywords: "chunk size", "legal documents", "chatbot", "RAG configuration"

### Key Finding: Trainable Rewriter

The paper shows that fine-tuning a small model (T5-base) as a dedicated query rewriter outperforms prompting a large LLM:
- Trained on (original_query, better_query) pairs
- "Better query" defined as: the query that retrieves the correct documents
- Small model (220M params) adds only ~50ms latency vs. ~500ms for an LLM call

**Practical takeaway**: For production systems with high query volume, consider training a small dedicated rewriter rather than calling an LLM for every query.

## Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)

**Reference**: Gao et al., 2022 — arXiv:2212.10496

### Core Insight

Queries and relevant documents live in different regions of embedding space. "What is chunking?" (a question) is semantically far from a passage that explains chunking (a statement).

**Solution**: Generate a hypothetical answer to the query, then embed that hypothetical answer instead of the query. The hypothetical answer, even if inaccurate, is closer in embedding space to the real answer.

### How HyDE Works

1. User asks: "What chunk size should I use for legal documents?"
2. LLM generates hypothetical answer: "For legal documents, a chunk size of 512-1024 tokens is recommended. Legal text contains long sentences and cross-references, so smaller chunks risk splitting important clauses. Use paragraph-level chunking with headers as split points."
3. Embed the hypothetical answer (not the original query)
4. Retrieve documents similar to the hypothetical answer
5. Generate the real answer from retrieved documents

### When HyDE Helps vs Hurts

**HyDE helps when**:
- Questions are short or keyword-style
- There's a clear query-document semantic gap
- Documents are expository/factual (encyclopedic style)
- The LLM has reasonable parametric knowledge about the topic

**HyDE hurts when**:
- The LLM's hypothetical answer is confidently wrong (leads retrieval astray)
- Queries are already well-formed and specific
- Documents use very different terminology than the LLM expects
- Multi-turn conversations (the hypothetical answer ignores conversation context)
- Highly specialized domains the LLM knows nothing about

### HyDE Variants

#### Multi-HyDE
Generate N hypothetical answers (N=3-5), embed each, retrieve for each, merge results with RRF.

**Benefit**: More robust — one bad hypothetical doesn't dominate.
**Cost**: N× LLM generation calls.

#### Domain-Specific HyDE
Customize the generation prompt to match your domain's style:
```
Generate a passage from a legal brief that would answer this question: {query}
```

This produces hypothetical answers closer to your actual document style.

#### HyDE + Reranking
1. HyDE retrieval gets top-20
2. Cross-encoder reranks using the **original query** (not the hypothetical answer)
3. This corrects for cases where HyDE retrieval drifts from the user's intent

### Implementation Considerations

**Latency budget**:
- LLM call for hypothetical answer: 300-800ms
- Embedding the hypothetical: 10-50ms
- Total added latency: ~400-900ms

**When to skip HyDE dynamically**:
```python
# Skip HyDE for queries that are already document-like (long, specific)
if len(query.split()) > 15 and not query.endswith("?"):
    # Query is detailed enough, use directly
    results = retrieve(query)
else:
    # Short or question-form query, use HyDE
    hypothetical = generate_hypothetical(query)
    results = retrieve(hypothetical)
```

## Combining Query Transformation Techniques

### Recommended Pipelines by Query Type

**Simple factual queries** ("What is X?"):
```
query → [optional HyDE] → retrieve(top_k=5) → generate
```

**Multi-turn conversation**:
```
history + query → condense_to_standalone → retrieve(top_k=5) → generate
```

**Complex analytical queries** ("Compare X and Y for Z"):
```
query → decompose(3 sub-queries) → retrieve_each(top_k=5) → RRF_merge → rerank(top_5) → generate
```

**Broad aggregative queries** ("Summarize all approaches to X"):
```
query → expand(3 variants) → retrieve_each(top_k=10) → RRF_merge → rerank(top_10) → map_reduce_generate
```

**Highly specific queries** ("What is the default value of X in library Y?"):
```
query → step_back(broader_query) → retrieve_both(original + broad, top_k=5 each) → RRF_merge → generate
```

### Latency Budgets

| Technique | Added Latency | Requires LLM Call |
|-----------|--------------|-------------------|
| No transformation | 0ms | No |
| Conversation condensation | 300-500ms | Yes |
| Query expansion | 300-500ms | Yes |
| HyDE | 400-900ms | Yes |
| Multi-query decomposition | 400-600ms (parallel) | Yes |
| Step-back prompting | 300-500ms | Yes |
| Cross-encoder reranking | 200-400ms | No (uses small model) |
| RRF fusion | <10ms | No |

### Cost-Effective Ordering

If budget is limited, add techniques in this order (highest ROI first):
1. **Conversation condensation** (required for multi-turn, massive impact)
2. **Cross-encoder reranking** (no LLM cost, significant quality improvement)
3. **Hybrid search with RRF** (no LLM cost, improves recall)
4. **Query expansion** (moderate cost, helps short queries)
5. **HyDE** (higher cost, situational benefit)
6. **Multi-query decomposition** (highest cost, only for complex queries)
