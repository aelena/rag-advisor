# RAG vs Long Context: When to Use Which

## Retrieval Meets Long Context Large Language Models

**Reference**: Xu et al., 2024 — arXiv:2407.14482

### Key Question

With LLMs now supporting 128K-1M+ token context windows, is RAG still necessary? Can we just stuff all documents into the prompt?

### Main Findings

1. **Long-context LLMs can outperform RAG** when the corpus is small enough to fit in the context window and the task is straightforward Q&A.

2. **RAG still wins** when:
   - The corpus exceeds the context window
   - The task requires precise retrieval from a large corpus
   - Cost efficiency matters (processing 100K tokens per query is expensive)
   - The task involves specific factual lookups rather than holistic understanding

3. **RAG + long context together is often best**: Retrieve top-k chunks (more than usual, e.g., top-30) and let the long-context model process them all. This combines retrieval precision with the model's ability to reason over longer context.

4. **Long context degrades with corpus size**: As more documents are stuffed into context, accuracy drops — the "needle in a haystack" problem. RAG's focused retrieval avoids this.

### Cost Analysis

| Approach | Tokens per Query | Relative Cost |
|----------|-----------------|---------------|
| RAG (top-5, 512 tokens each) | ~2,500 + prompt | 1× |
| RAG (top-20, 512 tokens each) | ~10,000 + prompt | 4× |
| Long context (50K corpus) | ~50,000 + prompt | 20× |
| Long context (200K corpus) | ~200,000 + prompt | 80× |

### The "Lost in the Middle" Problem

**Reference**: Liu et al., 2023 — "Lost in the Middle: How Language Models Use Long Contexts"

- LLMs attend strongly to the beginning and end of the context
- Information in the middle of a long context is often ignored
- This means stuffing more context can actually **hurt** performance
- RAG avoids this by providing only the most relevant chunks
- If using long context, place the most relevant information at the beginning

## Decision Framework: RAG vs Long Context vs Hybrid

### Use Pure RAG When:
- Corpus > 200K tokens (exceeds practical context windows)
- Corpus changes frequently (re-indexing is cheaper than re-processing)
- Need source attribution (RAG naturally tracks which chunks were used)
- Cost-sensitive (per-query cost must stay low)
- Latency-sensitive (processing 100K tokens takes time)
- High query volume (cost multiplies with each query)
- Precise factual retrieval (specific answers from specific documents)

### Use Long Context (No RAG) When:
- Corpus < 50K tokens (fits comfortably in context)
- Corpus is static (same documents for every query)
- Task requires holistic understanding (summarization, theme analysis)
- Few queries (cost per query is acceptable)
- Documents are highly interconnected (hard to chunk meaningfully)
- Setup simplicity matters (no vector DB, no embeddings, no indexing)

### Use Hybrid (RAG + Long Context) When:
- Corpus is medium-sized (50K-500K tokens)
- Task requires both precise retrieval and cross-document reasoning
- Can afford the compute cost
- Quality is the top priority over cost/latency

### Hybrid Implementation

```
# Retrieve more chunks than usual
results = retrieve(query, top_k=30)

# Let the long-context model process them all
# Sort by relevance so best matches are at the start (avoid "lost in the middle")
sorted_chunks = sorted(results, key=lambda x: -x.score)

# Build prompt with all chunks
context = "\n\n---\n\n".join(chunk.text for chunk in sorted_chunks)
prompt = f"Context:\n{context}\n\nQuestion: {query}"
```

## Practical Thresholds

Based on the research and practical experience:

| Corpus Size | Recommended Approach |
|-------------|---------------------|
| < 10K tokens | Direct context stuffing, no RAG needed |
| 10K - 50K tokens | Either approach works; long context is simpler |
| 50K - 200K tokens | RAG or hybrid; long context still possible but costly |
| 200K - 1M tokens | RAG strongly recommended; hybrid for quality-critical tasks |
| > 1M tokens | RAG required; long context physically impossible |

## Context Window Sizes (as of 2024-2025)

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128K tokens |
| Claude 3.5 / Claude 4 | 200K tokens |
| Gemini 1.5 Pro | 1M-2M tokens |
| Llama 3 | 8K-128K tokens (varies) |
| Mistral Large | 128K tokens |
| Command R+ | 128K tokens |

**Note**: Advertised context window ≠ effective context window. Most models degrade significantly when using >50% of their stated limit. Test with your actual data.

## Emerging Pattern: Contextual Retrieval

**Reference**: Anthropic, 2024 — "Contextual Retrieval"

A hybrid approach where:
1. Each chunk is augmented with context about where it sits in the document
2. An LLM generates a brief contextual summary prepended to each chunk
3. This reduces the semantic gap between query and chunk
4. Combined with BM25 + embeddings, reduces retrieval failures by ~67%

**Implementation**:
- Before indexing, for each chunk, prompt the LLM: "Given this document, provide a short context for this chunk"
- Prepend the context to the chunk text before embedding
- This is a one-time indexing cost, not a per-query cost
- Particularly effective for chunks that lose meaning without surrounding context
