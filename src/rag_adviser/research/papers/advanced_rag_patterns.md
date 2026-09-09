# Advanced RAG Patterns and Best Practices

## RAG Architecture Patterns

### Naive RAG
The simplest pipeline: embed query → retrieve top-k → stuff into prompt → generate.
- Works for simple Q&A over clean documents
- Fails for complex queries, noisy data, or nuanced answers

### Advanced RAG
Adds pre-retrieval and post-retrieval processing:
- **Pre-retrieval**: Query transformation, routing, classification
- **Retrieval**: Hybrid search, multi-index, filtered retrieval
- **Post-retrieval**: Reranking, compression, deduplication
- **Generation**: Prompt engineering, chain-of-thought, citation

### Modular RAG
Treats each component as a swappable module. Key modules:
- Query module (transform, route, classify)
- Retrieval module (vector, BM25, graph, SQL)
- Reranking module (cross-encoder, LLM-based)
- Generation module (prompt strategy, output format)

### Agentic RAG
Uses an LLM agent to dynamically decide:
- Whether to retrieve at all (some questions don't need retrieval)
- Which retrieval strategy to use
- Whether the retrieved context is sufficient or needs more searching
- When to give up and say "I don't know"

Reference: LlamaIndex's ReAct Agent, LangChain's AgentExecutor

## Prompt Strategies for RAG

### Stuff
Put all retrieved chunks directly into the prompt. Simple and effective for small context.
- **When**: < 5 chunks, or chunks are small
- **Limitation**: Fails when total context exceeds LLM's context window

### Map-Reduce
Process each chunk independently, then combine results.
- **Map step**: Apply prompt to each chunk separately
- **Reduce step**: Combine intermediate answers into final answer
- **When**: Summarization, aggregative queries, large number of chunks

### Refine
Iteratively refine the answer by processing chunks one at a time.
- Process chunk 1 → initial answer
- Process chunk 2 + previous answer → refined answer
- Continue until all chunks processed
- **When**: Legal analysis, detailed answers, when order matters

### Map-Rerank
Process each chunk independently, score each answer, return the highest-scored one.
- **When**: When only one chunk likely contains the answer

## Evaluation Frameworks

### RAGAS (Retrieval Augmented Generation Assessment)
Open-source framework for evaluating RAG pipelines. Metrics:
- **Faithfulness**: Is the answer supported by the context? (LLM-evaluated)
- **Answer Relevancy**: Does the answer address the question? (LLM-evaluated)
- **Context Precision**: Are the retrieved chunks relevant? (LLM-evaluated)
- **Context Recall**: Was all relevant information retrieved? (LLM-evaluated)

### Custom Evaluation
For production systems, build custom evaluation using:
- Ground truth Q&A pairs from domain experts
- Hit rate and MRR for retrieval quality
- Human evaluation for answer quality
- A/B testing for comparing configurations

### Evaluation Best Practices
1. Create at least 50-100 ground truth Q&A pairs
2. Cover edge cases: multi-hop, negation, out-of-scope queries
3. Include queries that SHOULD return "I don't know"
4. Evaluate retrieval and generation separately
5. Track metrics over time as data changes

## Common RAG Failures and Fixes

### Problem: Low retrieval quality
- **Symptom**: Correct answer exists in corpus but isn't retrieved
- **Fixes**: Try different embedding models, adjust chunk sizes, add query transformation, use hybrid search

### Problem: Hallucination despite good retrieval
- **Symptom**: Retrieved context is correct but answer is fabricated
- **Fixes**: Lower temperature, use "refine" strategy, add explicit instructions to only use provided context, add citation requirements

### Problem: Chunk boundaries split relevant info
- **Symptom**: Answer requires info from multiple consecutive chunks but only one is retrieved
- **Fixes**: Increase chunk overlap, use hierarchical chunking, increase top_k, use semantic chunking

### Problem: Too much irrelevant context
- **Symptom**: Retrieved chunks are marginally relevant, diluting the answer
- **Fixes**: Increase similarity threshold, add reranking, reduce top_k, improve chunking granularity

### Problem: Stale or contradictory information
- **Symptom**: Retrieved chunks contain outdated info that contradicts newer documents
- **Fixes**: Add document timestamps to metadata, filter by recency, use metadata-aware retrieval

## Production Checklist

1. **Chunking**: Test at least 3 strategies on your actual data
2. **Embedding model**: Evaluate on your domain, not just MTEB scores
3. **Vector DB**: Start embedded (ChromaDB), move to client-server for production
4. **Retrieval**: Always test with and without reranking
5. **Monitoring**: Log queries, retrieved chunks, and user feedback
6. **Evaluation**: Maintain a growing ground truth set, run evals on every change
7. **Guardrails**: Handle out-of-scope queries, add source citations
8. **Updates**: Plan for document updates and re-indexing
