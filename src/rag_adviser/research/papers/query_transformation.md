# Query Transformation Techniques for RAG

## Why Transform Queries?

The user's query and the relevant documents often exist in different semantic spaces. A question like "What causes servers to crash?" won't necessarily match a document that says "Memory leaks lead to out-of-memory errors and process termination." Query transformation bridges this gap.

## Technique 1: Query Expansion

**Problem**: Short, keyword-style queries have low semantic signal for embedding models.

**Solution**: Use an LLM to rewrite the query into a fuller natural language form, or generate multiple query variants.

**Example**:
- Original: "RAG chunking best practices"
- Expanded: "What are the best practices for chunking documents in a Retrieval-Augmented Generation system?"

**Implementation**: Prompt the LLM for 3 query variants (an LCEL chain `prompt | llm | StrOutputParser()`), retrieve for each, and merge with reciprocal rank fusion. LangChain's legacy `MultiQueryRetriever` did the same with a plain union.

**When to use**: Short keyword queries, search-box-style interfaces.

**Latency impact**: ~400ms (one LLM call).

## Technique 2: HyDE (Hypothetical Document Embeddings)

**Problem**: Questions and answers occupy different regions in embedding space. "What is X?" is far from the passage that explains X.

**Solution**: Generate a hypothetical answer to the query using an LLM, then embed that hypothetical answer instead of the query. The hypothetical answer is closer in embedding space to actual relevant documents.

**Reference**: Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"

**Example**:
- Query: "What is chunking in RAG?"
- HyDE generates: "Chunking in RAG is the process of splitting documents into smaller pieces called chunks, typically 256-1024 tokens, so they can be embedded and retrieved individually."
- This hypothetical answer is then embedded and used for similarity search.

**When to use**: Factual Q&A, when query-document semantic gap is large.

**When NOT to use**: Keyword search, multi-turn conversation, very specific queries.

**Latency impact**: ~600ms (one LLM call + one embedding).

## Technique 3: Multi-Query Decomposition

**Problem**: Complex queries that require information from multiple documents or topics.

**Solution**: Decompose the complex query into simpler sub-queries, retrieve for each independently, then merge results.

**Example**:
- Original: "Compare the advantages of ChromaDB vs Qdrant for production use"
- Sub-queries:
  1. "What are the advantages of ChromaDB for production?"
  2. "What are the advantages of Qdrant for production?"
  3. "ChromaDB vs Qdrant feature comparison"

**Implementation**: LCEL chain with a decomposition prompt that emits one sub-question per line; retrieve per sub-question and fuse results.

**When to use**: Comparative queries, multi-hop reasoning, complex analytical questions.

**Latency impact**: ~800-1200ms (one LLM call + N retrievals).

## Technique 4: Step-Back Prompting

**Problem**: Very specific queries that are too narrow to match any document.

**Solution**: Ask a more general version of the question first, retrieve context for that, then answer the specific question with that broader context.

**Reference**: Zheng et al., 2023 — "Take a Step Back: Evoking Reasoning via Abstraction"

**Example**:
- Original: "What is the default chunk overlap in LangChain's RecursiveCharacterTextSplitter?"
- Step-back: "How does LangChain's RecursiveCharacterTextSplitter work and what are its parameters?"

**When to use**: Highly specific questions about details, when direct retrieval misses.

**Latency impact**: ~500ms (one LLM call).

## Technique 5: Conversation Condensation

**Problem**: In multi-turn conversations, the latest message often lacks context. "What about the pricing?" makes no sense without knowing what product was discussed.

**Solution**: Condense the conversation history into a standalone query before retrieval.

**Example**:
- History: "Tell me about ChromaDB" → "Is it open source?" → "What about pricing?"
- Condensed: "What is the pricing model for ChromaDB?"

**Implementation**: An LCEL condense chain that rewrites (history, latest message) into a standalone question before retrieval. This replaces the deprecated `ConversationalRetrievalChain`.

**When to use**: Any multi-turn interface (chatbots, assistants).

**This is not optional for multi-turn RAG** — without condensation, retrieval quality degrades severely after the first turn.

**Latency impact**: ~500ms (one LLM call per turn).

## Technique 6: Reciprocal Rank Fusion (RRF)

**Problem**: Multiple retrieval methods (vector search, BM25, multiple query variants) produce separate ranked lists.

**Solution**: Combine ranked lists using RRF formula: `score(d) = Σ 1/(k + rank_i(d))` where k=60 is the standard constant.

**When to use**: Hybrid search, multi-query retrieval, any scenario with multiple result sets to merge.

**Latency impact**: Negligible (pure computation).

## Technique 7: Reranking with Cross-Encoders

**Problem**: Bi-encoder (embedding model) retrieval is fast but approximate. It compares query and document independently.

**Solution**: Use a cross-encoder to jointly process query-document pairs. Cross-encoders are more accurate but slower (can't be pre-computed).

**Common models**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `BAAI/bge-reranker-v2-m3` (multilingual).

**Pipeline**: Retrieve top-20 with bi-encoder, rerank to top-5 with cross-encoder.

**When to use**: When latency budget allows (adds ~300ms). Not for <500ms latency targets.

**Latency impact**: ~300ms for 20 documents.

## Choosing the Right Techniques

| Query Pattern | Recommended Techniques |
|--------------|----------------------|
| Short keywords | Query Expansion |
| Natural questions (simple) | None or optional HyDE |
| Multi-turn conversation | Conversation Condensation (required) |
| Comparative questions | Multi-Query Decomposition |
| Multi-hop reasoning | Multi-Query Decomposition + Reranking |
| Aggregative ("summarize all X") | Broad retrieval (top_k=20) + Reranking |
| Very specific details | Step-Back Prompting |
| Hybrid keyword + semantic | RRF to merge BM25 + vector results |
