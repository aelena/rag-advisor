# Practical Chunking Analysis and Proposition-Based Retrieval

## Chunking Strategies for LLM Applications

**Reference**: Kamradt, 2023 — Practical analysis and benchmarking

### Experimental Setup

Kamradt's analysis systematically tests chunking strategies across multiple dimensions:
- Chunk sizes from 128 to 2048 tokens
- Different content types (narrative, technical, legal, conversational)
- Measured retrieval accuracy using ground truth Q&A pairs

### Key Findings

#### 1. Chunk Size Sweet Spot Varies by Content

| Content Type | Optimal Chunk Size | Why |
|-------------|-------------------|-----|
| Technical documentation | 256-512 tokens | Information is dense, self-contained sections |
| Narrative/prose | 512-1024 tokens | Meaning spans longer passages |
| Legal text | 512-768 tokens | Clauses and references need context |
| Chat/conversation | 128-256 tokens | Turns are short, context switches fast |
| Code | Per-function/class | Logical units matter more than token count |
| FAQ/Knowledge base | Per-entry | Each Q&A pair is a natural chunk |

#### 2. Overlap Matters More Than Expected

- 0% overlap: Frequent answer fragmentation — relevant info split across chunks
- 10% overlap: Marginal improvement
- **20% overlap: Sweet spot** — captures most cross-boundary information
- 50% overlap: Diminishing returns, doubles storage and indexing cost

**Recommendation**: Use 15-25% overlap relative to chunk size. For 512-token chunks, use 100-128 token overlap.

#### 3. Splitting Boundaries Are Critical

Ranking of split boundaries from best to worst:
1. **Semantic boundaries** (topic changes, section headers) — best retrieval
2. **Paragraph boundaries** — close second, much simpler to implement
3. **Sentence boundaries** — good, avoids mid-sentence cuts
4. **Fixed token windows** — worst, frequently splits mid-thought

**Practical rule**: Always split at sentence boundaries at minimum. If your content has clear structure (headers, sections), use those as primary split points.

#### 4. Small Chunks + Big Context Pattern

The "sentence window" approach works better than any single chunk size:
1. Index small chunks (1-2 sentences) for precise retrieval
2. At query time, expand retrieved chunks to include surrounding sentences
3. Pass the expanded window to the LLM

This gets the precision of small chunks with the context of large chunks.

### Chunking Pipeline Recommendation

```
1. Parse document structure (headers, paragraphs, lists, code blocks)
2. Split at structural boundaries first
3. If any section > max_chunk_size:
   a. Split at paragraph boundaries
   b. If still too large, split at sentence boundaries
4. Apply overlap between adjacent chunks
5. Attach metadata: source file, section header, page number, position
```

## Dense X Retrieval: Proposition-Based Chunking

**Reference**: Chen et al., 2023 — arXiv:2312.06648

### The Problem with Passage-Level Chunks

Standard chunks contain multiple facts, some relevant and some not. When you retrieve a 512-token chunk, maybe only 2 sentences are actually relevant. The rest is noise that can confuse the LLM.

### Proposition-Level Retrieval

**Propositions** are atomic, self-contained factual statements extracted from text:

**Original text**:
"ChromaDB is an open-source embedding database. It supports both in-memory and persistent storage. The Python client provides a simple API for adding, querying, and deleting embeddings. ChromaDB was founded in 2022 and has gained significant traction in the RAG community."

**Propositions**:
1. "ChromaDB is an open-source embedding database."
2. "ChromaDB supports in-memory storage."
3. "ChromaDB supports persistent storage."
4. "ChromaDB has a Python client."
5. "The ChromaDB Python client provides an API for adding embeddings."
6. "The ChromaDB Python client provides an API for querying embeddings."
7. "The ChromaDB Python client provides an API for deleting embeddings."
8. "ChromaDB was founded in 2022."
9. "ChromaDB has gained significant traction in the RAG community."

### Why Propositions Help

- Each proposition maps to exactly one fact
- Retrieval is highly precise — you get exactly the fact you need
- No noise from surrounding irrelevant sentences
- Embedding of a short, focused statement is more semantically accurate

### Results

- Proposition-based retrieval outperforms passage-level on factoid QA
- Especially strong for specific factual questions ("When was X founded?")
- Less effective for questions requiring extended reasoning or context

### Practical Implementation

**Extracting propositions** requires an LLM:
```
Given the following text, extract all atomic factual propositions.
Each proposition should be self-contained (understandable without context)
and express exactly one fact.

Text: {chunk_text}

Propositions:
```

**Trade-offs**:
- **Indexing cost**: LLM call per chunk to extract propositions (one-time cost)
- **Storage**: ~3-5× more entries than passage-level chunking
- **Retrieval precision**: Significantly higher for factual queries
- **Context**: Propositions lose narrative flow — may need to retrieve adjacent propositions

### Hybrid Approach: Propositions + Parent Chunks

Best of both worlds:
1. Extract propositions from each chunk
2. Index propositions for retrieval
3. When a proposition is retrieved, return its parent chunk for context
4. The LLM gets the full context, but retrieval was precise

This is the "small-to-big" retrieval pattern.

## Hierarchical Chunking in Practice

### Parent-Child Architecture

```
Document
├── Section 1 (parent chunk: ~2000 tokens)
│   ├── Paragraph 1.1 (child chunk: ~200 tokens)
│   ├── Paragraph 1.2 (child chunk: ~200 tokens)
│   └── Paragraph 1.3 (child chunk: ~200 tokens)
├── Section 2 (parent chunk: ~1500 tokens)
│   ├── Paragraph 2.1 (child chunk: ~200 tokens)
│   └── Paragraph 2.2 (child chunk: ~200 tokens)
```

**Retrieval flow**:
1. Search at the child level (small, precise chunks)
2. If multiple children from the same parent are retrieved, return the parent instead
3. This auto-merges related information

### Auto-Merging Retrieval

**LlamaIndex implementation pattern**:
```
# Define threshold: if > 50% of a parent's children are retrieved,
# return the parent chunk instead of individual children

parent_hit_counts = {}
for child in retrieved_children:
    parent_id = child.metadata["parent_id"]
    parent_hit_counts[parent_id] = parent_hit_counts.get(parent_id, 0) + 1

for parent_id, hits in parent_hit_counts.items():
    total_children = get_child_count(parent_id)
    if hits / total_children > 0.5:
        # Replace children with parent
        replace_children_with_parent(parent_id)
```

### When to Use Hierarchical Chunking

**Good fit**:
- Documents with clear structure (technical docs, legal texts, textbooks)
- Questions that vary in specificity (some need a fact, others need a section)
- When you want adaptive context size without pre-determining chunk size

**Poor fit**:
- Flat documents without structure (chat logs, social media)
- Very short documents (already chunk-sized)
- Real-time indexing requirements (hierarchical indexing is more complex)

## Semantic Chunking

### Embedding-Based Boundary Detection

Instead of fixed-size windows, detect where the topic changes:

1. Split text into sentences
2. Embed each sentence
3. Compute cosine similarity between adjacent sentence embeddings
4. Where similarity drops below a threshold → chunk boundary

**Threshold selection**:
- Too low: Very large chunks (few boundaries detected)
- Too high: Very small chunks (too many boundaries)
- **Adaptive approach**: Use the mean similarity minus one standard deviation as the threshold

### Advantages Over Fixed-Size Chunking

- Chunks align with topic boundaries naturally
- No arbitrary splitting of related content
- Chunk sizes adapt to content structure
- Better retrieval for topic-focused queries

### Disadvantages

- Requires embedding each sentence at index time (slow for large corpora)
- Chunks have variable size — harder to predict storage and context window usage
- Sensitive to threshold parameter
- Doesn't work well for content without clear topic shifts (e.g., a single continuous argument)

### When to Choose Semantic Chunking

- Documents cover multiple topics within a single file
- Topic boundaries don't align with structural boundaries (no headers)
- Retrieval quality is more important than indexing speed
- You have compute budget for sentence-level embedding at index time
