# Embedding Models Research: E5, BGE, and Contrastive Learning

## Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)

**Reference**: Wang et al., 2022 — arXiv:2212.03533

### Core Approach

E5 (EmbEddings from bidirEctional Encoder rEpresentations) trains text embeddings using:
1. Large-scale weakly-supervised contrastive learning on (query, passage) pairs mined from the web
2. Over 1 billion text pairs from CCPairs dataset (Common Crawl, Reddit, academic papers, etc.)
3. Consistency filtering to remove noisy pairs

### Key Design Decisions

**Prompt-based encoding**: E5 prepends task-specific prefixes to inputs:
- `"query: "` for search queries
- `"passage: "` for documents to be indexed

This signals to the model whether the input is a query or a document, improving the alignment between query and document embeddings.

**Why this matters for RAG**: Always use the correct prefix when embedding. Embedding a document with the "query:" prefix (or vice versa) significantly degrades retrieval quality. Check your embedding code — this is a common mistake.

### Model Variants

| Model | Dimensions | Parameters | MTEB Average |
|-------|-----------|------------|-------------|
| e5-small-v2 | 384 | 33M | 59.9 |
| e5-base-v2 | 768 | 109M | 61.5 |
| e5-large-v2 | 1024 | 335M | 62.3 |
| multilingual-e5-small | 384 | 118M | 57.5 |
| multilingual-e5-base | 768 | 278M | 59.3 |
| multilingual-e5-large | 1024 | 560M | 61.5 |

### Multilingual E5

**Reference**: Wang et al., 2024 — arXiv:2402.05672

- Supports 100+ languages
- Competitive with language-specific models
- Single model for multilingual corpora — no need to detect language and route to different models
- **Use case**: Mixed-language document collections, multilingual customer support, international legal documents

## C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE)

**Reference**: Xiao et al., 2023 — arXiv:2309.07597

### BAAI General Embedding (BGE) Models

BGE models from the Beijing Academy of AI represent a family of high-quality embedding models with several innovations:

### Training Pipeline

1. **Pre-training**: RetroMAE (Retrieval-oriented Masked Auto-Encoder) — masks and reconstructs text pairs
2. **General fine-tuning**: Contrastive learning on large-scale (query, passage) pairs
3. **Task-specific fine-tuning**: Further tuning on specific retrieval tasks

### BGE Model Variants

| Model | Dimensions | Max Tokens | Language | MTEB Rank |
|-------|-----------|-----------|----------|-----------|
| bge-small-en-v1.5 | 384 | 512 | English | Good |
| bge-base-en-v1.5 | 768 | 512 | English | Very Good |
| bge-large-en-v1.5 | 1024 | 512 | English | Excellent |
| bge-m3 | 1024 | 8192 | 100+ langs | State-of-art |
| bge-reranker-v2-m3 | — | 8192 | 100+ langs | Top reranker |

### BGE-M3: Multi-Functionality, Multi-Linguality, Multi-Granularity

BGE-M3 is notable for supporting three retrieval modes simultaneously:

1. **Dense retrieval**: Standard embedding similarity (like other models)
2. **Sparse retrieval**: Learned sparse representations (like SPLADE) — generates term weights for BM25-style matching
3. **ColBERT-style retrieval**: Multi-vector late interaction — token-level matching

**Why this matters**: A single model can power hybrid search without needing separate dense and sparse models.

**Practical usage**:
```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Get all three representations at once
output = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
dense_embeddings = output['dense_vecs']
sparse_embeddings = output['lexical_weights']
colbert_embeddings = output['colbert_vecs']
```

### Long Document Support

BGE-M3 supports up to 8192 tokens — significantly more than most embedding models (512 tokens). This means:
- Larger chunks are possible without truncation
- Entire short documents can be embedded as single units
- Less information loss from chunking

**Trade-off**: Longer inputs = slower embedding. For batch indexing this is fine; for real-time query embedding the extra tokens add latency.

## Practical Embedding Model Selection

### Decision Matrix

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| English, low resource | all-MiniLM-L6-v2 | Fast, 80MB, good enough |
| English, quality focus | bge-large-en-v1.5 | Top MTEB scores |
| Multilingual | multilingual-e5-large | Best multilingual balance |
| Multilingual + hybrid search | bge-m3 | Dense + sparse in one model |
| Long documents (>512 tokens) | bge-m3, jina-embeddings-v2 | 8K token support |
| Code | voyage-code-2 | Trained on code specifically |
| Cost-sensitive API | OpenAI text-embedding-3-small | $0.02/1M tokens |
| Quality-first API | OpenAI text-embedding-3-large | Best API embedding |
| Air-gapped / privacy | any sentence-transformers model | Runs fully offline |

### Embedding Dimensions and Storage

| Dimensions | Vector Size (float32) | Per 1M Vectors | Impact on Search |
|-----------|----------------------|-----------------|------------------|
| 384 | 1.5 KB | 1.5 GB | Fastest |
| 768 | 3 KB | 3 GB | Balanced |
| 1024 | 4 KB | 4 GB | Slightly slower |
| 1536 | 6 KB | 6 GB | Slower but richer |
| 3072 | 12 KB | 12 GB | Diminishing returns |

**Practical note**: For < 1M documents, dimension has negligible impact on search speed. Dimension matters for cost (storage, memory) and indexing throughput at scale.

### Matryoshka Representation Learning (MRL)

**Reference**: Kusupati et al., 2022 — arXiv:2205.13147

Some newer models (OpenAI text-embedding-3, nomic-embed-text) support Matryoshka embeddings:
- Train the model so that the first N dimensions are a valid embedding in N-dimensional space
- You can truncate a 1536-dim embedding to 256-dim and still get decent retrieval
- Trade-off: quality vs. storage/speed

**Use case**: Start with full dimensions, reduce later if storage or speed becomes a bottleneck. No need to re-embed.

### Common Embedding Mistakes

1. **Mismatched prefixes**: E5 and BGE models expect specific prefixes ("query:", "passage:"). Omitting them drops recall by 5-15%.

2. **Not normalizing**: Some models output unnormalized embeddings. Cosine similarity requires normalized vectors. Most sentence-transformers models normalize by default, but verify.

3. **Exceeding max tokens silently**: Most models truncate at 512 tokens without warning. If your chunks are 800 tokens, half the content is ignored. Check your model's max token limit.

4. **Using the wrong similarity metric**: Some models are trained for cosine similarity, others for dot product. Using the wrong metric in your vector DB degrades results. Check model documentation.

5. **Embedding model mismatch**: Your query embedding model must be identical to your document embedding model. Different models produce incompatible embedding spaces.

6. **Not batching**: Embedding one document at a time is 10-50× slower than batching. Always batch embed (batch_size=32-128 depending on GPU memory).
