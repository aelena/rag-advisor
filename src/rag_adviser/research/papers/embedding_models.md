# Embedding Models for RAG Systems

## How Embeddings Work in RAG

Embedding models convert text into dense vector representations (typically 384-1536 dimensions). Similar texts produce vectors that are close together in this high-dimensional space. RAG uses this property to find documents relevant to a query.

The embedding model is the most critical choice in a RAG system. It determines retrieval quality, and unlike other components, changing it requires re-embedding the entire corpus.

## Model Selection Criteria

### 1. Dimension and Quality Trade-off
- **384 dimensions** (e.g., all-MiniLM-L6-v2): Fast, small, good for prototyping
- **768 dimensions** (e.g., bge-base-en-v1.5): Good balance of quality and size
- **1024 dimensions** (e.g., bge-large-en-v1.5): High quality, more storage/compute
- **1536 dimensions** (e.g., text-embedding-3-large): Best quality, most expensive

Higher dimensions capture more semantic nuance but increase storage, memory, and search latency.

### 2. Multilingual Support
- **English-only models** (e.g., all-MiniLM-L6-v2, bge-en series): Higher quality for English
- **Multilingual models** (e.g., multilingual-e5-large, bge-m3): Support 100+ languages, slightly lower per-language quality
- **Critical rule**: If you might add non-English content later, use a multilingual model from day one. Switching models requires re-embedding everything.

### 3. Max Token Length
- **256 tokens**: Fast but truncates longer passages (all-MiniLM-L6-v2)
- **512 tokens**: Standard for most models (bge-base, e5-base)
- **8192 tokens**: Long-context models (bge-m3, nomic-embed-text-v1.5, jina-embeddings-v2)
- **Rule**: Your chunk size should not exceed the model's max token length. If it does, the text is silently truncated and you lose information.

### 4. Model Size and Hardware
- **< 100MB** (MiniLM-L6): Runs on CPU, edge devices, ~50ms/query
- **100MB-500MB** (bge-base, mpnet-base): Needs ~500MB RAM, ~100ms/query on CPU
- **500MB-2GB** (bge-large, e5-large): Benefits from GPU, ~200ms/query on CPU
- **> 2GB** (bge-m3, multilingual-e5-large): Needs GPU for reasonable latency

### 5. Fine-tuning Capability
For domain-specific use cases (medical, legal, financial), fine-tuning the embedding model on domain data can improve retrieval by 10-30%. Models from sentence-transformers and BAAI support fine-tuning with contrastive learning.

## Recommended Models by Scenario

### CPU-only, English, fast prototyping
- **sentence-transformers/all-MiniLM-L6-v2**: 384d, 80MB, 256 tokens, Apache-2.0

### Production English, good quality
- **BAAI/bge-base-en-v1.5**: 768d, 440MB, 512 tokens, MIT
- **sentence-transformers/all-mpnet-base-v2**: 768d, 440MB, 384 tokens, Apache-2.0

### High-quality English
- **BAAI/bge-large-en-v1.5**: 1024d, 1.34GB, 512 tokens, MIT

### Multilingual
- **intfloat/multilingual-e5-base**: 768d, 1.11GB, 512 tokens, MIT (good balance)
- **intfloat/multilingual-e5-large**: 1024d, 2.24GB, 512 tokens, MIT (best quality)
- **BAAI/bge-m3**: 1024d, 2.27GB, 8192 tokens, MIT (long context + multilingual)

### Long-context documents
- **nomic-ai/nomic-embed-text-v1.5**: 768d, 550MB, 8192 tokens, Apache-2.0
- **BAAI/bge-m3**: 1024d, 2.27GB, 8192 tokens, MIT

### API-based (no local compute)
- **OpenAI text-embedding-3-small**: 1536d, $0.02/1M tokens, API-only
- **OpenAI text-embedding-3-large**: 3072d, $0.13/1M tokens, API-only
- **Cohere embed-v3**: 1024d, $0.10/1M tokens, API-only

## MTEB Benchmark

The Massive Text Embedding Benchmark (MTEB) is the standard benchmark for comparing embedding models. Check https://huggingface.co/spaces/mteb/leaderboard for current rankings. Key categories:
- **Retrieval**: Most relevant for RAG (measures how well the model retrieves relevant documents)
- **STS (Semantic Textual Similarity)**: How well the model captures semantic similarity
- **Classification**: Less relevant for RAG

Caution: MTEB scores are on general benchmarks. A model that ranks #1 on MTEB may not be best for your specific domain. Always evaluate on your own data.

## Re-indexing Warning

Changing the embedding model requires re-embedding ALL documents. Vector spaces from different models are incompatible. Plan this choice carefully:
- Start with a model that handles your likely future needs (multilingual if you might add languages)
- Test with a small subset before committing to a full re-index
- Keep the original documents — you'll need them if you switch models
