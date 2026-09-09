# Vector Databases for RAG Systems

## Role of the Vector Database

The vector database stores document embeddings and enables fast similarity search. It's the persistence and retrieval layer of a RAG system. Choosing the right one depends on scale, deployment, privacy requirements, and feature needs.

## Database Categories

### Embedded (In-Process)
Run inside your application process. No separate server needed.

- **ChromaDB**: Python-native, simple API, great for prototyping. Up to ~1M vectors. Supports metadata filtering. No hybrid search. `pip install chromadb`.
- **FAISS** (Facebook AI Similarity Search): Fastest raw search speed. In-memory. No metadata filtering built-in. Best for batch processing or when you manage metadata externally. Up to ~10M vectors in memory. `pip install faiss-cpu`.
- **LanceDB**: Serverless, disk-based. Supports hybrid search. Good for local/edge deployment. `pip install lancedb`.
- **SQLite + sqlite-vec**: Single-file database. Good if you already use SQLite. New but promising. `pip install sqlite-vec`.

### Client-Server
Separate database process. Better for production, multi-user, larger scale.

- **Qdrant**: Rust-based, fast. Supports metadata filtering, quantization, hybrid search. Production-ready. Docker or cloud. Up to 100M+ vectors. `pip install qdrant-client`.
- **Weaviate**: Full-featured. Built-in vectorization, GraphQL API. Supports hybrid search. Up to 100M+ vectors. `pip install weaviate-client`.
- **Milvus**: Distributed architecture. Handles billions of vectors. Self-hosted. Good for large-scale production. `pip install pymilvus`.
- **pgvector**: PostgreSQL extension. Use your existing Postgres. Familiar SQL interface. Up to ~10M vectors efficiently. `pip install pgvector psycopg`.

### Managed (Cloud)
Fully managed service. No infrastructure to maintain.

- **Pinecone**: Auto-scaling, managed. Supports metadata filtering and hybrid search. Up to 1B+ vectors. API key required. `pip install pinecone-client`.
- **Weaviate Cloud**: Managed Weaviate.
- **Qdrant Cloud**: Managed Qdrant.
- **Zilliz Cloud**: Managed Milvus.

## Selection Guide

### By Scale
| Vectors | Recommended |
|---------|------------|
| < 10K | ChromaDB, FAISS, sqlite-vec — keep it simple |
| 10K-1M | ChromaDB, Qdrant, LanceDB |
| 1M-10M | Qdrant, pgvector, FAISS (if in-memory is OK) |
| 10M-100M | Qdrant, Weaviate, Milvus |
| 100M+ | Milvus, Pinecone, Qdrant (distributed) |

### By Deployment Constraint
| Constraint | Recommended |
|-----------|------------|
| No server, embedded | ChromaDB, FAISS, LanceDB, sqlite-vec |
| Air-gapped/offline | ChromaDB, FAISS, LanceDB, sqlite-vec |
| Already using PostgreSQL | pgvector |
| Fully managed, no ops | Pinecone, Weaviate Cloud |
| Maximum search speed | FAISS (in-memory) |
| Hybrid search (BM25 + vector) | Qdrant, Weaviate, LanceDB |

### Privacy Considerations
- **Air-gapped**: Must be self-hosted. Use ChromaDB, FAISS, LanceDB, or sqlite-vec.
- **Strict privacy**: No cloud services. Use self-hosted options.
- **Moderate privacy**: Cloud is OK if data doesn't leave region. Check provider's data handling policies.
- **No restrictions**: Any option works, including managed services.

## Hybrid Search

Hybrid search combines semantic (vector) search with keyword (BM25/full-text) search. This is recommended when:
- Queries mix specific terms with natural language ("section 4.2 penalty clause")
- Users expect exact keyword matches to always appear
- Domain has specific terminology that embedding models may not capture well

Databases that support hybrid search natively: Qdrant, Weaviate, LanceDB, Pinecone.
For databases without hybrid search, you can implement it by running BM25 (e.g., via Elasticsearch or SQLite FTS5) in parallel and merging results with Reciprocal Rank Fusion (RRF).

## Performance Tuning

### Index Types
- **Flat (brute force)**: Exact search. Best quality. O(n) time. Use for < 10K vectors.
- **IVF (Inverted File Index)**: Approximate. Clusters vectors. Good for 10K-10M. Tunable via `nlist` and `nprobe`.
- **HNSW (Hierarchical Navigable Small World)**: Approximate. Graph-based. Best quality/speed trade-off for most use cases. Default in most modern databases.

### Quantization
Reduce storage and speed up search by compressing vectors:
- **Product Quantization (PQ)**: 4-8x compression, some quality loss
- **Scalar Quantization (SQ)**: 4x compression, minimal quality loss
- **Binary Quantization**: 32x compression, significant quality loss

Use quantization when memory is a constraint or you have 1M+ vectors.
