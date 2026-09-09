# RAG Foundations — Key Concepts and Architecture

## What is Retrieval-Augmented Generation (RAG)?

RAG is a technique that enhances LLM outputs by first retrieving relevant documents from an external knowledge base, then providing those documents as context for the LLM to generate answers. It was introduced by Lewis et al. (2020) in "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

The core motivation: LLMs have a knowledge cutoff and can hallucinate. By grounding generation in retrieved documents, RAG reduces hallucination and provides up-to-date, verifiable answers.

## RAG Pipeline Architecture

A standard RAG pipeline has these stages:

1. **Ingestion**: Documents are loaded, chunked, embedded, and stored in a vector database.
2. **Retrieval**: When a query arrives, it is embedded and used to find similar chunks via vector similarity search.
3. **Augmentation**: Retrieved chunks are inserted into the LLM prompt as context.
4. **Generation**: The LLM generates an answer grounded in the retrieved context.

### Ingestion Pipeline Detail

The ingestion pipeline is critical and often underestimated:

- **Document loading**: Parse PDFs, HTML, Markdown, Word docs into plain text. Tools: unstructured.io, docling, PyPDF, python-docx.
- **Chunking**: Split documents into smaller pieces. Strategy depends on content type. Typical sizes: 256-1024 tokens.
- **Embedding**: Convert chunks to dense vector representations using an embedding model (e.g., sentence-transformers, OpenAI text-embedding-3-small).
- **Indexing**: Store embeddings in a vector database for fast similarity search.

### Retrieval Pipeline Detail

The retrieval pipeline determines answer quality:

- **Query embedding**: The user query is embedded using the same model as the documents.
- **Similarity search**: Find the top-k most similar chunks using cosine similarity, dot product, or L2 distance.
- **Reranking** (optional): Use a cross-encoder model to rerank the initial results for higher precision.
- **Context assembly**: Combine retrieved chunks into a prompt for the LLM.

## When RAG Works Best

RAG excels when:
- The knowledge base changes frequently (can't retrain/fine-tune the LLM)
- You need verifiable, source-cited answers
- The domain is specialized (medical, legal, technical docs)
- You need to control what information the model has access to
- Privacy requires keeping data separate from the model

## When RAG is Not the Best Approach

RAG may be the wrong choice when:
- The corpus is small enough to fit in the LLM's context window (use direct context stuffing)
- Data is structured/tabular (use Text-to-SQL instead)
- Queries are simple keyword lookups (use full-text search like BM25)
- You need real-time data with no persistence (use direct API calls)
- The task is pure summarization of a known document (use long-context LLM)

## Key Metrics for RAG Evaluation

- **Hit Rate@k**: Fraction of queries where at least one relevant document appears in top-k results
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of the first relevant result
- **NDCG@k (Normalized Discounted Cumulative Gain)**: Measures ranking quality, penalizing relevant results that appear lower
- **Context Precision**: Fraction of retrieved chunks that are actually relevant
- **Context Recall**: Fraction of relevant information that was retrieved
- **Faithfulness**: Whether the generated answer is supported by the retrieved context (requires LLM evaluation)
- **Answer Relevancy**: Whether the answer addresses the query (requires LLM evaluation)
