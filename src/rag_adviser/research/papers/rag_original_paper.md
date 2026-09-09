# RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Reference**: Lewis et al., 2020 — arXiv:2005.11401

## Core Idea

RAG combines a pre-trained parametric memory (the LLM) with a non-parametric memory (a dense retrieval index over documents). At inference time, the model retrieves relevant documents and conditions its generation on them.

## Architecture

Two variants proposed:

### RAG-Sequence
- Retrieve top-k documents for the query
- Generate the **entire** output sequence conditioned on each document independently
- Marginalize over documents: P(y|x) = Σ P(z|x) · P(y|x,z)
- Each document produces a complete answer candidate
- **Best for**: Tasks where the answer comes from a single coherent source

### RAG-Token
- Retrieve top-k documents for the query
- At **each generation step**, marginalize over documents to pick the next token
- Different tokens in the output can be influenced by different retrieved documents
- **Best for**: Tasks requiring synthesis across multiple sources (e.g., "list all X")

## Key Findings

1. **RAG outperforms closed-book models** on knowledge-intensive tasks (open-domain QA, fact verification, slot filling) even when the closed-book model is much larger.

2. **RAG generates more specific and factual text** compared to pure parametric models. Human evaluators rated RAG outputs as more factual.

3. **The retrieval component is critical** — replacing the retriever with random documents severely degrades performance. The quality of retrieval directly bounds answer quality.

4. **Documents can be "hot-swapped"** without retraining. Updating the document index immediately changes the model's knowledge — unlike parametric models that need fine-tuning.

5. **RAG-Token is better for generation tasks** (abstractive QA), while RAG-Sequence is better for extractive tasks.

## Retrieval Component: DPR

The original RAG paper uses Dense Passage Retrieval (DPR):
- Bi-encoder architecture: separate encoders for query and document
- BERT-base encoders fine-tuned on Natural Questions
- MIPS (Maximum Inner Product Search) over document embeddings
- Top-k documents retrieved (k=5 or k=10 in experiments)

## Practical Implications for RAG System Design

### What this means for chunk size
- The original paper uses Wikipedia passages of ~100 words (~128 tokens)
- This is significantly smaller than many modern RAG implementations use
- Smaller chunks = more precise retrieval but risk losing context
- The paper's success with small chunks suggests retrieval precision matters more than chunk completeness

### What this means for top-k
- Paper experiments with k=5 and k=10
- Marginal improvement beyond k=5 for most tasks
- More documents = more noise but also more recall
- **Recommendation**: Start with k=5, increase only if recall is the bottleneck

### What this means for the generation model
- The generation model must be able to **use** the retrieved context effectively
- Simply concatenating documents isn't enough — the model needs to attend to relevant parts
- Instruction-tuned models are much better at this than base models

### Hot-swapping implications
- One of RAG's key advantages: update knowledge without retraining
- This means your pipeline should support efficient re-indexing
- Incremental index updates are important for production systems
- Vector databases with upsert support (Qdrant, Weaviate, Pinecone) are preferred for frequently updated corpora

## Limitations Acknowledged in the Paper

1. **Retrieval ceiling**: If the relevant document isn't retrieved, the answer can't be correct. Retrieval recall is the hard upper bound on RAG performance.

2. **Retrieval latency**: Each generation requires a retrieval step, adding latency. RAG-Token requires retrieval at every token (impractical without caching).

3. **No multi-hop reasoning**: Single-step retrieval can't handle questions requiring chains of reasoning across multiple documents. This limitation led to later work on iterative/recursive RAG.

4. **Fixed retriever**: In the original formulation, the retriever isn't updated during training. Later work (RETRO, Atlas) explored joint training of retriever and generator.
