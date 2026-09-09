# Self-Reflective and Adaptive RAG

## Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

**Reference**: Asai et al., 2024 — arXiv:2310.11511

### Core Idea

Standard RAG always retrieves, even when unnecessary. Self-RAG trains the LLM to:
1. Decide **whether** to retrieve (some questions don't need it)
2. Evaluate **whether** retrieved passages are relevant
3. Check **whether** the generated response is supported by the passages
4. Assess overall **utility** of the response

### Reflection Tokens

Self-RAG introduces four special tokens the model learns to generate:

- **[Retrieve]**: yes/no — should I retrieve for this segment?
- **[IsRel]**: relevant/irrelevant — is this passage relevant to the query?
- **[IsSup]**: fully supported/partially supported/no support — is my response grounded?
- **[IsUse]**: 1-5 rating — how useful is my overall response?

### How It Works

1. Given input, model generates [Retrieve] token
2. If [Retrieve]=yes → retrieve top-k passages
3. For each passage, generate [IsRel] → filter irrelevant ones
4. Generate response conditioned on relevant passages
5. Generate [IsSup] → check if response is grounded
6. Generate [IsUse] → score overall quality
7. If quality is low, can re-retrieve or revise

### Performance Results

- Outperforms ChatGPT and Llama2-chat on open-domain QA
- Significantly reduces hallucination (measured by factual precision)
- Retrieves only when needed — ~40-60% of queries trigger retrieval
- The selective retrieval actually improves both accuracy and efficiency

### Practical Implications

**For RAG system designers**:
- Not every query needs retrieval — implement a retrieval gate
- A simple retrieval gate: classify the query as needing external knowledge or not
- Cheaper alternative to full Self-RAG: use a lightweight classifier before retrieval
- Saves latency and cost for queries the LLM can answer from parametric knowledge

**Simple retrieval gate implementation**:
- Use the LLM itself: "Does this question require external documents to answer? yes/no"
- Or train a small classifier on your query distribution
- Rule-based: if query is about your specific domain → retrieve; if general knowledge → skip

## REPLUG: Retrieval-Augmented Black-Box Language Models

**Reference**: Shi et al., 2023 — arXiv:2301.12652

### Core Idea

REPLUG treats the LLM as a **black box** — no access to model internals. Retrieval is purely a pre-processing step that prepends documents to the input.

### Key Contributions

**Ensemble approach**: Instead of concatenating all documents, REPLUG:
1. Retrieves top-k documents
2. Prepends each document separately to the query
3. Runs the LLM k times (once per document)
4. Ensembles the output probabilities

This avoids the "lost in the middle" problem where LLMs ignore middle context.

**LSR (LM-Supervised Retrieval)**:
- Uses the LLM's perplexity as a training signal for the retriever
- Documents that lead to lower perplexity (better generation) get higher retrieval scores
- No need for human-labeled retrieval training data

### Practical Implications

- **You don't need to fine-tune the LLM** — retrieval augmentation works as pure pre-processing
- **Ensemble over documents** is more robust than concatenation, but costs k× inference
- **Trade-off**: k separate LLM calls vs. one call with all documents concatenated
- **Compromise**: Retrieve top-20, rerank to top-3, concatenate the top-3. This gets most of the benefit without k× cost.

## Corrective RAG (CRAG)

**Reference**: Yan et al., 2024 — arXiv:2401.15884

### Core Idea

Add a lightweight evaluator that assesses retrieval quality before generation. If retrieval quality is poor, trigger corrective actions.

### Three Actions Based on Retrieval Quality

1. **Correct** (high confidence): Retrieved documents are relevant → proceed with generation
2. **Incorrect** (low confidence): Retrieved documents are irrelevant → fall back to web search or refuse to answer
3. **Ambiguous** (medium confidence): Partially relevant → extract relevant sentences, discard the rest, optionally supplement with web search

### Knowledge Refinement

For each retrieved document, CRAG:
1. Decomposes it into fine-grained knowledge strips (sentences/clauses)
2. Scores each strip for relevance to the query
3. Filters out irrelevant strips
4. Concatenates only relevant strips for generation

This is essentially **context compression** at the sentence level.

### Practical Implications

- **Always score your retrieved context** — don't blindly pass everything to the LLM
- Implement a relevance threshold: if max similarity score < threshold, consider the query unanswerable
- **Context compression saves tokens and improves accuracy** — less noise means better generation
- **Fallback strategies matter**: web search, different index, or graceful "I don't know"

## Design Patterns for Adaptive RAG

Combining insights from Self-RAG, REPLUG, and CRAG, here are practical design patterns:

### Pattern 1: Retrieval Gate
```
if query_needs_retrieval(query):
    results = retrieve(query)
else:
    results = []  # LLM answers from parametric knowledge
```
**Benefit**: 30-50% latency reduction on mixed query workloads.

### Pattern 2: Retrieval Quality Check
```
results = retrieve(query, top_k=10)
max_score = max(score for _, score in results)
if max_score < THRESHOLD:
    return "I don't have enough information to answer this."
```
**Benefit**: Reduces hallucination from poor retrieval.

### Pattern 3: Context Compression
```
results = retrieve(query, top_k=10)
compressed = []
for chunk, score in results:
    relevant_sentences = extract_relevant_sentences(chunk, query)
    compressed.extend(relevant_sentences)
```
**Benefit**: Fits more relevant information in the context window.

### Pattern 4: Adaptive Top-K
```
results = retrieve(query, top_k=20)
# Keep adding results until cumulative relevance plateaus
selected = []
for chunk, score in sorted(results, key=lambda x: -x[1]):
    if score < RELEVANCE_FLOOR:
        break
    selected.append(chunk)
    if len(selected) >= MAX_K:
        break
```
**Benefit**: Uses fewer chunks for easy queries, more for hard ones.
