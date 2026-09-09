# RAG Evaluation: RAGAS, RGB Benchmark, and Best Practices

## RAGAS: Automated Evaluation of Retrieval Augmented Generation

**Reference**: Es et al., 2024 — arXiv:2309.15217

### What is RAGAS?

RAGAS is an open-source framework for evaluating RAG pipelines **without** human-labeled ground truth. It uses LLM-as-judge to assess both retrieval and generation quality.

### Core Metrics

#### 1. Faithfulness
**Question**: Is the generated answer supported by the retrieved context?

**How it works**:
1. Decompose the answer into individual claims/statements
2. For each claim, check if it can be inferred from the context
3. Score = (claims supported by context) / (total claims)

**Score range**: 0.0 - 1.0 (higher is better)
**Target**: ≥ 0.85 for production systems

**What low faithfulness means**: The LLM is hallucinating — generating information not present in the retrieved context. Fix by: lowering temperature, improving prompt instructions, using "refine" strategy instead of "stuff".

#### 2. Answer Relevancy
**Question**: Does the answer actually address the question?

**How it works**:
1. Given the answer, use an LLM to generate N questions that the answer could address
2. Compute cosine similarity between the generated questions and the original question
3. Score = average similarity

**Score range**: 0.0 - 1.0 (higher is better)
**Target**: ≥ 0.80

**What low relevancy means**: The answer is off-topic or too vague. Usually a generation problem — the LLM is not focusing on the question. Fix by: improving the prompt template, reducing irrelevant context, adding explicit instruction to answer the specific question.

#### 3. Context Precision
**Question**: Are the retrieved chunks actually relevant to the question?

**How it works**:
1. For each retrieved chunk, use an LLM to judge if it's relevant to answering the question
2. Compute precision at each rank position (weighted by position)
3. Higher-ranked relevant chunks contribute more to the score

**Score range**: 0.0 - 1.0 (higher is better)
**Target**: ≥ 0.75

**What low precision means**: Retrieving too much irrelevant context. Fix by: improving embeddings, adding reranking, reducing top_k, improving chunking granularity, using hybrid search.

#### 4. Context Recall
**Question**: Was all necessary information retrieved?

**How it works** (requires ground truth answer):
1. Decompose the ground truth answer into claims
2. For each claim, check if any retrieved chunk contains that information
3. Score = (claims found in context) / (total claims in ground truth)

**Score range**: 0.0 - 1.0 (higher is better)
**Target**: ≥ 0.80

**What low recall means**: Missing relevant documents. Fix by: increasing top_k, trying different embedding models, adding query expansion, using hybrid search, improving chunking (overlap, size).

### RAGAS Limitations

1. **LLM-as-judge is not perfect** — LLM evaluators can be biased, inconsistent, or wrong. Always validate with human evaluation on a sample.
2. **Cost**: Each evaluation requires multiple LLM calls per query. Evaluating 100 queries can cost $5-20 depending on the model.
3. **Context Recall requires ground truth** — unlike other metrics, you need reference answers.
4. **Not deterministic** — scores can vary between runs due to LLM non-determinism. Run evaluations 3× and average.

## Benchmarking Large Language Models in Retrieval-Augmented Generation

**Reference**: Chen et al., 2024 — arXiv:2309.01431 (RGB Benchmark)

### What RGB Tests

The RGB (Retrieval-Augmented Generation Benchmark) evaluates four critical abilities:

#### 1. Noise Robustness
Can the model ignore irrelevant retrieved documents?

**Test setup**: Mix relevant documents with irrelevant ones. Vary the noise ratio from 0% to 80%.

**Findings**:
- All models degrade as noise increases, but at different rates
- GPT-4 class models maintain accuracy up to ~40% noise
- Smaller models fail above ~20% noise
- **Implication**: High retrieval precision matters more for smaller models. If using a weaker LLM, invest more in retrieval quality and reranking.

#### 2. Negative Rejection
Can the model say "I don't know" when the context doesn't contain the answer?

**Test setup**: Provide only irrelevant documents for a question.

**Findings**:
- Most models struggle with this — they attempt to answer even without relevant context
- Explicit instruction ("If the context doesn't contain the answer, say 'I don't know'") helps but doesn't fully solve it
- **Implication**: Always set a retrieval similarity threshold. If the best match is below threshold, don't even send to the LLM.

#### 3. Information Integration
Can the model synthesize information spread across multiple documents?

**Test setup**: Answers require combining facts from 2-4 different documents.

**Findings**:
- Performance drops significantly when information is split across documents
- Models are better at integration when documents are presented in logical order
- **Implication**: For multi-hop questions, consider iterative retrieval (retrieve → generate partial answer → retrieve more → refine) rather than single-shot retrieval.

#### 4. Counterfactual Robustness
Can the model detect and handle contradictory information in the context?

**Test setup**: Include documents with intentionally wrong facts alongside correct ones.

**Findings**:
- Most models fail to detect contradictions and may use the wrong information
- Majority voting (if 3 documents agree and 1 disagrees, go with the majority) is not reliably performed
- **Implication**: For corpora with potential contradictions (version histories, competing sources), add document timestamps and instruct the model to prefer recent information.

## Practical Evaluation Playbook

### Phase 1: Quick Sanity Check (Day 1)
- Create 10-20 hand-written Q&A pairs covering your main use cases
- Compute Hit Rate@5 and MRR for retrieval
- Manually inspect generated answers
- **Goal**: Verify the pipeline works at all

### Phase 2: Systematic Evaluation (Week 1)
- Expand to 50-100 Q&A pairs from domain experts
- Include edge cases: negation, multi-hop, out-of-scope, adversarial
- Run RAGAS metrics (faithfulness, relevancy, precision, recall)
- **Goal**: Identify the weakest component (retrieval vs. generation)

### Phase 3: Continuous Evaluation (Ongoing)
- Log all production queries and retrieved contexts
- Sample 10-20 queries per week for human evaluation
- Track RAGAS metrics over time
- Set up alerts for metric drops
- **Goal**: Catch degradation early (data drift, index staleness)

### Evaluation Anti-Patterns

1. **Only evaluating end-to-end** — If the answer is wrong, you don't know if it's retrieval or generation. Always evaluate retrieval separately.

2. **Using MTEB scores as proxy for your domain** — MTEB evaluates general-purpose retrieval. Your legal/medical/code domain may behave very differently. Always evaluate on your own data.

3. **Testing only happy paths** — Real users ask ambiguous, poorly-worded, and out-of-scope questions. Include these in your test set.

4. **One-time evaluation** — Your data changes, your users change, your models get updated. Evaluation must be continuous.

5. **Ignoring latency in evaluation** — A perfect answer in 10 seconds may be worse than a good answer in 500ms for your use case.
