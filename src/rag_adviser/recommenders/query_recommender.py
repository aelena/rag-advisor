"""Query transformation recommender — suggests query pipeline techniques."""

from __future__ import annotations

from rag_adviser.models import (
    AnswerType,
    LatencyBudget,
    QueryComplexity,
    QueryTransformationRecommendation,
    QueryType,
    UseCase,
    UserAnswers,
)


class QueryRecommender:
    """Recommend query transformation techniques based on query patterns."""

    def recommend(self, answers: UserAnswers) -> QueryTransformationRecommendation:
        """Generate query transformation recommendations."""
        rec = QueryTransformationRecommendation()
        latency = answers.constraints.latency_budget

        # Multi-turn: always recommend conversation condensation
        if answers.query_type == QueryType.MULTI_TURN:
            rec.techniques.append({
                "name": "Conversation Condensation",
                "priority": "required",
                "description": (
                    "Compress multi-turn chat history into a standalone query "
                    "before retrieval. Without this, the retriever sees only the "
                    "latest message and loses context."
                ),
            })
            rec.requires_llm = True
            rec.latency_impact_ms += 500
            rec.notes.append(
                "Multi-turn queries MUST be condensed before retrieval — "
                "this is not optional"
            )

        # Short keywords: query expansion
        if answers.query_type == QueryType.SHORT_KEYWORDS:
            rec.techniques.append({
                "name": "Query Expansion",
                "priority": "recommended",
                "description": (
                    "Expand short keyword queries into fuller natural language "
                    "queries. Short queries have low semantic signal for embedding "
                    "models, leading to poor retrieval."
                ),
            })
            rec.requires_llm = True
            rec.latency_impact_ms += 400

        # Multi-hop: HyDE + multi-query
        if answers.query_complexity == QueryComplexity.MULTI_HOP:
            rec.techniques.append({
                "name": "HyDE (Hypothetical Document Embeddings)",
                "priority": "recommended",
                "description": (
                    "Generate a hypothetical answer to the query, then embed "
                    "that answer instead of the query. This bridges the semantic "
                    "gap between questions and document passages."
                ),
            })
            rec.techniques.append({
                "name": "Multi-Query Decomposition",
                "priority": "recommended",
                "description": (
                    "Decompose the complex query into sub-queries, retrieve for "
                    "each independently, then merge results. Essential for "
                    "multi-hop reasoning where the answer spans multiple documents."
                ),
            })
            rec.requires_llm = True
            rec.latency_impact_ms += 1200
            rec.notes.append(
                "Multi-hop queries benefit greatly from decomposition — "
                "expect 2-3x latency increase but significantly better recall"
            )

        # Comparative: multi-query retrieval
        if answers.query_complexity == QueryComplexity.COMPARATIVE:
            rec.techniques.append({
                "name": "Multi-Query Retrieval",
                "priority": "recommended",
                "description": (
                    "Generate multiple query variants (one per entity being "
                    "compared), retrieve for each, then merge results. This "
                    "ensures both sides of the comparison are represented."
                ),
            })
            rec.requires_llm = True
            rec.latency_impact_ms += 800

        # Aggregative: high top_k + broad retrieval
        if answers.query_complexity == QueryComplexity.AGGREGATIVE:
            rec.techniques.append({
                "name": "Broad Retrieval with Reranking",
                "priority": "recommended",
                "description": (
                    "Retrieve a large initial set (top_k=20+), then rerank to "
                    "select the most relevant. Aggregative queries need breadth "
                    "first, then precision."
                ),
            })
            rec.latency_impact_ms += 300
            rec.notes.append(
                "Consider using top_k=20 with a cross-encoder reranker "
                "for aggregative queries"
            )

        # HyDE for natural questions when passage retrieval is needed
        if (
            answers.query_type == QueryType.NATURAL_QUESTIONS
            and answers.expected_answer_type == AnswerType.EXACT_PASSAGE
            and answers.query_complexity == QueryComplexity.SIMPLE_FACTUAL
            and not any(t["name"].startswith("HyDE") for t in rec.techniques)
        ):
            rec.techniques.append({
                "name": "HyDE (Hypothetical Document Embeddings)",
                "priority": "optional",
                "description": (
                    "For exact passage retrieval, HyDE can improve recall by "
                    "generating a hypothetical passage that looks like the answer, "
                    "then using that for similarity search."
                ),
            })
            rec.requires_llm = True
            rec.latency_impact_ms += 600

        # Step-back prompting for complex queries with strict latency
        if (
            answers.query_complexity in (
                QueryComplexity.MULTI_HOP,
                QueryComplexity.COMPARATIVE,
            )
            and answers.use_case == UseCase.QA
            and not any(t["name"] == "Step-Back Prompting" for t in rec.techniques)
        ):
            rec.techniques.append({
                "name": "Step-Back Prompting",
                "priority": "optional",
                "description": (
                    "Ask a more general version of the query first, retrieve "
                    "context for that, then answer the specific question. "
                    "Helps when the original query is too specific to match."
                ),
            })
            rec.requires_llm = True
            rec.latency_impact_ms += 500

        # Latency warnings
        if latency == LatencyBudget.FAST and rec.latency_impact_ms > 300:
            rec.notes.append(
                f"WARNING: Recommended techniques add ~{rec.latency_impact_ms}ms, "
                f"which conflicts with your <500ms latency budget. Consider "
                f"caching transformed queries or using only the highest-priority "
                f"technique."
            )
        elif latency == LatencyBudget.MODERATE and rec.latency_impact_ms > 1500:
            rec.notes.append(
                f"Total estimated latency impact: ~{rec.latency_impact_ms}ms. "
                f"Consider running query transformations in parallel where possible."
            )

        # Generate code snippet if we have techniques
        if rec.techniques:
            rec.code_snippet = self._generate_code_snippet(rec, answers)

        return rec

    def _generate_code_snippet(
        self,
        rec: QueryTransformationRecommendation,
        answers: UserAnswers,
    ) -> str:
        """Generate a ready-to-use code snippet for the recommended techniques.

        Snippets use LangChain Expression Language (``langchain_core`` only) plus
        plain Python, so they do not depend on the legacy chain classes that
        were deprecated in LangChain 0.2 and moved out of the main package in
        LangChain 1.0. ``llm``, ``embeddings`` and ``retriever`` are assumed to
        be already-constructed objects.
        """
        technique_names = [t["name"] for t in rec.techniques]

        lines = [
            "# Query transformation pipeline",
            "# pip install langchain-core  (plus your LLM/embedding provider package)",
            "# Assumes: llm (chat model), embeddings, retriever = vectorstore.as_retriever()",
            "from langchain_core.output_parsers import StrOutputParser",
            "from langchain_core.prompts import ChatPromptTemplate",
            "",
        ]

        if "Conversation Condensation" in technique_names:
            lines.extend([
                "# --- Conversation condensation: rewrite chat history into a standalone query",
                "condense_prompt = ChatPromptTemplate.from_messages([",
                '    ("system", "Rewrite the latest user message as a self-contained '
                'question, using the chat history for context. Return only the question."),',
                '    ("human", "History:\\n{history}\\n\\nLatest message: {question}"),',
                "])",
                "condense = condense_prompt | llm | StrOutputParser()",
                "",
                "def condense_query(history: list[tuple[str, str]], question: str) -> str:",
                '    text = "\\n".join(f"{role}: {msg}" for role, msg in history)',
                '    return condense.invoke({"history": text, "question": question})',
                "",
            ])

        if "Query Expansion" in technique_names:
            lines.extend([
                "# --- Query expansion: turn short keywords into a full natural-language query",
                "expand_prompt = ChatPromptTemplate.from_messages([",
                '    ("system", "Expand the keyword query into one clear, specific question '
                'a document might answer. Return only the question."),',
                '    ("human", "{query}"),',
                "])",
                "expand = expand_prompt | llm | StrOutputParser()",
                "",
                "def expand_query(query: str) -> str:",
                '    return expand.invoke({"query": query})',
                "",
            ])

        if "HyDE (Hypothetical Document Embeddings)" in technique_names:
            lines.extend([
                "# --- HyDE: embed a hypothetical answer instead of the question",
                "hyde_prompt = ChatPromptTemplate.from_messages([",
                '    ("system", "Write a short, factual passage that would answer the '
                'question. Do not hedge; write as if quoting a reference document."),',
                '    ("human", "{question}"),',
                "])",
                "hyde = hyde_prompt | llm | StrOutputParser()",
                "",
                "def hyde_search(question: str, k: int = 5):",
                '    hypothetical = hyde.invoke({"question": question})',
                "    vector = embeddings.embed_query(hypothetical)",
                "    return vectorstore.similarity_search_by_vector(vector, k=k)",
                "",
            ])

        needs_multi_query = (
            "Multi-Query Decomposition" in technique_names
            or "Multi-Query Retrieval" in technique_names
        )
        if needs_multi_query:
            purpose = (
                "sub-questions that together answer the original"
                if "Multi-Query Decomposition" in technique_names
                else "query variants, one per entity or aspect being compared"
            )
            lines.extend([
                "# --- Multi-query: retrieve for several queries, merge via reciprocal rank fusion",
                "multi_prompt = ChatPromptTemplate.from_messages([",
                f'    ("system", "Generate 3 {purpose}. One per line, no numbering."),',
                '    ("human", "{question}"),',
                "])",
                "multi = multi_prompt | llm | StrOutputParser()",
                "",
                "def rrf_merge(result_lists, k: int = 60):",
                "    scores: dict[str, float] = {}",
                "    docs = {}",
                "    for results in result_lists:",
                "        for rank, doc in enumerate(results):",
                "            key = doc.page_content",
                "            docs[key] = doc",
                "            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)",
                "    return [docs[key] for key, _ in sorted(scores.items(), key=lambda x: -x[1])]",
                "",
                "def multi_query_search(question: str, k: int = 5):",
                '    raw = multi.invoke({"question": question})',
                "    variants = [q.strip() for q in raw.splitlines() if q.strip()]",
                "    result_lists = [retriever.invoke(q) for q in [question, *variants]]",
                "    return rrf_merge(result_lists)[:k]",
                "",
            ])

        if "Step-Back Prompting" in technique_names:
            lines.extend([
                "# --- Step-back prompting: retrieve for a more general question as well",
                "step_back_prompt = ChatPromptTemplate.from_messages([",
                '    ("system", "Rewrite the question as a more general, high-level '
                'question about the same topic. Return only the question."),',
                '    ("human", "{question}"),',
                "])",
                "step_back = step_back_prompt | llm | StrOutputParser()",
                "",
                "def step_back_search(question: str):",
                '    general = step_back.invoke({"question": question})',
                "    return retriever.invoke(question) + retriever.invoke(general)",
                "",
            ])

        if "Broad Retrieval with Reranking" in technique_names:
            lines.extend([
                "# --- Broad retrieval (top_k=20) then cross-encoder rerank to top 5",
                "# pip install sentence-transformers",
                "from sentence_transformers import CrossEncoder",
                "",
                'reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")',
                "",
                "def rerank_search(question: str, fetch_k: int = 20, k: int = 5):",
                '    candidates = vectorstore.similarity_search(question, k=fetch_k)',
                "    scores = reranker.predict([(question, d.page_content) for d in candidates])",
                "    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])",
                "    return [doc for doc, _ in ranked[:k]]",
                "",
            ])

        return "\n".join(lines)
