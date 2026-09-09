# Chunking Strategies for RAG Systems

## Why Chunking Matters

Chunking is the most impactful and least appreciated part of a RAG pipeline. Poor chunking leads to:
- Retrieving irrelevant context (chunk boundaries cut through important information)
- Missing relevant information (answer spans multiple chunks that aren't retrieved together)
- Wasted context window tokens on padding/overlap

The ideal chunk contains exactly the information needed to answer a query — no more, no less.

## Fixed-Size Chunking

The simplest approach: split text every N tokens/characters.

- **Pros**: Simple, predictable, fast
- **Cons**: Cuts through sentences, paragraphs, and semantic boundaries
- **When to use**: Baseline, or when content is homogeneous and unstructured
- **Typical sizes**: 256-512 tokens for Q&A, 512-1024 for summarization

## Recursive Character Splitting

Split on a hierarchy of separators: paragraphs first, then sentences, then words.

- **Implementation**: LangChain's RecursiveCharacterTextSplitter
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`
- **Pros**: Respects paragraph boundaries, good default
- **Cons**: Still content-agnostic, no semantic understanding
- **When to use**: General-purpose text, articles, documentation

## Semantic Chunking

Split at topic boundaries detected by embedding similarity.

- **How it works**: Embed each sentence, compute cosine similarity between adjacent sentences. Split where similarity drops below a threshold.
- **Implementation**: LlamaIndex SemanticSplitterNodeParser, or custom using sentence-transformers
- **Pros**: Chunks are topically coherent, boundaries align with content shifts
- **Cons**: Slower (requires embedding every sentence), chunk sizes vary widely
- **When to use**: Long-form prose, articles, reports where topic shifts matter
- **Threshold tuning**: Start at 0.5, lower for more splits, higher for fewer

## Hierarchical / Parent-Child Chunking

Create two levels: large parent chunks and small child chunks.

- **How it works**: Split into large parents (1024 tokens), then split each parent into small children (256 tokens). Index children for retrieval, but return the parent for context.
- **Implementation**: LangChain ParentDocumentRetriever
- **Pros**: Precise retrieval (small chunks match well) + rich context (parent provides surrounding info)
- **Cons**: More complex indexing, storage overhead
- **When to use**: Legal documents, scientific papers, any content where context around a match matters
- **Key insight**: Retrieval precision improves with smaller chunks, but LLM comprehension improves with larger context. This strategy gets both.

## Document-Structure-Aware Chunking

Split according to the document's natural structure.

### Markdown/HTML: Split at headings
- Use heading hierarchy (H1 > H2 > H3) as chunk boundaries
- Each chunk inherits its heading path as metadata
- Implementation: LangChain MarkdownHeaderTextSplitter

### Code: Split at function/class boundaries
- Use AST parsing to identify function and class definitions
- Each function/method becomes a chunk, with class context in metadata
- Implementation: LangChain Language-aware splitter with tree-sitter

### Legal: Split at section/clause boundaries
- Detect "Section", "Article", "Clause" markers
- Preserve hierarchical numbering (1.2.3) in metadata
- Keep cross-references intact within chunks

### Chat/Transcripts: Split by speaker turns
- Group consecutive messages from the same speaker
- Or split by topic shift within conversation
- Preserve speaker identity and timestamps in metadata

## Chunk Size Guidelines

| Use Case | Recommended Size | Overlap | Rationale |
|----------|-----------------|---------|-----------|
| Q&A | 256-512 tokens | 10-20% | Precision matters — smaller chunks match specific questions |
| Summarization | 512-1024 tokens | 15-20% | Broader context needed for coherent summaries |
| Code retrieval | 100-300 tokens | ~function | Natural boundaries (functions) are better than token counts |
| Legal analysis | 512-1024 tokens | 10% | Sections are natural units; needs context for interpretation |
| Semantic search | 256-512 tokens | 10% | Smaller chunks improve ranking precision |

## Overlap

Overlap prevents information loss at chunk boundaries.

- **Typical range**: 10-20% of chunk size
- **Too little**: Important sentences at boundaries get split
- **Too much**: Redundant retrieval, wasted context window
- **Zero overlap is fine for**: Tabular data, code (when using AST boundaries), chat (when splitting by turn)

## CJK Language Considerations

Chinese, Japanese, and Korean text requires special handling:
- **Character-based counting**: CJK characters carry more meaning per character than Latin scripts
- **Reduce chunk size by 30%**: A 512-character CJK chunk contains roughly as much information as a 700-token English chunk
- **Use language-specific tokenizers**: jieba (Chinese), MeCab/sudachi (Japanese), mecab (Korean)
- **Sentence boundaries differ**: CJK uses different sentence-ending punctuation (。！？ vs .!?)
