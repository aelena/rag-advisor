"""Scale and constraint-based vector database recommender."""

from __future__ import annotations

from rag_adviser.config import load_defaults
from rag_adviser.models import (
    HardwareConstraints,
    PrivacyLevel,
    UpdateFrequency,
    VectorDBRecommendation,
)


class VectorDBRecommender:
    """Select the best vector database based on scale, constraints, and usage patterns."""

    def __init__(self) -> None:
        defaults = load_defaults()
        self._db_profiles: list[dict] = defaults.get("vector_db_profiles", [])
        self._dimension: int = 384

    def recommend(
        self,
        doc_count: int,
        constraints: HardwareConstraints,
        update_frequency: UpdateFrequency = UpdateFrequency.NEVER,
        estimated_chunks: int | None = None,
        embedding_dimension: int = 0,
    ) -> VectorDBRecommendation:
        """Select and return the best-fit vector database.

        Args:
            doc_count: Estimated number of documents (not chunks).
            constraints: Hardware and deployment constraints.
            update_frequency: How often the corpus is updated.
            estimated_chunks: Chunk count derived from real token volume. When
                omitted a rough ~5 chunks/document heuristic is used.
            embedding_dimension: Dimension of the chosen embedding model, used
                to fill in the code snippet. 0 = unknown.

        Returns:
            VectorDBRecommendation with provider, reasons, and code snippet.
        """
        if estimated_chunks is None:
            # Fallback heuristic: ~5 chunks per document on average
            estimated_chunks = doc_count * 5
        self._dimension = embedding_dimension if embedding_dimension > 0 else 384

        scored: list[tuple[float, dict, list[str]]] = []

        for db in self._db_profiles:
            score, reasons = self._score_db(
                db, estimated_chunks, constraints, update_frequency
            )
            scored.append((score, db, reasons))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return VectorDBRecommendation(
                provider="ChromaDB",
                reason="Default recommendation (no profiles available)",
                category="embedded",
                library="chromadb",
            )

        best_score, best_db, reasons = scored[0]

        return VectorDBRecommendation(
            provider=best_db["name"],
            reason="; ".join(reasons),
            category=best_db.get("category", "embedded"),
            supports_metadata_filter=best_db.get("supports_metadata_filter", True),
            supports_hybrid_search=best_db.get("supports_hybrid_search", False),
            estimated_capacity=f"~{best_db.get('max_docs', 0):,} documents",
            library=best_db.get("library", ""),
            code_snippet=self._generate_code_snippet(best_db),
        )

    def _score_db(
        self,
        db: dict,
        chunk_count: int,
        constraints: HardwareConstraints,
        update_frequency: UpdateFrequency,
    ) -> tuple[float, list[str]]:
        """Score a DB profile against requirements."""
        score = 0.0
        reasons: list[str] = []

        max_docs = db.get("max_docs", 0)
        category = db.get("category", "")
        deployment = db.get("deployment", [])

        # ── Scale fitness ──────────────────────────────────────────────────
        if chunk_count <= max_docs:
            score += 0.25
            reasons.append(
                f"Handles estimated {chunk_count:,} chunks "
                f"(capacity: {max_docs:,})"
            )
        else:
            score -= 0.5
            reasons.append(f"May not handle {chunk_count:,} chunks")

        # Bonus for right-sized DB (not massively over-provisioned)
        if max_docs > 0 and chunk_count > 0:
            ratio = max_docs / chunk_count
            if 1 <= ratio <= 100:
                score += 0.1
                reasons.append("Well-sized for your corpus")

        # ── Deployment compatibility ───────────────────────────────────────
        env_value = constraints.environment.value
        if env_value in deployment:
            score += 0.2
            reasons.append(f"Compatible with {env_value} deployment")
        else:
            score -= 0.3

        # ── Privacy compliance ─────────────────────────────────────────────
        if constraints.privacy in (PrivacyLevel.STRICT, PrivacyLevel.AIR_GAPPED):
            if category == "managed":
                score -= 1.0  # Managed services require cloud
            elif category in ("embedded", "in-memory"):
                score += 0.2
                reasons.append("Fully local, no external calls needed")
            elif category == "client-server":
                score += 0.1
                reasons.append("Can be self-hosted locally")

        # ── Update frequency ───────────────────────────────────────────────
        if update_frequency in (UpdateFrequency.DAILY, UpdateFrequency.REALTIME):
            if db.get("supports_hybrid_search"):
                score += 0.1
                reasons.append("Supports efficient upserts and hybrid search")
            if category == "in-memory":
                score -= 0.3  # No persistence or incremental deletes
                reasons.append("In-memory index must be rebuilt on every update")

        # ── Simplicity bonus for small scale ───────────────────────────────
        if chunk_count < 50_000 and category == "embedded":
            score += 0.15
            reasons.append("Simple embedded DB, no server setup needed")

        # ── Hybrid search bonus ────────────────────────────────────────────
        if db.get("supports_hybrid_search"):
            score += 0.05

        return score, reasons

    def _generate_code_snippet(self, db: dict) -> str:
        """Generate a ready-to-use code snippet for the recommended DB."""
        name = db.get("name", "").lower()
        library = db.get("library", "")
        dim = self._dimension

        snippets = {
            "chromadb": """import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"},
)

# Add documents
collection.add(
    documents=["doc1", "doc2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ids=["id1", "id2"],
)""",

            "qdrant": f"""from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size={dim}, distance=Distance.COSINE),
)""",

            "faiss": f"""import faiss
import numpy as np

dimension = {dim}  # Matches the recommended embedding model
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)

# Add vectors
vectors = np.array(embeddings, dtype="float32")
faiss.normalize_L2(vectors)
index.add(vectors)""",

            "weaviate": """import weaviate

client = weaviate.connect_to_local()
collection = client.collections.create(
    name="RagCollection",
    vectorizer_config=None,  # We provide our own vectors
)""",

            "pinecone": f"""from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")
pc.create_index(
    name="rag-index",
    dimension={dim},
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)""",

            "milvus": f"""from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema, connections,
)

connections.connect("default", host="localhost", port="19530")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim={dim}),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
]
schema = CollectionSchema(fields, description="RAG collection")
collection = Collection("rag_collection", schema)""",

            "pgvector": f"""-- SQL setup for pgvector
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector({dim})
);
-- HNSW gives better recall/latency than ivfflat and needs no training step
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);""",

            "lancedb": """import lancedb

db = lancedb.connect("./lance_db")
table = db.create_table("rag_collection", data=[
    {"text": "example", "vector": [0.1, 0.2, ...]},
])""",
        }

        return snippets.get(name, f"# pip install {library}\n# See {name} documentation")
