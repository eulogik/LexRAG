from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, Fusion, FusionQuery, Prefetch
import uuid
import json
import os

COLLECTION_NAME = "lexrag_docs_v2"
QDRANT_URL = "http://localhost:6333"

# Memory-efficient models for 1GB RAM server
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = LexEmbedder()
    return _embedder

class LexEmbedder:
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(root_dir, "embeddings_cache")
        os.makedirs(cache_path, exist_ok=True)
        
        print(f"Initializing LexEmbedder with {DENSE_MODEL} and {SPARSE_MODEL}...")
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL, cache_dir=cache_path)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=cache_path)
        storage_path = os.path.join(root_dir, "qdrant_storage")
        self.client = QdrantClient(path=storage_path)

    def ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=1024, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=None)
                }
            )
            print(f"Created Hybrid Collection: {COLLECTION_NAME}")
        else:
            print(f"Hybrid Collection exists: {COLLECTION_NAME}")

    def embed(self, text: str):
        dense_vec = list(self.dense_model.embed([f"passage: {text}"]))[0].tolist()
        sparse_vec = list(self.sparse_model.embed([text]))[0]
        return dense_vec, sparse_vec

    def upsert_document(self, text: str, metadata: dict):
        self.ensure_collection()
        doc_id = str(uuid.uuid4())
        dense_vec, sparse_vec = self.embed(text)
        
        sparse_vector_data = {
            "indices": sparse_vec.indices.tolist(),
            "values": sparse_vec.values.tolist()
        }

        point = PointStruct(
            id=doc_id,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vector_data
            },
            payload={**metadata, "text": text}
        )
        self.client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return doc_id

    def search(self, query: str, top_k: int = 15, filters: dict = None) -> list:
        self.ensure_collection()
        dense_query = list(self.dense_model.embed([f"query: {query}"]))[0].tolist()
        sparse_query_vec = list(self.sparse_model.embed([query]))[0]
        
        sparse_query = {
            "indices": sparse_query_vec.indices.tolist(),
            "values": sparse_query_vec.values.tolist()
        }

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        # Hybrid Search using Prefetch and Fusion
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=top_k
                ),
                Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=top_k
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        ).points

        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source": r.payload.get("source", ""),
                "source_type": r.payload.get("source_type", ""),
                "jurisdiction": r.payload.get("jurisdiction", ""),
                "date": r.payload.get("date", ""),
                "doc_title": r.payload.get("doc_title", ""),
                "url": r.payload.get("url", "")
            }
            for r in results
        ]

# Lazy initialization wrappers
def ensure_collection(): return get_embedder().ensure_collection()
def upsert_document(text, meta): return get_embedder().upsert_document(text, meta)
def search(query, top_k=5, filters=None): return get_embedder().search(query, top_k, filters)
