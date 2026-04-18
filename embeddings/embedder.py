from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
import json

COLLECTION_NAME = "lexrag_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
QDRANT_URL = "http://localhost:6333"
VECTOR_DIM = 768

model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(url=QDRANT_URL)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

def embed_text(text: str) -> list:
    return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

def embed_query(query: str) -> list:
    return model.encode(f"query: {query}", normalize_embeddings=True).tolist()

def upsert_document(text: str, metadata: dict):
    ensure_collection()
    doc_id = str(uuid.uuid4())
    vector = embed_text(text)
    point = PointStruct(
        id=doc_id,
        vector=vector,
        payload={**metadata, "text": text}
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return doc_id

def search(query: str, top_k: int = 5, filters: dict = None) -> list:
    ensure_collection()
    import httpx
    query_vector = embed_query(query)
    
    json_payload = {
        "vector": query_vector,
        "top": top_k,
        "with_payload": True
    }
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append({"key": key, "match": {"value": value}})
        json_payload["filter"] = {"must": conditions}
    
    with httpx.Client(timeout=30.0) as http_client:
        resp = http_client.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json=json_payload
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", [])
    
    return [
        {
            "text": r.get("payload", {}).get("text", ""),
            "score": r.get("score", 0),
            "source": r.get("payload", {}).get("source", ""),
            "source_type": r.get("payload", {}).get("source_type", ""),
            "jurisdiction": r.get("payload", {}).get("jurisdiction", ""),
            "date": r.get("payload", {}).get("date", ""),
            "doc_title": r.get("payload", {}).get("doc_title", ""),
            "url": r.get("payload", {}).get("url", "")
        }
        for r in results
    ]