import os
import sys
from unstructured.partition.pdf import partition_pdf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.embedder import upsert_document
import httpx

def is_api_running() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

def _local_ingest_pdf(filepath: str, metadata: dict, thorough: bool = True):
    print(f"Starting advanced partitioning for {filepath}...")
    
    # Partitioning logic
    elements = partition_pdf(
        filename=filepath,
        strategy="hi_res" if thorough else "fast",
    )
    
    chunks = []
    current_chunk = ""
    for el in elements:
        if hasattr(el, 'text'):
            if len(current_chunk) + len(el.text) > 2000: # Max chunk size
                chunks.append(current_chunk)
                current_chunk = el.text
            else:
                current_chunk += "\n" + el.text
    if current_chunk:
        chunks.append(current_chunk)
        
    print(f"Ingesting {filepath}: {len(chunks)} high-fidelity chunks")
    for i, chunk in enumerate(chunks):
        chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks), "method": "unstructured"}
        upsert_document(chunk, chunk_meta)
    print(f"Done: {filepath}")

def ingest_pdf(filepath: str, metadata: dict, thorough: bool = True):
    """
    Advanced ingestion using unstructured.io for high-fidelity partitioning.
    Routes to API if server is active to prevent lock conflict and double models in memory.
    """
    if is_api_running():
        print(f"API server is active. Routing PDF ingestion for {filepath} through endpoint...")
        try:
            r = httpx.post("http://127.0.0.1:8000/api/ingest/pdf", json={
                "filepath": os.path.abspath(filepath),
                "metadata": metadata,
                "thorough": thorough
            }, timeout=180.0)
            r.raise_for_status()
            print(f"API successfully ingested {filepath}")
            return
        except Exception as e:
            print(f"API Ingestion failed, falling back to local database write: {e}")

    _local_ingest_pdf(filepath, metadata, thorough)

def _local_ingest_text(text: str, metadata: dict):
    # Basic chunking for plain text
    words = text.split()
    chunks = []
    chunk_size = 500
    overlap = 50
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 100:
            chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            upsert_document(chunk, chunk_meta)

def ingest_text(text: str, metadata: dict):
    if is_api_running():
        try:
            r = httpx.post("http://127.0.0.1:8000/api/ingest", json={
                "text": text,
                "metadata": metadata
            }, timeout=300.0)
            r.raise_for_status()
            return
        except Exception as e:
            print(f"API Ingestion failed, falling back to local database write: {e}")

    _local_ingest_text(text, metadata)

if __name__ == "__main__":
    # Test ingestion with a sample text
    sample = """
    The UAE Federal Tax Authority (FTA) administers Value Added Tax (VAT) 
    at a standard rate of 5% on most goods and services. Corporate Tax was 
    introduced in June 2023 at 9% on taxable income exceeding AED 375,000. 
    """
    ingest_text(sample, {
        "source": "sample_v2",
        "source_type": "statute",
        "jurisdiction": "UAE",
        "doc_title": "Tax Overview UAE 2024",
        "date": "2024-01-01",
        "url": ""
    })
    print("Sample V2 ingestion complete.")