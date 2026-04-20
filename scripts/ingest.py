import os
import sys
from unstructured.partition.pdf import partition_pdf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.embedder import upsert_document

def ingest_pdf(filepath: str, metadata: dict, thorough: bool = True):
    """
    Advanced ingestion using unstructured.io for high-fidelity partitioning.
    """
    print(f"Starting advanced partitioning for {filepath}...")
    
    # Partitioning logic
    elements = partition_pdf(
        filename=filepath,
        # thorough=True uses ML models for layout detection, slower but much better for legal docs
        strategy="hi_res" if thorough else "fast",
        # Only extract text for now, but unstructured can do tables too
    )
    
    # Group elements into chunks (unstructured does some of this, but we'll combine for RAG)
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

def ingest_text(text: str, metadata: dict):
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
        "jurisdiction": "Both",
        "doc_title": "Tax Overview UAE 2024",
        "date": "2024-01-01",
        "url": ""
    })
    print("Sample V2 ingestion complete.")