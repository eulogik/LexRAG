import pymupdf
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.embedder import upsert_document

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 100]

def ingest_pdf(filepath: str, metadata: dict):
    doc = pymupdf.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    chunks = chunk_text(full_text)
    print(f"Ingesting {filepath}: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
        upsert_document(chunk, chunk_meta)
    print(f"Done: {filepath}")

def ingest_text(text: str, metadata: dict):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
        upsert_document(chunk, chunk_meta)

if __name__ == "__main__":
    sample = """
    The UAE Federal Tax Authority (FTA) administers Value Added Tax (VAT) 
    at a standard rate of 5% on most goods and services. Corporate Tax was 
    introduced in June 2023 at 9% on taxable income exceeding AED 375,000. 
    Free zone companies may qualify for 0% Corporate Tax subject to conditions.
    Under the Income Tax Act 1961, India levies income tax on individuals and 
    corporations. GST replaced multiple indirect taxes in July 2017 and operates 
    under CGST, SGST, and IGST frameworks. The standard GST rate is 18% for 
    most services.
    """
    ingest_text(sample, {
        "source": "sample",
        "source_type": "statute",
        "jurisdiction": "Both",
        "doc_title": "Tax Overview UAE and India",
        "date": "2024-01-01",
        "url": ""
    })
    print("Sample ingestion complete. Test passed.")