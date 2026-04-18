You are implementing a production-ready Legal & Accounting RAG AI system for UAE and Indian law on a MacBook Air M4. Ollama is already installed and running. Follow every step exactly in order. Do not skip any step. Do not assume anything is installed unless explicitly stated. After each major step, confirm it works before proceeding.

=== SYSTEM OVERVIEW ===
We are building: LexRAG — a retrieval-augmented AI for UAE and Indian law, accounting standards, and case law.
Stack: Python 3.11, Qdrant (vector DB via Docker), Qwen3:14b via Ollama, multilingual-e5-base embeddings, FastAPI backend, Streamlit UI, daily scraper pipeline.

=== PREREQUISITES CHECK ===

Step 0: Verify environment
Run these commands one by one and confirm each succeeds:

  ollama --version
  # Expected: some version number. If error, Ollama is not installed — stop and report.

  ollama list
  # Shows installed models

  docker --version
  # If Docker not installed, install it:
  # Download Docker Desktop for Mac (Apple Silicon): https://desktop.docker.com/mac/main/arm64/Docker.dmg
  # Install it, open Docker Desktop, wait for it to say "Docker is running"

  python3 --version
  # Must be 3.10 or higher. If not:
  # brew install python@3.11
  # Then use python3.11 everywhere below instead of python3

=== STEP 1: Pull the model ===

Run:
  ollama pull qwen3:14b
# This downloads 9.3GB. Wait for it to complete fully.
# Verify: ollama list should show qwen3:14b

=== STEP 2: Start Qdrant vector database ===

Run:
  docker pull qdrant/qdrant
  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Verify Qdrant is running:
  curl http://localhost:6333/
# Expected response: {"title":"qdrant","version":"..."}
# If Docker not running, open Docker Desktop app first

=== STEP 3: Create project structure ===

Run these commands exactly:
  mkdir -p {data/raw,data/processed,scrapers,embeddings,api,ui,scripts}
  cd LexRAG
  python3 -m venv venv
  source venv/bin/activate

# Now install ALL dependencies in one command:
  pip install \
    langchain==0.3.25 \
    langchain-community==0.3.23 \
    langchain-ollama==0.3.3 \
    qdrant-client==1.13.3 \
    sentence-transformers==3.4.1 \
    pymupdf==1.25.5 \
    beautifulsoup4==4.13.3 \
    requests==2.32.3 \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    streamlit==1.45.0 \
    python-dotenv==1.1.0 \
    schedule==1.2.2 \
    httpx==0.28.1 \
    lxml==5.3.2 \
    pydantic==2.11.4

# Verify no errors. If any package fails, run pip install <that-package> individually.

=== STEP 4: Create the embedding module ===

Create file: embeddings/embedder.py
Exact content:

---FILE START: embeddings/embedder.py---
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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
    query_vector = embed_query(query)
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        qdrant_filter = Filter(must=conditions)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True
    )
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
---FILE END---

=== STEP 5: Create the document ingestion module ===

Create file: scripts/ingest.py
Exact content:

---FILE START: scripts/ingest.py---
import fitz  # pymupdf
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
    doc = fitz.open(filepath)
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
    # Test ingestion with a sample text
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
---FILE END---

Test it:
  cd LexRAG
  source venv/bin/activate
  python3 scripts/ingest.py
# Expected: "Sample ingestion complete. Test passed."

=== STEP 6: Create the scrapers ===

Create file: scrapers/uae_scraper.py
Exact content:

---FILE START: scrapers/uae_scraper.py---
import requests
from bs4 import BeautifulSoup
import os
import sys
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest import ingest_text

SEEN_HASHES_FILE = os.path.join(os.path.dirname(__file__), "seen_hashes_uae.txt")

def load_seen():
    if not os.path.exists(SEEN_HASHES_FILE):
        return set()
    with open(SEEN_HASHES_FILE) as f:
        return set(line.strip() for line in f)

def save_hash(h: str):
    with open(SEEN_HASHES_FILE, "a") as f:
        f.write(h + "\n")

def scrape_fta_updates():
    """Scrapes UAE FTA latest announcements page"""
    url = "https://www.tax.gov.ae/en/legislation.aspx"
    seen = load_seen()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        items = soup.find_all("a", href=True)
        count = 0
        for item in items:
            text = item.get_text(strip=True)
            if len(text) < 30:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "FTA UAE",
                "source_type": "ruling",
                "jurisdiction": "UAE",
                "doc_title": text[:100],
                "date": "scraped",
                "url": "https://www.tax.gov.ae" + item["href"]
            })
            save_hash(h)
            count += 1
        print(f"FTA UAE: {count} new items ingested")
    except Exception as e:
        print(f"FTA scrape error: {e}")

def scrape_moj_uae():
    """Scrapes UAE Ministry of Justice legislation page"""
    url = "https://uaelegislation.gov.ae/en/legislations"
    seen = load_seen()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        links = soup.find_all("a", href=True)
        count = 0
        for link in links:
            text = link.get_text(strip=True)
            if len(text) < 20:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "UAE Legislation Portal",
                "source_type": "statute",
                "jurisdiction": "UAE",
                "doc_title": text[:100],
                "date": "scraped",
                "url": link["href"] if link["href"].startswith("http") else "https://uaelegislation.gov.ae" + link["href"]
            })
            save_hash(h)
            count += 1
        print(f"MOJ UAE: {count} new items ingested")
    except Exception as e:
        print(f"MOJ UAE scrape error: {e}")

if __name__ == "__main__":
    scrape_fta_updates()
    scrape_moj_uae()
---FILE END---

Create file: scrapers/india_scraper.py
Exact content:

---FILE START: scrapers/india_scraper.py---
import requests
from bs4 import BeautifulSoup
import os
import sys
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest import ingest_text

SEEN_HASHES_FILE = os.path.join(os.path.dirname(__file__), "seen_hashes_india.txt")
INDIAN_KANOON_API = "https://api.indiankanoon.org/search/"
# Indian Kanoon API: Free, register at https://api.indiankanoon.org to get a token
# Set your token as environment variable: export INDIANKANOON_TOKEN=your_token_here
INDIANKANOON_TOKEN = os.environ.get("INDIANKANOON_TOKEN", "")

def load_seen():
    if not os.path.exists(SEEN_HASHES_FILE):
        return set()
    with open(SEEN_HASHES_FILE) as f:
        return set(line.strip() for line in f)

def save_hash(h: str):
    with open(SEEN_HASHES_FILE, "a") as f:
        f.write(h + "\n")

def scrape_cbic_gst():
    """Scrapes CBIC GST circulars and notifications"""
    url = "https://cbic-gst.gov.in/gst-goods-services-rates.html"
    seen = load_seen()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        items = soup.find_all(["p", "li", "td"])
        count = 0
        for item in items:
            text = item.get_text(strip=True)
            if len(text) < 50:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "CBIC GST India",
                "source_type": "ruling",
                "jurisdiction": "India",
                "doc_title": text[:100],
                "date": "scraped",
                "url": url
            })
            save_hash(h)
            count += 1
        print(f"CBIC GST India: {count} new items ingested")
    except Exception as e:
        print(f"CBIC scrape error: {e}")

def fetch_indian_kanoon_cases(query: str, num_results: int = 10):
    """
    Fetches case law from Indian Kanoon API.
    API docs: https://api.indiankanoon.org
    Register free at: https://api.indiankanoon.org/api/
    """
    if not INDIANKANOON_TOKEN:
        print("INDIANKANOON_TOKEN not set. Skip. Get free token at https://api.indiankanoon.org/api/")
        return
    seen = load_seen()
    try:
        resp = requests.post(
            INDIAN_KANOON_API,
            data={"formInput": query, "pagenum": 0},
            headers={"Authorization": f"Token {INDIANKANOON_TOKEN}"},
            timeout=15
        )
        data = resp.json()
        docs = data.get("docs", [])
        count = 0
        for doc in docs[:num_results]:
            text = doc.get("headline", "") + " " + doc.get("title", "")
            if len(text) < 30:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "Indian Kanoon",
                "source_type": "case",
                "jurisdiction": "India",
                "doc_title": doc.get("title", "")[:100],
                "date": doc.get("publishdate", ""),
                "url": f"https://indiankanoon.org/doc/{doc.get('tid', '')}/"
            })
            save_hash(h)
            count += 1
        print(f"Indian Kanoon [{query}]: {count} new cases ingested")
    except Exception as e:
        print(f"Indian Kanoon error: {e}")

if __name__ == "__main__":
    scrape_cbic_gst()
    fetch_indian_kanoon_cases("GST input tax credit")
    fetch_indian_kanoon_cases("income tax section 80C deduction")
    fetch_indian_kanoon_cases("VAT tribunal ruling")
---FILE END---

=== STEP 7: Create the RAG engine ===

Create file: api/rag_engine.py
Exact content:

---FILE START: api/rag_engine.py---
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.embedder import search
import httpx
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:14b"

SYSTEM_PROMPT = """You are LexRAG, an expert AI assistant specializing in UAE and Indian law, taxation, accounting, and case law. 

You answer questions using ONLY the provided context documents. Always:
1. Cite the source document title and jurisdiction in your answer
2. If the question involves complex legal reasoning, think step by step
3. If context is insufficient, say "I don't have enough information in my knowledge base for this specific query" — never hallucinate
4. For case law questions, reference specific case names and decisions from the context
5. For accounting queries, reference specific standards (IFRS, Ind AS, UAE GAAP) cited in the context
6. Always mention if a law has a specific effective date from the context

/no_think for simple factual lookups. Use reasoning for complex multi-step legal analysis."""

def build_prompt(query: str, context_docs: list) -> str:
    context_str = "\n\n---\n\n".join([
        f"Source: {d['source']} | Type: {d['source_type']} | Jurisdiction: {d['jurisdiction']} | Date: {d['date']}\nTitle: {d['doc_title']}\nURL: {d['url']}\n\nContent:\n{d['text']}"
        for d in context_docs
    ])
    return f"""Use the following legal/accounting documents to answer the question.

CONTEXT DOCUMENTS:
{context_str}

QUESTION: {query}

Answer based strictly on the above context. Cite sources explicitly."""

def query_rag(question: str, jurisdiction: str = None, source_type: str = None, top_k: int = 5) -> dict:
    filters = {}
    if jurisdiction and jurisdiction != "Both":
        filters["jurisdiction"] = jurisdiction
    if source_type and source_type != "All":
        filters["source_type"] = source_type
    
    context_docs = search(question, top_k=top_k, filters=filters if filters else None)
    
    if not context_docs:
        return {
            "answer": "No relevant documents found in the knowledge base for this query. Please add more documents or broaden your search.",
            "sources": [],
            "context_used": 0
        }
    
    prompt = build_prompt(question, context_docs)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    }
    
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            answer = data["message"]["content"]
            # Strip thinking tags if present
            if "<think>" in answer:
                answer = answer.split("</think>")[-1].strip()
    except Exception as e:
        answer = f"Model error: {str(e)}. Make sure Ollama is running and qwen3:14b is pulled."
    
    sources = [
        {"title": d["doc_title"], "source": d["source"], "jurisdiction": d["jurisdiction"], 
         "type": d["source_type"], "url": d["url"], "score": round(d["score"], 3)}
        for d in context_docs
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "context_used": len(context_docs)
    }

if __name__ == "__main__":
    # Quick test
    result = query_rag("What is the VAT rate in UAE and how does it compare to GST in India?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:", result["sources"])
---FILE END---

Test it:
  cd LexRAG
  source venv/bin/activate
  python3 api/rag_engine.py
# Should print an answer about VAT/GST. If Ollama error, check: ollama list (qwen3:14b must be there)

=== STEP 8: Create FastAPI backend ===

Create file: api/main.py
Exact content:

---FILE START: api/main.py---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.rag_engine import query_rag

app = FastAPI(title="LexRAG API", description="UAE and India Legal & Accounting AI", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    jurisdiction: Optional[str] = "Both"  # "UAE", "India", "Both"
    source_type: Optional[str] = "All"    # "statute", "case", "ruling", "All"
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    context_used: int

@app.get("/")
def root():
    return {"status": "LexRAG is running", "model": "qwen3:14b", "vector_db": "qdrant"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = query_rag(req.question, req.jurisdiction, req.source_type, req.top_k)
    return QueryResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok"}
---FILE END---

=== STEP 9: Create Streamlit UI ===

Create file: ui/app.py
Exact content:

---FILE START: ui/app.py---
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.rag_engine import query_rag

st.set_page_config(page_title="LexRAG — UAE & India Legal AI", page_icon="⚖️", layout="wide")
st.title("⚖️ LexRAG — Legal & Accounting AI")
st.caption("UAE and Indian Law | Taxation | Case Law | Accounting Standards")

with st.sidebar:
    st.header("Search Options")
    jurisdiction = st.selectbox("Jurisdiction", ["Both", "UAE", "India"])
    source_type = st.selectbox("Document Type", ["All", "statute", "case", "ruling"])
    top_k = st.slider("Documents to retrieve", 3, 10, 5)
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- UAE FTA: [tax.gov.ae](https://www.tax.gov.ae)")
    st.markdown("- UAE Laws: [uaelegislation.gov.ae](https://uaelegislation.gov.ae)")
    st.markdown("- India GST: [cbic-gst.gov.in](https://cbic-gst.gov.in)")
    st.markdown("- Case Law: [indiankanoon.org](https://indiankanoon.org)")

question = st.text_area("Ask a legal or accounting question", height=100,
    placeholder="e.g. What are the VAT exemptions for healthcare in UAE? / What is the penalty for late GST filing in India?")

if st.button("Ask LexRAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base and generating answer..."):
            result = query_rag(question, jurisdiction, source_type if source_type != "All" else None, top_k)
        
        st.subheader("Answer")
        st.markdown(result["answer"])
        
        if result["sources"]:
            st.subheader(f"Sources Used ({result['context_used']} documents)")
            for s in result["sources"]:
                with st.expander(f"📄 {s['title'][:80]} | {s['jurisdiction']} | Score: {s['score']}"):
                    st.write(f"**Source:** {s['source']}")
                    st.write(f"**Type:** {s['type']}")
                    if s["url"]:
                        st.write(f"**URL:** {s['url']}")
---FILE END---

=== STEP 10: Create the daily scheduler ===

Create file: scripts/daily_update.py
Exact content:

---FILE START: scripts/daily_update.py---
import schedule
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.uae_scraper import scrape_fta_updates, scrape_moj_uae
from scrapers.india_scraper import scrape_cbic_gst, fetch_indian_kanoon_cases

def run_all_scrapers():
    print("=== Daily LexRAG Update Started ===")
    scrape_fta_updates()
    scrape_moj_uae()
    scrape_cbic_gst()
    fetch_indian_kanoon_cases("corporate tax UAE 2024")
    fetch_indian_kanoon_cases("GST input credit")
    fetch_indian_kanoon_cases("income tax penalty")
    fetch_indian_kanoon_cases("VAT exemption")
    print("=== Daily Update Complete ===")

# Run immediately once on start
run_all_scrapers()

# Then schedule daily at 2 AM
schedule.every().day.at("02:00").do(run_all_scrapers)

print("Scheduler running. Will update daily at 2 AM. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)
---FILE END---

=== STEP 11: Add PDF ingestion for bulk documents ===

Create file: scripts/bulk_ingest_pdfs.py
Exact content:

---FILE START: scripts/bulk_ingest_pdfs.py---
"""
Drop PDF files into data/raw/ folder.
Run this script to ingest all of them.
Name files like: uae_vat_law_2018.pdf, india_gst_act_2017.pdf
The filename is used as the document title.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest import ingest_pdf

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

JURISDICTION_HINTS = {
    "uae": "UAE", "dubai": "UAE", "adgm": "UAE", "difc": "UAE",
    "fta": "UAE", "cbuae": "UAE",
    "india": "India", "cbic": "India", "gst": "India", "incometax": "India",
    "sebi": "India", "rbi": "India"
}
SOURCE_TYPE_HINTS = {
    "act": "statute", "law": "statute", "decree": "statute", "regulation": "statute",
    "case": "case", "judgment": "case", "ruling": "ruling", "circular": "ruling",
    "notification": "ruling", "order": "ruling"
}

def guess_meta(filename: str) -> dict:
    name_lower = filename.lower()
    jurisdiction = "Both"
    for hint, j in JURISDICTION_HINTS.items():
        if hint in name_lower:
            jurisdiction = j
            break
    source_type = "statute"
    for hint, t in SOURCE_TYPE_HINTS.items():
        if hint in name_lower:
            source_type = t
            break
    return {
        "source": "Bulk PDF Import",
        "source_type": source_type,
        "jurisdiction": jurisdiction,
        "doc_title": filename.replace("_", " ").replace(".pdf", ""),
        "date": "imported",
        "url": ""
    }

if __name__ == "__main__":
    pdfs = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Drop your PDF files there and rerun.")
    for pdf in pdfs:
        meta = guess_meta(pdf)
        print(f"Processing: {pdf} → jurisdiction={meta['jurisdiction']}, type={meta['source_type']}")
        ingest_pdf(os.path.join(RAW_DIR, pdf), meta)
    print(f"Done. {len(pdfs)} PDFs ingested.")
---FILE END---

=== STEP 12: Launch everything ===

Open THREE terminal tabs. In each tab, run:

TAB 1 — FastAPI backend:
  cd LexRAG && source venv/bin/activate && uvicorn api.main:app --reload --port 8000
  # Visit http://localhost:8000/docs to see API docs

TAB 2 — Streamlit UI:
  cd LexRAG && source venv/bin/activate && streamlit run ui/app.py --server.port 8501
  # Visit http://localhost:8501 to use the app

TAB 3 — Daily scraper (optional, run when ready):
  cd LexRAG && source venv/bin/activate && python3 scripts/daily_update.py

=== STEP 13: Add your own PDF documents ===

Drop any UAE or India legal PDF files into: data/raw/
Then run:
  cd ~/lexrag && source venv/bin/activate && python3 scripts/bulk_ingest_pdfs.py

Recommended PDFs to download and add:
- UAE VAT Law (Federal Decree-Law No. 8 of 2017): https://www.tax.gov.ae/en/legislation/vat.aspx
- UAE Corporate Tax Law (Federal Decree-Law No. 47 of 2022): https://www.tax.gov.ae/en/legislation/corporate.tax.aspx
- India GST Act 2017: https://cbic-gst.gov.in/gst-acts.html
- India Income Tax Act 1961: https://incometaxindia.gov.in/pages/acts/income-tax-act.aspx
- IFRS Standards (free with registration): https://www.ifrs.org/issued-standards/list-of-standards/

=== STEP 14: Get Indian Kanoon API token (free) ===

1. Go to: https://api.indiankanoon.org/api/
2. Register with your email
3. You will receive a token
5. Add these keys to your `.env` file in the project root:
   INDIANKANOON_TOKEN=7f55bbf6...
   GROQ_API_KEY=gsk_...
   OPENROUTER_API_KEY=sk-or-v1-...
   LLM_PROVIDER=openrouter

=== STEP 15: HuggingFace publishing ===

  pip install huggingface_hub
  huggingface-cli login
  # Enter your HuggingFace token from: https://huggingface.co/settings/tokens

Create a new Space at: https://huggingface.co/new-space
  - Owner: evolucent-ai (create org at https://huggingface.co/organizations/new)
  - Space name: lexrag-uae-india
  - SDK: Streamlit
  - License: Apache 2.0

Then push:
  cd LexRAG
  git init
  git add .
  git commit -m "LexRAG v1.0 — UAE and India Legal AI by Evolucent AI"
  huggingface-cli repo create lexrag-uae-india --type space --space_sdk streamlit
  git remote add hf https://huggingface.co/spaces/evolucent-ai/lexrag-uae-india
  git push hf main

=== VERIFICATION CHECKLIST ===

After all steps, verify:
[ ] ollama list shows qwen3:14b
[ ] curl http://localhost:6333/ returns Qdrant version
[ ] python3 scripts/ingest.py prints "Sample ingestion complete"
[ ] python3 api/rag_engine.py prints an answer about VAT/GST
[ ] http://localhost:8000/docs shows FastAPI swagger UI
[ ] http://localhost:8501 shows LexRAG Streamlit app
[ ] Asking "What is VAT rate in UAE?" returns a sourced answer

=== TROUBLESHOOTING ===

Problem: "model not found" error
Fix: run → ollama pull qwen3:14b

Problem: Qdrant connection refused
Fix: run → docker start qdrant (if container exists) OR re-run the docker run command from Step 2

Problem: pip install fails on sentence-transformers
Fix: run → pip install --upgrade pip → then retry

Problem: "No module named X"
Fix: make sure you ran "source venv/bin/activate" in that terminal tab

Problem: Ollama slow on M4 MacBook Air
Expected: qwen3:14b runs at ~15-25 tokens/sec on M4. Normal. For faster dev, use qwen3:8b instead (6.5GB).

Problem: Empty answers / no context
Fix: Run the scrapers first (Step 12, TAB 3) or ingest PDFs (Step 13) to populate the knowledge base.