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
    jurisdiction: Optional[str] = "Both"
    source_type: Optional[str] = "All"
    top_k: Optional[int] = 5
    provider: Optional[str] = "openrouter"

class QueryResponse(BaseModel):
    answer: str
    sources: list
    context_used: int
    provider: Optional[str] = "openrouter"

@app.get("/")
def root():
    return {"status": "LexRAG is running", "default_provider": "openrouter", "vector_db": "qdrant"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = query_rag(req.question, req.jurisdiction, req.source_type, req.top_k, req.provider)
    return QueryResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok"}