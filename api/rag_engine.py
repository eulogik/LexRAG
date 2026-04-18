import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.embedder import search
import httpx
import json

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "qwen/qwen3-32b"

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")


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

def query_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
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
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data["message"]["content"]
        if "<think>" in answer:
            answer = answer.split("</think>")[-1].strip()
    return answer

def query_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

def query_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

def query_provider(prompt: str, provider: str) -> str:
    if provider == "ollama":
        return query_ollama(prompt)
    elif provider == "groq":
        return query_groq(prompt)
    elif provider == "openrouter":
        return query_openrouter(prompt)
    else:
        return query_openrouter(prompt)

def query_rag(question: str, jurisdiction: str = None, source_type: str = None, top_k: int = 5, provider: str = None) -> dict:
    provider = provider or LLM_PROVIDER
    
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
    
    try:
        answer = query_provider(prompt, provider)
    except Exception as e:
        answer = f"Model error: {str(e)}. Check your API key and provider settings."
    
    sources = [
        {"title": d["doc_title"], "source": d["source"], "jurisdiction": d["jurisdiction"], 
         "type": d["source_type"], "url": d["url"], "score": round(d["score"], 3)}
        for d in context_docs
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "context_used": len(context_docs),
        "provider": provider
    }

if __name__ == "__main__":
    result = query_rag("What is the VAT rate in UAE and how does it compare to GST in India?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:", result["sources"])
    print("\nPROVIDER:", result.get("provider", "openrouter"))