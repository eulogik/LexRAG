import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedder import search
from sentence_transformers import CrossEncoder
from api.utils import detect_jurisdiction, auto_context_depth, tier_sources, strip_think_tags

# ─── Reranker ────────────────────────────────────────────────────────────────
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        print("Initializing Reranker (Lazy)...")
        _reranker = CrossEncoder('BAAI/bge-reranker-base')
    return _reranker

# ─── Provider Config ──────────────────────────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = "meta-llama/llama-3.3-70b-instruct:free"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"


LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LexRAG, an expert AI counsel for UAE and Indian law, taxation, and accounting.

RULES:
1. If context documents are provided and relevant, answer ONLY from them. Cite source title and jurisdiction.
2. If context is insufficient or the question is off-topic, answer helpfully using your knowledge. Prefix ALL such paragraphs with [INDEPENDENT ANALYSIS].
3. Be concise and structured. Use bullet points for lists of rules or rates.
4. Always end your response with a one-line tag: JURISDICTION: India | UAE | Both | General
5. Never fabricate statute numbers or case names."""

# ─── Retrieval + Reranking ────────────────────────────────────────────────────
def search_and_rerank(question: str, jurisdiction: str = None, top_k: int = 5) -> list:
    filters = {}
    if jurisdiction and jurisdiction != "Both":
        filters["jurisdiction"] = [jurisdiction, "Both"]
    initial = search(question, top_k=20, filters=filters if filters else None)
    if not initial:
        return []
    try:
        pairs  = [[question, d["text"]] for d in initial]
        scores = get_reranker().predict(pairs)
        for i, s in enumerate(scores):
            initial[i]["rerank_score"] = float(s)
        ranked = sorted(initial, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        print(f"Rerank error: {e}")
        return initial[:top_k]

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_prompt(query: str, context_docs: list, history: list = None, confidence: str = "GROUNDED") -> str:
    if context_docs:
        ctx = "\n\n---\n\n".join([
            f"[Source: {d['source']} | Jurisdiction: {d['jurisdiction']} | Date: {d.get('date','')}]\n"
            f"Title: {d['doc_title']}\n\n{d['text']}"
            for d in context_docs
        ])
    else:
        ctx = "No relevant documents found."

    hist_str = ""
    if history:
        hist_str = "CONVERSATION HISTORY:\n" + "\n".join(
            f"{h['role'].upper()}: {h['content']}" for h in history
        ) + "\n\n"

    fallback = ""
    if confidence == "SYNTHESIZED":
        fallback = "\nNote: No strong document matches found. Provide an independent analysis based on your knowledge and mark paragraphs with [INDEPENDENT ANALYSIS].\n"

    return f"""{hist_str}CONTEXT DOCUMENTS:
{ctx}
{fallback}
QUESTION: {query}"""

# ─── Streaming Generators ────────────────────────────────────────────────────
import httpx

async def stream_groq(messages: list, model: str = None):
    model    = model or GROQ_MODEL
    in_think = False
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": True}
        ) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                try: err_json = json.loads(err_body)
                except: err_json = {"error": {"message": err_body.decode()}}
                msg = err_json.get("error", {}).get("message", "Unknown Groq error")
                raise Exception(f"Groq API Error ({resp.status_code}): {msg}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "): continue
                data = line[6:]
                if data.strip() == "[DONE]": break
                try:
                    pdata = json.loads(data)
                    if "error" in pdata:
                        raise Exception(f"Groq Stream Error: {pdata['error'].get('message', 'Unknown')}")
                    token = pdata["choices"][0]["delta"].get("content", "")
                    if not token: continue
                    clean, in_think = strip_think_tags(token, in_think)
                    if clean: yield clean
                except Exception as e:
                    if "Stream Error" in str(e) or "API Error" in str(e): raise e
                    pass

async def stream_openrouter(messages: list, model: str = None):
    model    = model or OPENROUTER_MODEL
    in_think = False
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": True}
        ) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                try: err_json = json.loads(err_body)
                except: err_json = {"error": {"message": err_body.decode()}}
                msg = err_json.get("error", {}).get("message", "Unknown OpenRouter error")
                raise Exception(f"OpenRouter API Error ({resp.status_code}): {msg}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "): continue
                data = line[6:]
                if data.strip() == "[DONE]": break
                try:
                    pdata = json.loads(data)
                    if "error" in pdata:
                        raise Exception(f"OpenRouter Stream Error: {pdata['error'].get('message', 'Unknown')}")
                    token = pdata["choices"][0]["delta"].get("content", "")
                    if not token: continue
                    clean, in_think = strip_think_tags(token, in_think)
                    if clean: yield clean
                except Exception as e:
                    if "Stream Error" in str(e) or "API Error" in str(e): raise e
                    pass

async def stream_ollama(messages: list, model: str = None):
    model = model or OLLAMA_MODEL
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST", OLLAMA_URL,
            json={"model": model, "messages": messages, "stream": True}
        ) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                raise Exception(f"Ollama API Error ({resp.status_code}): {err_body.decode()}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token: yield token
                    if chunk.get("done"): break
                except Exception:
                    pass

async def stream_provider(messages: list, provider: str, model: str = None):
    if provider == "groq":
        async for t in stream_groq(messages, model): yield t
    elif provider == "openrouter":
        async for t in stream_openrouter(messages, model): yield t
    elif provider == "ollama":
        async for t in stream_ollama(messages, model): yield t
    else:
        async for t in stream_groq(messages, model): yield t

# ─── Legacy sync query (CLI) ──────────────────────────────────────────────────
def query_rag(question: str, jurisdiction: str = None, source_type: str = None,
              top_k: int = None, provider: str = None, session_id: str = "default") -> dict:
    from api.memory import save_message, get_history
    from api.utils import parse_citations
    provider   = provider or LLM_PROVIDER
    top_k      = top_k or auto_context_depth(question)
    jurisdiction = jurisdiction or detect_jurisdiction(question)
    docs       = search_and_rerank(question, jurisdiction, top_k)
    confidence = tier_sources(docs)
    history    = get_history(session_id, limit=5)
    prompt     = build_prompt(question, docs, history, confidence)
    messages   = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        import httpx as _h
        if provider == "groq":
            r = _h.Client(timeout=60).post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": GROQ_MODEL, "messages": messages}
            ).json()["choices"][0]["message"]["content"]
        elif provider == "openrouter":
            r = _h.Client(timeout=60).post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={"model": OPENROUTER_MODEL, "messages": messages}
            ).json()["choices"][0]["message"]["content"]
        else:
            r = "Provider not supported in sync mode."
        r = parse_citations(r)
    except Exception as e:
        r = f"Error: {e}"
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", r, sources=docs, provider=provider)
    return {"answer": r, "sources": [{"title": d["doc_title"], "source": d["source"],
            "jurisdiction": d["jurisdiction"], "type": d["source_type"], "url": d["url"],
            "score": round(d.get("rerank_score", 0), 3)} for d in docs],
            "context_used": len(docs), "provider": provider, "session_id": session_id,
            "confidence": confidence, "jurisdiction": jurisdiction}

if __name__ == "__main__":
    r = query_rag("What is the GST rate on online gaming contest entry fees in India?")
    print("ANSWER:", r["answer"][:300])
    print("CONFIDENCE:", r["confidence"])
    print("JURISDICTION:", r["jurisdiction"])