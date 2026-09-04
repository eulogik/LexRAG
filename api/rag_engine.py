import json
import logging
import os
import sys
import threading

from dotenv import load_dotenv

log = logging.getLogger("lexrag")

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.utils import (
    auto_context_depth,
    detect_jurisdiction,
    strip_think_tags,
    tier_sources,
)
from embeddings.embedder import search

# ─── Reranker ────────────────────────────────────────────────────────────────
_reranker = None
_reranker_lock = threading.Lock()

def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                log.info("Initializing Reranker (Lazy)...")
                _reranker = CrossEncoder('BAAI/bge-reranker-base')
    return _reranker

# ─── Provider Config ──────────────────────────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

OPENROUTER_MODEL   = "google/gemma-4-31b-it:free"

GROQ_MODEL   = "openai/gpt-oss-120b"


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
        log.warning(f"Rerank error: {e}")
        return initial[:top_k]

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_prompt(query: str, context_docs: list, history: list = None, confidence: str = "GROUNDED") -> str:
    if context_docs:
        ctx = "\n\n---\n\n".join([
            f"[Source: {d.get('source', '')} | Jurisdiction: {d.get('jurisdiction', '')} | Date: {d.get('date', '')}]\n"
            f"Title: {d.get('doc_title', d.get('source', 'Unknown'))}\n\n{d.get('text', '')}"
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


async def stream_openai_compatible(messages: list, model: str, base_url: str,
                                   api_key: str = "", extra_headers: dict = None,
                                   label: str = "provider", timeout: float = 120.0):
    headers = {"Content-Type": "application/json", **(extra_headers or {})}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    in_think = False
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages, "stream": True}
        ) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                try: err_json = json.loads(err_body)
                except Exception: err_json = {"error": {"message": err_body.decode()}}
                msg = err_json.get("error", {}).get("message", "Unknown error")
                raise Exception(f"{label} API Error ({resp.status_code}): {msg}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "): continue
                data = line[6:]
                if data.strip() == "[DONE]": break
                try:
                    pdata = json.loads(data)
                    if "error" in pdata:
                        raise Exception(f"{label} Stream Error: {pdata['error'].get('message', 'Unknown')}")
                    token = pdata["choices"][0]["delta"].get("content", "")
                    if not token: continue
                    clean, in_think = strip_think_tags(token, in_think)
                    if clean: yield clean
                except Exception as e:
                    if "Stream Error" in str(e) or "API Error" in str(e): raise e
                    pass

async def stream_ollama(messages: list, model: str = None, base_url: str = None):
    model = model or OLLAMA_MODEL
    base_url = (base_url or OLLAMA_URL).rstrip("/")
    url = base_url if base_url.endswith("/api/chat") else base_url + "/api/chat"
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST", url,
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

async def stream_provider(messages: list, provider: str, model: str = None, settings: dict = None):
    from api.providers import provider_config
    cfg = provider_config(provider, settings or {})
    if cfg["kind"] == "ollama":
        async for t in stream_ollama(messages, model, cfg["base_url"]): yield t
        return
    extra = {"HTTP-Referer": "https://github.com/eulogik/LexRAG"} if provider == "openrouter" else None
    async for t in stream_openai_compatible(
            messages, model or _default_model(provider), cfg["base_url"],
            cfg["api_key"], extra_headers=extra, label=cfg["label"]): yield t


def _default_model(provider: str) -> str:
    return {"groq": GROQ_MODEL, "openrouter": OPENROUTER_MODEL,
            "ollama": OLLAMA_MODEL}.get(provider, GROQ_MODEL)


def complete_once(provider: str, model: str, messages: list, settings: dict = None,
                  timeout: float = 60.0) -> str:
    import httpx as _h
    from api.providers import provider_config
    cfg = provider_config(provider, settings or {})
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    if cfg["kind"] == "ollama":
        base = cfg["base_url"]
        url = base if base.endswith("/api/chat") else base + "/api/chat"
        r = _h.Client(timeout=timeout).post(
            url, json={"model": model, "messages": messages, "stream": False}).json()
        return r.get("message", {}).get("content", "")
    r = _h.Client(timeout=timeout).post(
        cfg["base_url"] + "/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages}).json()
    return r["choices"][0]["message"]["content"]

# ─── Legacy sync query (CLI) ──────────────────────────────────────────────────
def query_rag(question: str, jurisdiction: str = None, source_type: str = None,
              top_k: int = None, provider: str = None, model: str = None,
              session_id: str = "default") -> dict:
    from api.memory import get_history, save_message
    from api.utils import parse_citations
    provider   = provider or LLM_PROVIDER
    model      = model or _default_model(provider)
    top_k      = top_k or auto_context_depth(question)
    jurisdiction = jurisdiction or detect_jurisdiction(question)
    filters = {"source_type": [source_type]} if source_type else None
    docs = search_and_rerank(question, jurisdiction, top_k)
    if filters:
        docs = [d for d in docs if d.get("source_type") in filters["source_type"]]
    confidence = tier_sources(docs)
    history    = get_history(session_id, limit=5)
    prompt     = build_prompt(question, docs, history, confidence)
    messages   = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        r = complete_once(provider, model, messages)
        r = parse_citations(r)
    except Exception as e:
        r = f"Error: {e}"
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", r, sources=docs, provider=provider)
    return {"answer": r, "sources": [{"title": d.get("doc_title", d.get("source", "Unknown")), "source": d.get("source", ""),
            "jurisdiction": d.get("jurisdiction", ""), "type": d.get("source_type", ""), "url": d.get("url", ""),
            "score": round(d.get("rerank_score", d.get("score", 0)), 3)} for d in docs],
            "context_used": len(docs), "provider": provider, "model": model, "session_id": session_id,
            "confidence": confidence, "jurisdiction": jurisdiction}

if __name__ == "__main__":
    r = query_rag("What is the GST rate on online gaming contest entry fees in India?")
    print("ANSWER:", r["answer"][:300])
    print("CONFIDENCE:", r["confidence"])
    print("JURISDICTION:", r["jurisdiction"])