import os
import sys
import json
import asyncio
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.rag_engine import (
    search_and_rerank, build_prompt, SYSTEM_PROMPT, stream_provider,
    detect_jurisdiction, auto_context_depth, tier_sources, LLM_PROVIDER,
    GROQ_MODEL, OPENROUTER_MODEL
)
from api.memory import (
    save_message, get_history, get_history_full,
    list_sessions, delete_session, update_session_name, get_session_name
)
from api.utils import parse_citations

app = FastAPI(title="LexRAG", version="3.1")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UI_DIR        = os.path.join(ROOT_DIR, "ui")
MARKETING_DIR = os.path.join(ROOT_DIR, "marketing")
SETTINGS_FILE = os.path.join(ROOT_DIR, "settings.json")

app.mount("/ui",        StaticFiles(directory=UI_DIR), name="ui")
app.mount("/marketing", StaticFiles(directory=MARKETING_DIR), name="marketing")

# ─── Model Catalog ───────────────────────────────────────────────────────────
MODEL_CATALOG = {
    "groq": [
        {"id": "llama-3.3-70b-versatile",  "name": "Llama 3.3 70B"},
        {"id": "llama-3.1-8b-instant",      "name": "Llama 3.1 8B"},
        {"id": "gemma2-9b-it",              "name": "Gemma 2 9B"},
    ],
    "openrouter": [
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B (Free)"},
        {"id": "deepseek/deepseek-r1:free",          "name": "DeepSeek R1 (Free)"},
        {"id": "google/gemma-2-9b-it:free",          "name": "Gemma 2 9B (Free)"},
        {"id": "qwen/qwen-2.5-coder-32b-instruct:free", "name": "Qwen 2.5 Coder 32B (Free)"},
    ],
    "ollama": [
        {"id": "qwen3:14b",      "name": "Qwen3 14B (Local)"},
        {"id": "llama3:latest",  "name": "Llama 3 (Local)"},
    ]
}

# Default active models if settings don't exist
DEFAULT_ACTIVE_MODELS = {
    "groq":       ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
    "openrouter": ["meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-r1:free", "google/gemma-2-9b-it:free"],
    "ollama":     ["qwen3:14b", "llama3:latest"]
}

DEFAULT_PROVIDER_MODELS = {
    "groq":       "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "ollama":     "qwen3:14b"
}

DEFAULT_SETTINGS = {
    "provider":              LLM_PROVIDER,
    "model":                 DEFAULT_PROVIDER_MODELS.get(LLM_PROVIDER, "llama-3.3-70b-versatile"),
    "jurisdiction_override": None,
    "active_models":         DEFAULT_ACTIVE_MODELS,
    "custom_models":         {}   # {"groq": [{"id": "...", "name": "..."}]}
}

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        saved = {}
        with open(SETTINGS_FILE) as f:
            try: saved = json.load(f)
            except Exception: pass
        merged = {**DEFAULT_SETTINGS, **saved}
        # Ensure active_models has all providers
        for p in DEFAULT_ACTIVE_MODELS:
            if p not in merged.get("active_models", {}):
                merged.setdefault("active_models", {})[p] = DEFAULT_ACTIVE_MODELS[p]
        return merged
    return DEFAULT_SETTINGS.copy()

def save_settings_to_disk(data: dict):
    current = load_settings()
    current.update(data)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current, f, indent=2)

# ─── Pages ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_app():
    with open(os.path.join(UI_DIR, "index.html")) as f:
        return HTMLResponse(content=f.read())

# ─── API: Models ─────────────────────────────────────────────────────────────
@app.get("/api/models")
def get_models():
    """Returns full catalog plus any custom models from settings."""
    settings = load_settings()
    catalog  = {p: list(models) for p, models in MODEL_CATALOG.items()}
    # Merge custom models
    for provider, custom in settings.get("custom_models", {}).items():
        if provider in catalog:
            existing_ids = {m["id"] for m in catalog[provider]}
            for cm in custom:
                if cm["id"] not in existing_ids:
                    catalog[provider].append(cm)
        else:
            catalog[provider] = custom
    return catalog

# ─── API: Settings ───────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings_endpoint():
    return load_settings()

@app.post("/api/settings")
async def update_settings_endpoint(request: Request):
    data    = await request.json()
    current = load_settings()
    # Deep merge active_models and custom_models
    for deep_key in ("active_models", "custom_models"):
        if deep_key in data and isinstance(data[deep_key], dict):
            current.setdefault(deep_key, {}).update(data[deep_key])
            del data[deep_key]
    current.update(data)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current, f, indent=2)
    return current

# ─── API: Sessions ────────────────────────────────────────────────────────────
@app.get("/api/sessions")
def api_list_sessions():
    return list_sessions()

@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    return {
        "session_id": session_id,
        "name":       get_session_name(session_id),
        "messages":   get_history_full(session_id)
    }

@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str):
    delete_session(session_id)
    return {"success": True}

# ─── API: Ingestion Endpoints (Server-Centric De-confliction) ──────────────────
class IngestTextRequest(BaseModel):
    text: str
    metadata: dict

class IngestPDFRequest(BaseModel):
    filepath: str
    metadata: dict
    thorough: Optional[bool] = True

@app.post("/api/ingest")
async def api_ingest_text(req: IngestTextRequest):
    from scripts.ingest import _local_ingest_text
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _local_ingest_text(req.text, req.metadata)
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/pdf")
async def api_ingest_pdf(req: IngestPDFRequest):
    from scripts.ingest import _local_ingest_pdf
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail="PDF file not found at specified path")
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _local_ingest_pdf(req.filepath, req.metadata, req.thorough)
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Chat (SSE Streaming) ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question:              str
    session_id:            str
    provider:              Optional[str] = None
    model:                 Optional[str] = None
    jurisdiction_override: Optional[str] = None

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    settings = load_settings()
    provider = req.provider or settings.get("provider") or LLM_PROVIDER
    model    = req.model    or settings.get("model")

    async def generate():
        start_all    = asyncio.get_event_loop().time()
        full_answer  = ""
        sources_out  = []
        jurisdiction = "Both"
        confidence   = "GROUNDED"

        def format_sse(event: str, data: any) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        # ── Step 0: Save user message IMMEDIATELY ───────────────────────────
        try:
            session_name = req.question[:60].strip()
            save_message(req.session_id, "user", req.question)
            update_session_name(req.session_id, session_name)
        except Exception as e:
            print(f"Warning: Could not save user message: {e}")

        try:
            # ── Step 1: Jurisdiction & Pings ────────────────────────────────
            # Send initial ping to confirm stream start
            yield ": ping\n\n"
            
            override = req.jurisdiction_override or settings.get("jurisdiction_override")
            if override and override != "Both":
                jurisdiction = override
            else:
                jurisdiction = detect_jurisdiction(req.question)

            # ── Step 2: Retrieval (with 15s timeout) ───────────────────────
            top_k = auto_context_depth(req.question)
            try:
                t0 = asyncio.get_event_loop().time()
                docs = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: search_and_rerank(req.question, jurisdiction, top_k)
                    ),
                    timeout=15.0
                )
                print(f"Retrieval + Rerank took: {asyncio.get_event_loop().time() - t0:.3f}s")
            except asyncio.TimeoutError:
                docs = []
                print("Warning: Retrieval timed out, using general knowledge.")

            confidence  = tier_sources(docs)
            sources_out = [
                {
                    "title":        d.get("doc_title", d.get("source", "Unknown")),
                    "source":       d.get("source", ""),
                    "jurisdiction": d.get("jurisdiction", ""),
                    "type":         d.get("source_type", ""),
                    "url":          d.get("url", ""),
                    "score":        round(d.get("rerank_score", d.get("score", 0)), 3)
                }
                for d in docs
            ]

            # ── Step 3: Emit sources immediately ───────────────────────────
            yield format_sse("sources", {
                "sources":      sources_out,
                "confidence":   confidence,
                "jurisdiction": jurisdiction
            })

            # ── Step 4: Build prompt ────────────────────────────────────────
            history  = get_history(req.session_id, limit=5)
            prompt   = build_prompt(req.question, docs, history, confidence)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]

            # ── Step 5: Stream with total 90s timeout ──────────────────────
            async def stream_with_timeout():
                t_gen_start = asyncio.get_event_loop().time()
                first = True
                async for token in stream_provider(messages, provider, model):
                    if first:
                        print(f"Time to first token: {asyncio.get_event_loop().time() - t_gen_start:.3f}s")
                        first = False
                    yield token

            # Periodic ping wrapper to prevent proxy timeouts
            async def stream_with_pings():
                queue = asyncio.Queue()
                
                async def producer():
                    try:
                        async for token in stream_with_timeout():
                            await queue.put(("token", token))
                    except Exception as e:
                        await queue.put(("error", e))
                    finally:
                        await queue.put(("done", None))

                producer_task = asyncio.create_task(producer())
                got_first_token = False
                
                try:
                    async with asyncio.timeout(90):
                        while True:
                            try:
                                msg_type, val = await asyncio.wait_for(queue.get(), timeout=5.0)
                                if msg_type == "token":
                                    got_first_token = True
                                    yield val
                                elif msg_type == "error":
                                    raise val
                                elif msg_type == "done":
                                    break
                            except asyncio.TimeoutError:
                                # Yield SSE comment ping directly to bypass format_sse
                                yield ": ping\n\n"
                except asyncio.TimeoutError:
                    if not got_first_token:
                        yield "\n\n*Response timed out. Try a smaller model or check your network.*"
                finally:
                    producer_task.cancel()

            async for token in stream_with_pings():
                if token.startswith(": ping"):
                    yield token
                else:
                    full_answer += token
                    yield format_sse("token", {"content": token})

            # ── Step 6: Citation links + save answer ───────────────────────
            full_answer = parse_citations(full_answer)
            try:
                save_message(req.session_id, "assistant", full_answer,
                             sources=sources_out, provider=provider)
            except Exception as e:
                print(f"Warning: Could not save assistant message: {e}")

            yield format_sse("done", {
                "session_name": session_name,
                "confidence":   confidence,
                "jurisdiction": jurisdiction
            })

        except Exception as e:
            error_msg = str(e)
            print(f"Chat error: {error_msg}")
            
            # Flush error to UI
            yield format_sse("error", {"content": error_msg})
            yield format_sse("done", {"error": True})

            if full_answer:
                try:
                    save_message(req.session_id, "assistant", full_answer,
                                 sources=sources_out, provider=provider)
                except Exception:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.1"}