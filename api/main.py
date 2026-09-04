import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LEXRAG_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("lexrag")

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.memory import (
    delete_session,
    get_history,
    get_history_full,
    get_session_name,
    list_sessions,
    save_message,
    update_session_name,
)
from api.rag_engine import (
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENROUTER_MODEL,
    SYSTEM_PROMPT,
    build_prompt,
    search_and_rerank,
    stream_provider,
)
from api.security import API_KEY, SecurityMiddleware
from api.utils import (
    auto_context_depth,
    detect_jurisdiction,
    parse_citations,
    tier_sources,
)
from api.version import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading

    def warm():
        try:
            from embeddings.embedder import get_embedder
            get_embedder()
            from api.rag_engine import get_reranker
            get_reranker()
            log.info("Model warm-up complete")
        except Exception as e:
            log.warning(f"Model warm-up failed: {e}")

    threading.Thread(target=warm, daemon=True).start()
    yield

app = FastAPI(title="LexRAG", version=__version__, lifespan=lifespan)

_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("LEXRAG_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SecurityMiddleware)

if API_KEY:
    log.info("API key auth enabled for /api/*")
else:
    log.warning("LEXRAG_API_KEY not set — API is open. Set it before exposing this port.")

UI_DIR        = os.path.join(ROOT_DIR, "ui")
MARKETING_DIR = os.path.join(ROOT_DIR, "marketing")
SETTINGS_FILE = os.path.join(ROOT_DIR, "settings.json")

app.mount("/ui",        StaticFiles(directory=UI_DIR), name="ui")
app.mount("/marketing", StaticFiles(directory=MARKETING_DIR), name="marketing")

# ─── Model Catalog ───────────────────────────────────────────────────────────
MODEL_CATALOG = {
    "groq": [
        {"id": "openai/gpt-oss-120b",    "name": "GPT-OSS 120B (Recommended)"},
        {"id": "openai/gpt-oss-20b",     "name": "GPT-OSS 20B (Fast)"},
        {"id": "qwen/qwen3.6-27b",       "name": "Qwen3.6 27B (Reasoning)"},
    ],
    "openrouter": [
        {"id": "google/gemma-4-31b-it:free", "name": "Gemma 4 31B (Free)"},
        {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron-3 Super 120B (Free)"},
        {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "Nemotron-3 Ultra 550B (Free)"},
        {"id": "z-ai/glm-5.2:free", "name": "GLM-5.2 (Free)"},
    ],
    "ollama": [
        {"id": "qwen3:14b",    "name": "Qwen3 14B (Local)"},
        {"id": "llama3.1:8b",  "name": "Llama 3.1 8B (Local)"},
    ]
}

# Default active models if settings don't exist
DEFAULT_ACTIVE_MODELS = {
    "groq":       ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
    "openrouter": ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free", "z-ai/glm-5.2:free"],
    "ollama":     ["qwen3:14b", "llama3.1:8b"]
}

DEFAULT_PROVIDER_MODELS = {
    "groq":       "openai/gpt-oss-120b",
    "openrouter": "google/gemma-4-31b-it:free",
    "ollama":     "qwen3:14b"
}

DEFAULT_SETTINGS = {
    "provider":              LLM_PROVIDER,
    "model":                 DEFAULT_PROVIDER_MODELS.get(LLM_PROVIDER, "openai/gpt-oss-120b"),
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
        # Drop saved IDs the catalog no longer knows (stale after model rotations)
        known = {p: {m["id"] for m in MODEL_CATALOG.get(p, [])} for p in MODEL_CATALOG}
        active = merged.get("active_models", {})
        for p, ids in list(active.items()):
            if p in known:
                active[p] = [i for i in ids if i in known[p]] or DEFAULT_ACTIVE_MODELS[p]
        custom = merged.get("custom_models", {})
        for p, models in list(custom.items()):
            if p in known:
                custom[p] = [m for m in models if m.get("id") not in known[p]]
        if merged.get("provider") not in MODEL_CATALOG:
            merged["provider"] = DEFAULT_SETTINGS["provider"]
        prov = merged["provider"]
        valid_ids = known.get(prov, set()) | {m.get("id") for m in custom.get(prov, [])}
        if merged.get("model") not in valid_ids:
            merged["model"] = DEFAULT_PROVIDER_MODELS.get(prov, DEFAULT_SETTINGS["model"])
        return merged
    return DEFAULT_SETTINGS.copy()

# ─── Pages ───────────────────────────────────────────────────────────────────
_index_cache: dict = {}

@app.get("/", response_class=HTMLResponse)
def serve_app():
    path = os.path.join(UI_DIR, "index.html")
    mtime = os.path.getmtime(path)
    if _index_cache.get("mtime") != mtime:
        with open(path) as f:
            _index_cache["content"] = f.read()
        _index_cache["mtime"] = mtime
    return HTMLResponse(content=_index_cache["content"])

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
    for k, v in data.items():
        if k in DEFAULT_SETTINGS:
            current[k] = v
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
class IngestMetadata(BaseModel):
    source: str = "API Import"
    source_type: Literal["statute", "ruling", "case"] = "statute"
    jurisdiction: Literal["India", "UAE", "Both"] = "Both"
    doc_title: str = "Untitled"
    date: str = "imported"
    url: str = ""

class IngestTextRequest(BaseModel):
    text: str
    metadata: IngestMetadata = IngestMetadata()

@app.post("/api/ingest")
async def api_ingest_text(req: IngestTextRequest):
    from scripts.ingest import _local_ingest_text
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    if len(req.text) > 500_000:
        raise HTTPException(status_code=413, detail="Text too large (500k char limit)")
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _local_ingest_text(req.text, req.metadata.model_dump())
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/pdf")
async def api_ingest_pdf(file: UploadFile = File(...), metadata: str = Form("{}"),
                         thorough: bool = Form(True)):
    from scripts.ingest import _local_ingest_pdf
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF uploads accepted")
    try:
        meta = IngestMetadata.model_validate(json.loads(metadata or "{}"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        content = await file.read()
        if len(content) > 50_000_000:
            raise HTTPException(status_code=413, detail="PDF too large (50MB limit)")
        tmp.write(content)
        tmp.close()
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _local_ingest_pdf(tmp.name, meta.model_dump(), thorough)
        )
        return {"success": True, "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass

# ─── API: Chat (SSE Streaming) ────────────────────────────────────────────────
VALID_PROVIDERS = ("groq", "openrouter", "ollama")

class ChatRequest(BaseModel):
    question:              str = Field(max_length=8000)
    session_id:            str = Field(max_length=128)
    provider:              str | None = None
    model:                 str | None = Field(default=None, max_length=256)
    jurisdiction_override: str | None = None

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    settings = load_settings()
    provider = req.provider or settings.get("provider") or LLM_PROVIDER
    model    = req.model    or settings.get("model")
    if provider not in VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    async def generate():
        full_answer  = ""
        sources_out  = []
        jurisdiction = "Both"
        confidence   = "GROUNDED"

        def format_sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        # ── Step 0: History first (so the prompt never duplicates the
        # current question), then persist. Name the session only once. ──
        session_name = req.question[:60].strip()
        try:
            history = get_history(req.session_id, limit=5)
            if not history:
                update_session_name(req.session_id, session_name)
            save_message(req.session_id, "user", req.question)
        except Exception as e:
            history = []
            log.warning(f"Could not persist user message: {e}")

        try:
            # ── Step 1: Jurisdiction & Pings ────────────────────────────────
            # Send initial ping to confirm stream start
            yield ": ping\n\n"
            
            override = req.jurisdiction_override or settings.get("jurisdiction_override")
            if override and override != "Both":
                jurisdiction = override
            else:
                jurisdiction = detect_jurisdiction(req.question)

            # ── Step 2: Retrieval (45s budget; models pre-warmed at startup) ──
            top_k = auto_context_depth(req.question)
            try:
                t0 = asyncio.get_event_loop().time()
                docs = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: search_and_rerank(req.question, jurisdiction, top_k)
                    ),
                    timeout=45.0
                )
                log.info(f"Retrieval + Rerank took: {asyncio.get_event_loop().time() - t0:.3f}s")
            except asyncio.TimeoutError:
                docs = []
                log.warning("Retrieval timed out, using general knowledge.")

            confidence  = tier_sources(docs)
            # Deduplicate sources by (title, source) — same document split into many
            # chunks will produce identical titles; keep the highest-scoring one.
            seen_src: dict = {}
            for d in docs:
                key = (d.get("doc_title", d.get("source", "")), d.get("source", ""))
                score = round(d.get("rerank_score", d.get("score", 0)), 3)
                if key not in seen_src or score > seen_src[key]["score"]:
                    seen_src[key] = {
                        "title":        d.get("doc_title", d.get("source", "Unknown")),
                        "source":       d.get("source", ""),
                        "jurisdiction": d.get("jurisdiction", ""),
                        "type":         d.get("source_type", ""),
                        "url":          d.get("url", ""),
                        "score":        score
                    }
            sources_out = list(seen_src.values())

            # ── Step 3: Emit sources immediately ───────────────────────────
            yield format_sse("sources", {
                "sources":      sources_out,
                "confidence":   confidence,
                "jurisdiction": jurisdiction
            })

            # ── Step 4: Build prompt (history prefetched in Step 0) ─────────
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
                        log.info(f"Time to first token: {asyncio.get_event_loop().time() - t_gen_start:.3f}s")
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
                log.warning(f"Could not save assistant message: {e}")

            log.info("chat done session=%s provider=%s model=%s confidence=%s "
                     "jurisdiction=%s sources=%d chars=%d",
                     req.session_id[:8], provider, model, confidence,
                     jurisdiction, len(sources_out), len(full_answer))
            yield format_sse("done", {
                "session_name": session_name,
                "confidence":   confidence,
                "jurisdiction": jurisdiction
            })
            log.info(f"chat done session={req.session_id[:8]} provider={provider} "
                     f"model={model} confidence={confidence} jurisdiction={jurisdiction} "
                     f"sources={len(sources_out)} chars={len(full_answer)}")

        except Exception as e:
            error_msg = str(e)
            log.error(f"Chat error: {error_msg}")
            
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
    return {"status": "ok", "version": __version__}