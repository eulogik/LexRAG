# LexRAG HTTP Contract — v3.4.0

Base URL: `http://HOST:8000` (default `http://127.0.0.1:8000` via `lexrag serve`;
`HOST`/`PORT` env vars or `--host`/`--port` flags override).

## Auth

- If `LEXRAG_API_KEY` is **unset**: all endpoints open (dev mode; server logs a warning).
- If **set**: every `/api/*` request needs `X-API-Key: <key>` header
  (or `?api_key=<key>` query param). Missing/invalid → `401 {"detail": "Missing or invalid API key."}`
- Open (never authed): `/`, `/health`, `/ui/*`, `/marketing/*`.
- `OPTIONS` preflights always pass through.

## Rate limits (per client IP, sliding window)

| Group | Limit |
|---|---|
| `POST /api/chat` | 60 req / 60 s |
| `POST /api/ingest*` | 30 req / 60 s |
| other `/api/*` | 300 req / 60 s |

Exceeded → `429 {"detail": "Rate limit exceeded. Slow down."}`

## Stability notes (read this if you saw 503 ↔ 200)

- Nothing in the app returns 503. Flapping almost certainly came from **in front of
  the app**: a sleeping Hugging Face Space waking up, or a proxy hitting the server
  **during cold-start model load** (~30-60 s: embedder + reranker into RAM).
- v3.4.0 mitigations: startup model warm-up (first chat no longer pays cold-start),
  retrieval budget 15 s → 45 s, thread-safe model singletons, `HEALTHCHECK` in the
  Dockerfile, `restart: unless-stopped` in compose.
- Run it stably: `docker compose up -d` (health-gated), or
  `lexrag serve --host 0.0.0.0 --port 8000` under a process supervisor.
  Gate your hook on `GET /health` → 200 before sending chat traffic.

## `GET /health`

```json
{ "status": "ok", "version": "3.4.0" }
```

## `POST /api/chat` — streaming chat (SSE)

Request (`Content-Type: application/json`):

```json
{
  "question": "When is GSTR-3B due?",
  "session_id": "sess_abc123",
  "provider": "openrouter",
  "model": "google/gemma-4-31b-it:free",
  "jurisdiction_override": null
}
```

| Field | Required | Notes |
|---|---|---|
| `question` | yes | non-empty string |
| `session_id` | yes | client-generated id; new id = new conversation |
| `provider` | no | `groq` \| `openrouter` \| `ollama`; falls back to server default |
| `model` | no | model id; falls back to server default for the provider |
| `jurisdiction_override` | no | `India` \| `UAE` \| `Both` \| null (auto-detect) |

Validation: unknown `provider` → `400`; empty `question` → `400`;
`question` ≤ 8000 chars, `session_id` ≤ 128 chars, `model` ≤ 256 chars
(oversize → `422`). Aborting generation is client-side: cancel the
`fetch` (server stops streaming to the closed connection).

Response: `Content-Type: text/event-stream`. Event sequence is always
`sources` → zero or more `token` → `done` (or `error` → `done`).
`: ping` comment lines are keep-alives — ignore them.

```
: ping

event: sources
data: {"sources": [{"title": "...", "source": "...", "jurisdiction": "India",
      "type": "statute", "url": "...", "score": 0.93}],
      "confidence": "GROUNDED", "jurisdiction": "India"}

event: token
data: {"content": "GSTR-3B is due on"}

event: done
data: {"session_name": "When is GSTR-3B due?", "confidence": "GROUNDED",
      "jurisdiction": "India"}
```

- `confidence`: `GROUNDED` (cited, trust it) · `PARTIAL` (weak matches) ·
  `SYNTHESIZED` (no grounding — model knowledge only, treat as draft).
- Answer bodies end with a `JURISDICTION: India|UAE|Both|General` tag line
  (strip before display). `[INDEPENDENT ANALYSIS]` paragraphs = ungrounded.
- Errors: `event: error` + `data: {"content": "<message>"}` then
  `event: done` + `data: {"error": true}`. HTTP-level failures use plain
  status codes with a text body.

## `GET /api/models`

Full model catalog (+ server-side custom models), grouped by provider:

```json
{ "groq": [{"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (Recommended)"}],
  "openrouter": [{"id": "google/gemma-4-31b-it:free", "name": "Gemma 4 31B (Free)"}],
  "ollama": [{"id": "qwen3:14b", "name": "Qwen3 14B (Local)"}] }
```

## Settings

- `GET /api/settings` → `{"provider","model","jurisdiction_override","active_models","custom_models"}`
- `POST /api/settings` with any subset; `active_models`/`custom_models` deep-merge.

## Sessions

- `GET /api/sessions` → `[{session_id, name, last_active, created_at, message_count, preview}]` (newest first)
- `GET /api/sessions/{id}` → `{session_id, name, messages: [{role, content, sources, provider, timestamp}]}`
- `DELETE /api/sessions/{id}` → `{"success": true}`

## Ingestion

- `POST /api/ingest` — `{"text": "...", "metadata": {...}}`
  (text ≤ 500k chars; metadata keys: `source`, `source_type` ∈ statute|ruling|case,
  `jurisdiction` ∈ India|UAE|Both, `doc_title`, `date`, `url`) → `{"success": true}`
- `POST /api/ingest/pdf` — `multipart/form-data` with `file` (PDF, ≤ 50 MB),
  `metadata` (JSON string, same schema), `thorough` (bool, default true)
  → `{"success": true, "filename": "..."}`

## Pages

- `/` — chat UI · `/marketing` — landing page · response headers on all routes:
  `X-Request-ID`, `Content-Security-Policy`, `X-Content-Type-Options: nosniff`.
