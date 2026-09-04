import os
from datetime import datetime, timezone

PROVIDER_PRESETS = {
    "groq": {
        "label": "Groq", "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY", "models_path": "/models",
        "default_models": [
            {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (Recommended)"},
            {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B (Fast)"},
            {"id": "qwen/qwen3.6-27b", "name": "Qwen3.6 27B (Reasoning)"},
        ],
    },
    "openrouter": {
        "label": "OpenRouter", "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY", "models_path": "/models",
        "default_models": [
            {"id": "google/gemma-4-31b-it:free", "name": "Gemma 4 31B (Free)"},
            {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron-3 Super 120B (Free)"},
            {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "Nemotron-3 Ultra 550B (Free)"},
            {"id": "z-ai/glm-5.2:free", "name": "GLM-5.2 (Free)"},
        ],
    },
    "ollama": {
        "label": "Ollama", "kind": "ollama",
        "base_url": "http://localhost:11434", "key_env": None, "models_path": "/api/tags",
        "default_models": [
            {"id": "qwen3:14b", "name": "Qwen3 14B (Local)"},
            {"id": "llama3.1:8b", "name": "Llama 3.1 8B (Local)"},
        ],
    },
    "openai": {
        "label": "OpenAI", "kind": "openai",
        "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY",
        "models_path": "/models", "default_models": [],
    },
    "deepseek": {
        "label": "DeepSeek", "kind": "openai",
        "base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY",
        "models_path": "/models", "default_models": [
            {"id": "deepseek-chat", "name": "DeepSeek V3"},
            {"id": "deepseek-reasoner", "name": "DeepSeek R1"},
        ],
    },
    "mistral": {
        "label": "Mistral", "kind": "openai",
        "base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY",
        "models_path": "/models", "default_models": [
            {"id": "mistral-large-latest", "name": "Mistral Large"},
            {"id": "mistral-small-latest", "name": "Mistral Small"},
        ],
    },
    "together": {
        "label": "Together", "kind": "openai",
        "base_url": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY",
        "models_path": "/models", "default_models": [],
    },
    "fireworks": {
        "label": "Fireworks", "kind": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY",
        "models_path": "/models", "default_models": [],
    },
    "xai": {
        "label": "xAI", "kind": "openai",
        "base_url": "https://api.x.ai/v1", "key_env": "XAI_API_KEY",
        "models_path": "/models", "default_models": [
            {"id": "grok-4", "name": "Grok 4"},
        ],
    },
}

DEFAULT_PROVIDER = "groq"


def all_providers(settings: dict) -> dict:
    merged = {pid: dict(p) for pid, p in PROVIDER_PRESETS.items()}
    for custom in (settings.get("custom_providers") or []):
        pid = (custom.get("id") or "").strip()
        if not pid or pid in merged:
            continue
        merged[pid] = {
            "label": custom.get("name") or pid, "kind": "openai",
            "base_url": (custom.get("base_url") or "").rstrip("/"),
            "key_env": None, "models_path": "/models", "default_models": [],
            "custom": True,
        }
    return merged


def provider_config(provider_id: str, settings: dict) -> dict:
    providers = all_providers(settings or {})
    if provider_id not in providers:
        raise ValueError(f"Unknown provider '{provider_id}'")
    preset = providers[provider_id]
    base_url = preset["base_url"]
    if provider_id == "ollama":
        base_url = os.environ.get("OLLAMA_URL", base_url).rstrip("/")
        if base_url.endswith("/api/chat"):
            base_url = base_url[: -len("/api/chat")]
    key = (settings.get("provider_keys") or {}).get(provider_id, "")
    source = "settings" if key else "env"
    if not key and preset.get("key_env"):
        key = os.environ.get(preset["key_env"], "")
    if not key:
        source = "none"
    return {"id": provider_id, "label": preset["label"], "kind": preset["kind"],
            "base_url": base_url.rstrip("/"), "api_key": key, "key_source": source,
            "custom": bool(preset.get("custom"))}


def parse_openai_models(payload: dict) -> list:
    out = []
    for m in (payload.get("data") or []):
        mid = m.get("id")
        if mid:
            out.append({"id": mid, "name": m.get("name") or mid})
    return out


def parse_ollama_tags(payload: dict) -> list:
    return [{"id": m["name"], "name": m["name"]}
            for m in (payload.get("models") or []) if m.get("name")]


def discover_models(provider_id: str, settings: dict) -> list:
    import httpx
    cfg = provider_config(provider_id, settings)
    headers = {}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    if provider_id == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/eulogik/LexRAG"
    url = cfg["base_url"] + ("/api/tags" if cfg["kind"] == "ollama" else "/models")
    r = httpx.get(url, headers=headers, timeout=20.0)
    r.raise_for_status()
    payload = r.json()
    if cfg["kind"] == "ollama":
        return parse_ollama_tags(payload)
    return parse_openai_models(payload)


def build_catalog(settings: dict) -> dict:
    providers = all_providers(settings or {})
    discovered = (settings or {}).get("discovered_models") or {}
    custom = (settings or {}).get("custom_models") or {}
    catalog = {}
    for pid, preset in providers.items():
        seen, models = set(), []
        sources = ((discovered.get(pid) or {}).get("models") or [])
        sources += (custom.get(pid) or [])
        sources += (preset.get("default_models") or [])
        for source in sources:
            mid = source.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                models.append({"id": mid, "name": source.get("name") or mid})
        catalog[pid] = models
    return catalog


def providers_status(settings: dict) -> dict:
    settings = settings or {}
    discovered = settings.get("discovered_models") or {}
    status = {}
    for pid, preset in all_providers(settings).items():
        cfg = provider_config(pid, settings)
        needs_key = bool(preset.get("key_env"))
        status[pid] = {
            "label": preset["label"], "kind": preset["kind"],
            "custom": bool(preset.get("custom")),
            "key_configured": (cfg["key_source"] != "none") or not needs_key,
            "key_source": cfg["key_source"] if needs_key else "unneeded",
            "model_count": len(build_catalog(settings).get(pid, [])),
            "discovered_at": (discovered.get(pid) or {}).get("fetched_at"),
        }
    return status


def public_settings(settings: dict) -> dict:
    scrubbed = {k: v for k, v in (settings or {}).items() if k != "provider_keys"}
    keys = (settings or {}).get("provider_keys") or {}
    scrubbed["provider_keys_set"] = {pid: True for pid in keys if keys[pid]}
    return scrubbed


def stamp_discovery() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
