"""Tests for the provider registry (no heavy deps)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from api.providers import (
    PROVIDER_PRESETS,
    all_providers,
    build_catalog,
    parse_ollama_tags,
    parse_openai_models,
    provider_config,
    public_settings,
)


def test_presets_cover_major_providers():
    for pid in ("groq", "openrouter", "ollama", "openai", "deepseek",
                "mistral", "together", "fireworks", "xai"):
        assert pid in PROVIDER_PRESETS
        assert PROVIDER_PRESETS[pid]["base_url"].startswith("http")


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        provider_config("nope", {})


def test_env_key_used(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cfg = provider_config("groq", {})
    assert cfg["api_key"] == "gsk-test"
    assert cfg["key_source"] == "env"


def test_settings_key_beats_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-env")
    cfg = provider_config("groq", {"provider_keys": {"groq": "gsk-local"}})
    assert cfg["api_key"] == "gsk-local"
    assert cfg["key_source"] == "settings"


def test_ollama_needs_no_key():
    cfg = provider_config("ollama", {})
    assert cfg["kind"] == "ollama"
    assert cfg["key_source"] in ("none", "settings")


def test_custom_provider_merges():
    settings = {"custom_providers": [{"id": "mycloud", "name": "My Cloud",
                                      "base_url": "https://x.example/v1"}]}
    assert "mycloud" in all_providers(settings)
    cfg = provider_config("mycloud", {**settings, "provider_keys": {"mycloud": "k"}})
    assert cfg["base_url"] == "https://x.example/v1"
    assert cfg["api_key"] == "k"


def test_custom_provider_cannot_shadow_preset():
    settings = {"custom_providers": [{"id": "groq", "name": "Evil",
                                      "base_url": "https://evil.example"}]}
    assert all_providers(settings)["groq"]["base_url"].startswith("https://api.groq.com")


def test_parse_openai_models():
    payload = {"data": [{"id": "a", "name": "A"}, {"id": "b"}, {"nope": 1}]}
    assert parse_openai_models(payload) == [{"id": "a", "name": "A"}, {"id": "b", "name": "b"}]


def test_parse_ollama_tags():
    payload = {"models": [{"name": "qwen3:14b"}, {"name": ""}]}
    assert parse_ollama_tags(payload) == [{"id": "qwen3:14b", "name": "qwen3:14b"}]


def test_catalog_merge_dedupes():
    settings = {
        "discovered_models": {"groq": {"models": [{"id": "m1", "name": "Live M1"}]}},
        "custom_models": {"groq": [{"id": "m1", "name": "Stale"}, {"id": "mx", "name": "Custom"}]},
    }
    ids = [m["id"] for m in build_catalog(settings)["groq"]]
    assert ids.count("m1") == 1
    assert ids.index("m1") < ids.index("mx")
    assert "openai/gpt-oss-120b" in ids


def test_public_settings_scrubs_keys():
    settings = {"provider": "groq", "provider_keys": {"groq": "secret!"}}
    pub = public_settings(settings)
    assert "provider_keys" not in pub
    assert pub["provider_keys_set"] == {"groq": True}
    assert "secret!" not in str(pub)
