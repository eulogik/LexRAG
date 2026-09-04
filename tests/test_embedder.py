"""Tests for embedder module (skipped unless fastembed is installed)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

fastembed = pytest.importorskip("fastembed", reason="needs fastembed installed")

from embeddings.embedder import COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL


def test_collection_name():
    assert COLLECTION_NAME == "lexrag_docs_v3"


def test_model_names():
    assert "bge-small" in DENSE_MODEL
    assert "Splade" in SPARSE_MODEL


def test_get_embedder_caching(monkeypatch):
    """Singleton returns the same instance without loading real models."""
    import embeddings.embedder as E
    monkeypatch.setattr(E, "_embedder", None)
    created = []

    class FakeEmbedder:
        def __init__(self):
            created.append(1)

    monkeypatch.setattr(E, "LexEmbedder", FakeEmbedder)
    assert E.get_embedder() is E.get_embedder()
    assert len(created) == 1
