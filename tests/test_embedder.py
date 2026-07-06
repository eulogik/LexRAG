"""Tests for embedder module (mocked, no model loading)."""

import sys
import os
import unittest.mock as mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings.embedder import LexEmbedder


def test_collection_name():
    from embeddings.embedder import COLLECTION_NAME
    assert COLLECTION_NAME == "lexrag_docs_v3"


def test_model_names():
    from embeddings.embedder import DENSE_MODEL, SPARSE_MODEL
    assert "bge-small" in DENSE_MODEL
    assert "Splade" in SPARSE_MODEL


def test_get_embedder_caching():
    """Test that get_embedder returns the same instance."""
    from embeddings.embedder import get_embedder
    e1 = get_embedder()
    e2 = get_embedder()
    assert e1 is e2
