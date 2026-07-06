"""Tests for source confidence tiering."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.utils import tier_sources


def test_empty_docs_returns_synthesized():
    assert tier_sources([]) == "SYNTHESIZED"


def test_high_score_returns_grounded():
    docs = [{"rerank_score": 0.5}, {"rerank_score": 0.3}]
    assert tier_sources(docs) == "GROUNDED"


def test_medium_score_returns_partial():
    docs = [{"rerank_score": 0.2}, {"rerank_score": 0.15}]
    assert tier_sources(docs) == "PARTIAL"


def test_low_score_returns_synthesized():
    docs = [{"rerank_score": 0.05}]
    assert tier_sources(docs) == "SYNTHESIZED"


def test_fallback_to_score():
    docs = [{"score": 0.5}]
    assert tier_sources(docs) == "GROUNDED"


def test_mixed_scores():
    # max is 0.4 which is NOT > 0.4, so PARTIAL
    docs = [{"rerank_score": 0.1}, {"score": 0.4}]
    assert tier_sources(docs) == "PARTIAL"
