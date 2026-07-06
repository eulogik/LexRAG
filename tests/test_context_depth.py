"""Tests for auto-context-depth calculation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.utils import auto_context_depth


def test_short_simple_query():
    assert auto_context_depth("What is VAT?") == 5


def test_medium_complexity():
    depth = auto_context_depth("What is the GST rate on restaurants under section 7 of CGST Act?")
    assert depth >= 5


def test_complex_legal_query():
    depth = auto_context_depth(
        "Under section 194 of the Income Tax Act, what is the TDS rate for "
        "payments to contractors and how does the recent amendment in the "
        "Finance Act 2023 affect the exemption limit for professional fees?"
    )
    assert depth >= 8


def test_very_long_query():
    long_query = " ".join(["section"] * 20 + ["liability"] * 20 + ["exemption"] * 20)
    assert auto_context_depth(long_query) >= 10
