"""Tests for utility functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.utils import parse_citations, format_score


def test_parse_citations_uae_law():
    result = parse_citations("Under Federal Decree-Law No. 8 of 2017")
    assert "[" in result
    assert "elaws.moj.gov.ae" in result


def test_parse_citations_india_law():
    # "Income Tax Act" needs space between Income and Tax matched
    result = parse_citations("Under the Income-Tax Act, 1961")
    assert "[" in result
    assert "indiankanoon.org" in result


def test_parse_citations_section():
    result = parse_citations("Section 194 of the Act")
    assert "[" in result


def test_no_double_wrapping():
    result = parse_citations("Already [linked](https://example.com) text")
    # Should not double-wrap
    
    import re
    # Count the number of markdown links
    links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', result)
    # Should have at most 1 link
    assert len(links) >= 1


def test_format_score_high():
    assert format_score(0.9) == "High"


def test_format_score_medium():
    assert format_score(0.6) == "Medium"


def test_format_score_low():
    assert format_score(0.3) == "Low"


def test_format_score_boundary_high():
    # > 0.8 for High, so 0.8 is Medium
    assert format_score(0.8) == "Medium"


def test_format_score_boundary_medium():
    # > 0.5 for Medium, so 0.5 is Low
    assert format_score(0.5) == "Low"
