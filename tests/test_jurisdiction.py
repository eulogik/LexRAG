"""Tests for jurisdiction auto-detection."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.utils import detect_jurisdiction


def test_detect_india():
    assert detect_jurisdiction("What is the GST rate on restaurants?") == "India"


def test_detect_uae():
    assert detect_jurisdiction("What is the VAT rate in Dubai?") == "UAE"


def test_detect_both():
    assert detect_jurisdiction("Compare GST and VAT rates") == "Both"


def test_detect_ambiguous():
    assert detect_jurisdiction("Hello world") == "Both"


def test_india_keyword_tds():
    assert detect_jurisdiction("TDS under section 194J") == "India"


def test_uae_keyword_corporate_tax():
    assert detect_jurisdiction("UAE corporate tax 9% threshold") == "UAE"


def test_india_specific():
    assert detect_jurisdiction("Income Tax Act 1961 section 80C deduction") == "India"


def test_uae_specific():
    assert detect_jurisdiction("Cabinet Decision on VAT refund") == "UAE"
