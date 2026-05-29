import re
import os

import urllib.parse

# Citation patterns for UAE and India
UAE_LAW_PATTERN = r"((?:Federal\s+)?(?:Decree-)?Law\s+No\.\s*(?:\(?\d+\)?)\s+of\s+\d{4})"
INDIA_LAW_PATTERN = r"((?:Income-?Tax\s+Act|Companies\s+Act|GST\s+Act|Income-?Tax\s+Rules?|Indian\s+Penal\s+Code|Insolvency\s+and\s+Bankruptcy\s+Code),?\s+\d{4})"
INDIA_SECTION_PATTERN = r"(Section\s+\d+[A-Z]?(?:-\s*[A-Z]+)?)"

def parse_citations(text: str) -> str:
    """
    Scans text for legal citations and wraps them in professional tagging or markdown links.
    Avoids double wrapping already-linked text.
    """
    # UAE Law linking (Search on UAE Legislation Portal - ignores existing markdown links)
    uae_combined = r"(\[[^\]]+\]\([^\)]+\))|" + UAE_LAW_PATTERN
    def uae_repl(match):
        if match.group(1):
            return match.group(1)
        val = match.group(2)
        encoded = urllib.parse.quote(val)
        return f"[{val}](https://elaws.moj.gov.ae/UAE-Legislations-Search-en.aspx?query={encoded})"
    text = re.sub(uae_combined, uae_repl, text)
    
    # India Law linking (Search on Indian Kanoon)
    india_combined = r"(\[[^\]]+\]\([^\)]+\))|" + INDIA_LAW_PATTERN
    def india_repl(match):
        if match.group(1):
            return match.group(1)
        val = match.group(2)
        encoded = urllib.parse.quote(val)
        return f"[{val}](https://indiankanoon.org/search/?formInput={encoded})"
    text = re.sub(india_combined, india_repl, text)
    
    # Section linking (Contextual search on Indian Kanoon)
    section_combined = r"(\[[^\]]+\]\([^\)]+\))|" + INDIA_SECTION_PATTERN
    def section_repl(match):
        if match.group(1):
            return match.group(1)
        val = match.group(2)
        encoded = urllib.parse.quote(val)
        return f"[{val}](https://indiankanoon.org/search/?formInput={encoded})"
    text = re.sub(section_combined, section_repl, text)
    
    return text

def format_score(score: float) -> str:
    """Formats a search score into a user-friendly percentage or indicator."""
    if score > 0.8: return "High"
    if score > 0.5: return "Medium"
    return "Low"
