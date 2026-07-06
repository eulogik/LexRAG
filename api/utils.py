import re
import os
import urllib.parse


# ─── Jurisdiction Detection ───────────────────────────────────────────────────
INDIA_KEYWORDS = [
    "gst", "cgst", "sgst", "igst", "tds", "tcs", "itc", "sebi", "rbi", "india", "indian",
    "rupee", "inr", "crore", "lakh", "section 194", "companies act", "ipc", "ibc",
    "income tax", "delhi", "mumbai", "bombay", "chennai", "bangalore", "hyderabad",
    "goods and services tax", "finance act", "cbdt", "gstr", "pan", "tan",
    "fema", "rera", "msme", "nri", "oci", "itat", "nclt", "nclat"
]
UAE_KEYWORDS = [
    "vat", "uae vat", "fta", "uae", "dubai", "abu dhabi", "sharjah", "ajman",
    "dirham", "aed", "free zone", "difc", "adgm", "excise tax",
    "federal tax authority", "corporate tax", "uae corporate", "ministry of finance",
    "cabinet decision", "federal decree", "mainland uae", "freezone"
]

def detect_jurisdiction(query: str) -> str:
    q = query.lower()
    india = sum(1 for kw in INDIA_KEYWORDS if kw in q)
    uae   = sum(1 for kw in UAE_KEYWORDS   if kw in q)
    if india > 0 and uae == 0: return "India"
    if uae   > 0 and india == 0: return "UAE"
    return "Both"

# ─── Auto Context Depth ───────────────────────────────────────────────────────
COMPLEX_TERMS = [
    "section", "act", "regulation", "notification", "circular", "judgment",
    "case", "versus", "liability", "exemption", "penalty", "compliance",
    "provision", "clause", "amendment", "appeal", "tribunal", "writ",
    "holding", "ratio", "precedent", "article", "schedule"
]

def auto_context_depth(query: str) -> int:
    wc   = len(query.split())
    hits = sum(1 for t in COMPLEX_TERMS if t in query.lower())
    if wc < 12 and hits < 2: return 5
    if wc < 30 and hits < 4: return 8
    if wc < 60 or hits < 6:  return 12
    return 15

# ─── Source Confidence Tiering ────────────────────────────────────────────────
def tier_sources(docs: list) -> str:
    if not docs: return "SYNTHESIZED"
    max_score = max(d.get("rerank_score", d.get("score", 0)) for d in docs)
    if max_score > 0.4:  return "GROUNDED"
    if max_score > 0.12: return "PARTIAL"
    return "SYNTHESIZED"

# ─── Think-Tag State Machine ──────────────────────────────────────────────────
def strip_think_tags(token: str, in_think: bool) -> tuple[str, bool]:
    output = ""
    remaining = token
    while remaining:
        if in_think:
            end_idx = remaining.find("</think>")
            if end_idx == -1:
                remaining = ""
            else:
                in_think  = False
                remaining = remaining[end_idx + len("</think>"):]
        else:
            start_idx = remaining.find("<think>")
            if start_idx == -1:
                output   += remaining
                remaining = ""
            else:
                output   += remaining[:start_idx]
                in_think  = True
                remaining = remaining[start_idx + len("<think>"):]
    return output, in_think

# ─── Citation patterns for UAE and India ─────────────────────────────────────
UAE_LAW_PATTERN = r"((?:Federal\s+)?(?:Decree-)?Law\s+No\.\s*(?:\(?\d+\)?)\s+of\s+\d{4})"
INDIA_LAW_PATTERN = r"((?:Income[\s-]?Tax\s+Act|Companies\s+Act|GST\s+Act|Income[\s-]?Tax\s+Rules?|Indian\s+Penal\s+Code|Insolvency\s+and\s+Bankruptcy\s+Code),?\s+\d{4})"
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
