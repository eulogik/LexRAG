import re
import os

# Citation patterns for UAE and India
UAE_LAW_PATTERN = r"(Federal Decree-Law No\.\s+\d+\s+of\s+\d+)"
INDIA_LAW_PATTERN = r"((?:Income Tax Act|Companies Act|GST Act|Income Tax Rule),?\s+\d{4})"
INDIA_SECTION_PATTERN = r"(Section\s+\d+[A-Z]?)"

def parse_citations(text: str) -> str:
    """
    Scans text for legal citations and wraps them in professional tagging or markdown links.
    """
    # UAE Law linking (Search on UAE Legislation Portal as placeholder)
    text = re.sub(
        UAE_LAW_PATTERN, 
        r"[\1](https://elaws.moj.gov.ae/UAE-Legislations-Search-en.aspx)", 
        text
    )
    
    # India Law linking (Search on Indian Kanoon)
    text = re.sub(
        INDIA_LAW_PATTERN, 
        r"[\1](https://indiankanoon.org/search/?formInput=\1)", 
        text
    )
    
    # Section linking (Contextual search on Indian Kanoon)
    text = re.sub(
        INDIA_SECTION_PATTERN, 
        r"[\1](https://indiankanoon.org/search/?formInput=\1)", 
        text
    )
    
    return text

def format_score(score: float) -> str:
    """Formats a search score into a user-friendly percentage or indicator."""
    if score > 0.8: return "High"
    if score > 0.5: return "Medium"
    return "Low"
