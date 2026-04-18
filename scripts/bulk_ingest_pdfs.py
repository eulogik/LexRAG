import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest import ingest_pdf

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

JURISDICTION_HINTS = {
    "uae": "UAE", "dubai": "UAE", "adgm": "UAE", "difc": "UAE",
    "fta": "UAE", "cbuae": "UAE",
    "india": "India", "cbic": "India", "gst": "India", "incometax": "India",
    "sebi": "India", "rbi": "India"
}
SOURCE_TYPE_HINTS = {
    "act": "statute", "law": "statute", "decree": "statute", "regulation": "statute",
    "case": "case", "judgment": "case", "ruling": "ruling", "circular": "ruling",
    "notification": "ruling", "order": "ruling"
}

def guess_meta(filename: str) -> dict:
    name_lower = filename.lower()
    jurisdiction = "Both"
    for hint, j in JURISDICTION_HINTS.items():
        if hint in name_lower:
            jurisdiction = j
            break
    source_type = "statute"
    for hint, t in SOURCE_TYPE_HINTS.items():
        if hint in name_lower:
            source_type = t
            break
    return {
        "source": "Bulk PDF Import",
        "source_type": source_type,
        "jurisdiction": jurisdiction,
        "doc_title": filename.replace("_", " ").replace(".pdf", ""),
        "date": "imported",
        "url": ""
    }

if __name__ == "__main__":
    pdfs = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Drop your PDF files there and rerun.")
    for pdf in pdfs:
        meta = guess_meta(pdf)
        print(f"Processing: {pdf} → jurisdiction={meta['jurisdiction']}, type={meta['source_type']}")
        ingest_pdf(os.path.join(RAW_DIR, pdf), meta)
    print(f"Done. {len(pdfs)} PDFs ingested.")