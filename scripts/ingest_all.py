import os
import sys
from glob import glob

# Add root directory to path to ensure proper imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.ingest import ingest_text, ingest_pdf

DATA_DIR = os.path.join(ROOT_DIR, "data")

def main():
    # Ingest Text Files
    txt_files = glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True)
    for f in txt_files:
        print(f"Ingesting text file: {f}")
        with open(f, "r") as r:
            content = r.read()
            # Extract metadata from filename or use defaults
            title = os.path.basename(f).replace(".txt", "").replace("_", " ").title()
            jurisdiction = "India" if "india" in f.lower() else ("UAE" if "uae" in f.lower() else "Both")
            ingest_text(content, {
                "source": os.path.basename(f),
                "source_type": "statute",
                "jurisdiction": jurisdiction,
                "doc_title": title,
                "date": "2024-04-20",
                "url": ""
            })

    # Ingest PDF Files (if any)
    pdf_files = glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    for f in pdf_files:
        print(f"Ingesting PDF file: {f}")
        title = os.path.basename(f).replace(".pdf", "").replace("_", " ").title()
        jurisdiction = "India" if "india" in f.lower() else ("UAE" if "uae" in f.lower() else "Both")
        ingest_pdf(f, {
            "source": os.path.basename(f),
            "source_type": "statute",
            "jurisdiction": jurisdiction,
            "doc_title": title,
            "date": "2024-04-20",
            "url": ""
        })

if __name__ == "__main__":
    main()
