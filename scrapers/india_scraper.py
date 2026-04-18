import requests
from bs4 import BeautifulSoup
import os
import sys
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest import ingest_text

SEEN_HASHES_FILE = os.path.join(os.path.dirname(__file__), "seen_hashes_india.txt")
INDIAN_KANOON_API = "https://api.indiankanoon.org/search/"
INDIANKANOON_TOKEN = os.environ.get("INDIANKANOON_TOKEN")


def load_seen():
    if not os.path.exists(SEEN_HASHES_FILE):
        return set()
    with open(SEEN_HASHES_FILE) as f:
        return set(line.strip() for line in f)

def save_hash(h: str):
    with open(SEEN_HASHES_FILE, "a") as f:
        f.write(h + "\n")

def scrape_cbic_gst():
    """Scrapes CBIC GST circulars and notifications"""
    url = "https://cbic-gst.gov.in/gst-goods-services-rates.html"
    seen = load_seen()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        items = soup.find_all(["p", "li", "td"])
        count = 0
        for item in items:
            text = item.get_text(strip=True)
            if len(text) < 50:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "CBIC GST India",
                "source_type": "ruling",
                "jurisdiction": "India",
                "doc_title": text[:100],
                "date": "scraped",
                "url": url
            })
            save_hash(h)
            count += 1
        print(f"CBIC GST India: {count} new items ingested")
    except Exception as e:
        print(f"CBIC scrape error: {e}")

def fetch_indian_kanoon_cases(query: str, num_results: int = 10):
    if not INDIANKANOON_TOKEN:
        print("INDIANKANOON_TOKEN not set. Skip. Get free token at https://api.indiankanoon.org/api/")
        return
    seen = load_seen()
    try:
        resp = requests.post(
            INDIAN_KANOON_API,
            data={"formInput": query, "pagenum": 0},
            headers={"Authorization": f"Token {INDIANKANOON_TOKEN}"},
            timeout=15
        )
        data = resp.json()
        docs = data.get("docs", [])
        count = 0
        for doc in docs[:num_results]:
            text = doc.get("headline", "") + " " + doc.get("title", "")
            if len(text) < 30:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "Indian Kanoon",
                "source_type": "case",
                "jurisdiction": "India",
                "doc_title": doc.get("title", "")[:100],
                "date": doc.get("publishdate", ""),
                "url": f"https://indiankanoon.org/doc/{doc.get('tid', '')}/"
            })
            save_hash(h)
            count += 1
        print(f"Indian Kanoon [{query}]: {count} new cases ingested")
    except Exception as e:
        print(f"Indian Kanoon error: {e}")

if __name__ == "__main__":
    scrape_cbic_gst()
    fetch_indian_kanoon_cases("GST input tax credit")
    fetch_indian_kanoon_cases("income tax section 80C deduction")
    fetch_indian_kanoon_cases("VAT tribunal ruling")