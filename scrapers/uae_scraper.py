import requests
from bs4 import BeautifulSoup
import os
import sys
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ingest import ingest_text

SEEN_HASHES_FILE = os.path.join(os.path.dirname(__file__), "seen_hashes_uae.txt")

def load_seen():
    if not os.path.exists(SEEN_HASHES_FILE):
        return set()
    with open(SEEN_HASHES_FILE) as f:
        return set(line.strip() for line in f)

def save_hash(h: str):
    with open(SEEN_HASHES_FILE, "a") as f:
        f.write(h + "\n")

def scrape_fta_updates():
    """Scrapes UAE FTA latest announcements page"""
    url = "https://www.tax.gov.ae/en/legislation.aspx"
    seen = load_seen()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        items = soup.find_all("a", href=True)
        count = 0
        for item in items:
            text = item.get_text(strip=True)
            if len(text) < 30:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "FTA UAE",
                "source_type": "ruling",
                "jurisdiction": "UAE",
                "doc_title": text[:100],
                "date": "scraped",
                "url": "https://www.tax.gov.ae" + item["href"]
            })
            save_hash(h)
            count += 1
        print(f"FTA UAE: {count} new items ingested")
    except Exception as e:
        print(f"FTA scrape error: {e}")

def scrape_moj_uae():
    """Scrapes UAE Ministry of Justice legislation page"""
    url = "https://uaelegislation.gov.ae/en/legislations"
    seen = load_seen()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "lxml")
        links = soup.find_all("a", href=True)
        count = 0
        for link in links:
            text = link.get_text(strip=True)
            if len(text) < 20:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen:
                continue
            ingest_text(text, {
                "source": "UAE Legislation Portal",
                "source_type": "statute",
                "jurisdiction": "UAE",
                "doc_title": text[:100],
                "date": "scraped",
                "url": link["href"] if link["href"].startswith("http") else "https://uaelegislation.gov.ae" + link["href"]
            })
            save_hash(h)
            count += 1
        print(f"MOJ UAE: {count} new items ingested")
    except Exception as e:
        print(f"MOJ UAE scrape error: {e}")

if __name__ == "__main__":
    scrape_fta_updates()
    scrape_moj_uae()