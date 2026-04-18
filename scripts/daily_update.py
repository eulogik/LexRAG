import schedule
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.uae_scraper import scrape_fta_updates, scrape_moj_uae
from scrapers.india_scraper import scrape_cbic_gst, fetch_indian_kanoon_cases

def run_all_scrapers():
    print("=== Daily LexRAG Update Started ===")
    scrape_fta_updates()
    scrape_moj_uae()
    scrape_cbic_gst()
    fetch_indian_kanoon_cases("corporate tax UAE 2024")
    fetch_indian_kanoon_cases("GST input credit")
    fetch_indian_kanoon_cases("income tax penalty")
    fetch_indian_kanoon_cases("VAT exemption")
    print("=== Daily Update Complete ===")

run_all_scrapers()

schedule.every().day.at("02:00").do(run_all_scrapers)

print("Scheduler running. Will update daily at 2 AM. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)