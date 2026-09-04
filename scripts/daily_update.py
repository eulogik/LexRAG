import argparse
import schedule
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.uae_scraper import scrape_fta_updates, scrape_moj_uae
from scrapers.india_scraper import scrape_cbic_gst, fetch_indian_kanoon_cases

INDIA_QUERIES = [
    "GST input credit",
    "income tax penalty",
    "GST registration cancellation",
]


def run_all_scrapers():
    print("=== Daily LexRAG Update Started ===")
    scrape_fta_updates()
    scrape_moj_uae()
    scrape_cbic_gst()
    for q in INDIA_QUERIES:
        fetch_indian_kanoon_cases(q)
    print("=== Daily Update Complete ===")


def main(argv=None):
    p = argparse.ArgumentParser(description="LexRAG daily content refresh")
    p.add_argument("--once", action="store_true", help="Run once and exit (cron/systemd use)")
    args = p.parse_args(argv)

    run_all_scrapers()
    if args.once:
        return

    schedule.every().day.at("02:00").do(run_all_scrapers)
    print("Scheduler running. Will update daily at 2 AM. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
