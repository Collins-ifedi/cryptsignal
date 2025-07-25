import os
import json
import sys
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from utils.logger import get_logger
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = get_logger("FedScraper")

FED_NEWS_URL = "https://www.federalreserve.gov/newsevents.htm"
OUTPUT_PATH = "data/macro/federal_reserve_news.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

FILTER_KEYWORDS = [
    "fomc", "federal reserve", "fed", "inflation", "interest rate", 
    "monetary policy", "economic outlook", "press conference", 
    "tightening", "easing", "hike", "cut", "statement", "minutes", 
    "economic projections", "central bank"
]


def matches_keywords(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in FILTER_KEYWORDS)


def scrape_federal_reserve_news(max_results=20):
    """Scrape latest macroeconomic news and press releases from Federal Reserve."""
    logger.info("Scraping Federal Reserve website...")
    results = []

    try:
        response = requests.get(FED_NEWS_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        news_items = soup.select("div.media-item")[:max_results]

        for item in news_items:
            title_tag = item.find("a")
            title = title_tag.text.strip() if title_tag else "No title"
            url = "https://www.federalreserve.gov" + title_tag["href"] if title_tag else ""
            date_tag = item.find("div", class_="media-date")
            date_str = date_tag.text.strip() if date_tag else datetime.utcnow().strftime("%Y-%m-%d")

            if matches_keywords(title):
                results.append({
                    "title": title,
                    "url": url,
                    "source": "Federal Reserve",
                    "published_date": date_str,
                    "scraped_at": datetime.utcnow().isoformat()
                })

    except Exception as e:
        logger.exception(f"Failed to scrape Federal Reserve news: {e}")

    return results


def save_news(data):
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} filtered news items to {OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Failed to save news data: {e}")


def run_scraper():
    data = scrape_federal_reserve_news()
    save_news(data)


if __name__ == "__main__":
    run_scraper()