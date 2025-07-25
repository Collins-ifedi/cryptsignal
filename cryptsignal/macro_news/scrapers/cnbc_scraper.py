import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Add project root for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("CNBCScraper")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

CNBC_URLS = [
    "https://www.cnbc.com/world/?region=world",
    "https://www.cnbc.com/economy/",
    "https://www.cnbc.com/markets/",
    "https://www.cnbc.com/cryptocurrency/"
]

OUTPUT_PATH = "data/macro/cnbc_news.json"

def fetch_url_with_retry(url, retries=3, delay=5):
    """
    Retry logic for network calls.
    """
    for attempt in range(retries):
        try:
            logger.info(f"Fetching URL (attempt {attempt+1}): {url}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed ({e}). Retrying in {delay}s...")
            time.sleep(delay)
    logger.error(f"Failed to fetch URL after {retries} attempts: {url}")
    return None

def fetch_cnbc_news(max_results=30, coin_name=None):
    news_data = []
    seen_links = set()
    logger.info("Starting CNBC scraping...")

    try:
        for url in CNBC_URLS:
            if len(news_data) >= max_results:
                break

            response = fetch_url_with_retry(url)
            if not response:
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.select("a.Card-title, a.LatestNews-headline")

            for article in articles:
                if len(news_data) >= max_results:
                    break

                title = article.get_text(strip=True)
                link = article.get("href")

                if not title or not link:
                    continue

                if not link.startswith("http"):
                    link = "https://www.cnbc.com" + link

                if link in seen_links:
                    continue

                if coin_name and coin_name.lower() not in title.lower():
                    continue

                seen_links.add(link)
                news_data.append({
                    "title": title,
                    "url": link,
                    "source": "CNBC",
                    "timestamp": datetime.utcnow().isoformat()
                })

        logger.info(f"Fetched {len(news_data)} CNBC articles.")
    except Exception as e:
        logger.exception(f"Failed to fetch CNBC news: {e}")

    return news_data

def save_cnbc_news(data):
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"CNBC news saved to {OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Error saving CNBC news: {e}")

def run_cnbc_scraper():
    news = fetch_cnbc_news()
    if news:
        save_cnbc_news(news)

if __name__ == "__main__":
    run_cnbc_scraper()