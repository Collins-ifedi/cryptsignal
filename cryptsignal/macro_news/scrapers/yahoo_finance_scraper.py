import os
import json
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("YahooFinanceScraper")

YAHOO_FINANCE_URL = "https://finance.yahoo.com/topic/economic-news"
OUTPUT_PATH = "data/macro/yahoo_finance_news.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}


def parse_relative_time(text):
    now = datetime.utcnow()
    try:
        if "minute" in text:
            return now - timedelta(minutes=int(text.split()[0]))
        elif "hour" in text:
            return now - timedelta(hours=int(text.split()[0]))
        elif "day" in text:
            return now - timedelta(days=int(text.split()[0]))
        elif "Yesterday" in text:
            return now - timedelta(days=1)
        else:
            return datetime.strptime(text, "%B %d, %Y")
    except Exception:
        return now


def scrape_yahoo_finance_news(max_results=30, days_back=3):
    """Scrape Yahoo Finance economic news within last X days."""
    logger.info("Scraping Yahoo Finance news...")
    results = []
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)

    try:
        response = requests.get(YAHOO_FINANCE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        articles = soup.select("li.js-stream-content")[:max_results]

        for article in articles:
            title_tag = article.find("h3")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            url_tag = article.find("a", href=True)
            url = "https://finance.yahoo.com" + url_tag["href"] if url_tag else ""

            time_tag = article.find("span", class_="C(#959595)")
            raw_time = time_tag.get_text(strip=True) if time_tag else ""
            published = parse_relative_time(raw_time)

            if published < cutoff_date:
                continue

            results.append({
                "title": title,
                "url": url,
                "source": "Yahoo Finance",
                "published_date": published.strftime("%Y-%m-%d %H:%M:%S"),
                "scraped_at": datetime.utcnow().isoformat()
            })

    except Exception as e:
        logger.exception(f"Failed to scrape Yahoo Finance news: {e}")

    return results


def fetch_yahoo_finance_reports(max_results=30):
    """Fetch and store Yahoo Finance news."""
    logger.info("Fetching Yahoo Finance reports...")
    news = scrape_yahoo_finance_news(max_results=max_results)

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(news, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(news)} articles to {OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Error saving Yahoo Finance news: {e}")

    return news


if __name__ == "__main__":
    fetch_yahoo_finance_reports()