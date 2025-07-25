import os
import sys
import json
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("InvestingScraper")

INVESTING_URL = "https://www.investing.com/news/economy"
OUTPUT_PATH = "data/macro/investing_news.json"

FILTER_KEYWORDS = [
    "inflation", "recession", "economic", "growth", "rate hike", "interest rate",
    "Federal Reserve", "Fed", "ECB", "central bank", "monetary policy",
    "geopolitical", "Ukraine", "Middle East", "China", "Russia", "conflict",
    "court", "SEC", "lawsuit", "ruling", "judge", "sanctions", "BRICS", "oil"
]


def matches_keywords(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in FILTER_KEYWORDS)


def parse_date_string(raw_date: str) -> datetime:
    now = datetime.utcnow()
    raw_date = raw_date.strip().lower()
    try:
        if "minutes ago" in raw_date:
            return now - timedelta(minutes=int(raw_date.split()[0]))
        elif "hours ago" in raw_date:
            return now - timedelta(hours=int(raw_date.split()[0]))
        elif "yesterday" in raw_date:
            return now - timedelta(days=1)
        else:
            return datetime.strptime(raw_date, "%b %d, %Y %H:%M")
    except Exception:
        return now


def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


def scroll_down(driver, scroll_pause=2.0, max_scrolls=5):
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)


def scrape_investing_news(days_back=3, scrolls=5):
    logger.info("Scraping Investing.com news using Selenium...")
    driver = setup_selenium()
    results = []
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)

    try:
        driver.get(INVESTING_URL)
        time.sleep(4)
        scroll_down(driver, max_scrolls=scrolls)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.select("article.js-article-item")

        for article in articles:
            title_tag = article.find("a", class_="title")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            article_url = "https://www.investing.com" + title_tag["href"]

            date_tag = article.find("span", class_="date")
            raw_date = date_tag.get_text(strip=True) if date_tag else ""
            published = parse_date_string(raw_date)

            if published < cutoff_date:
                logger.info("Reached old articles. Stopping.")
                break

            if matches_keywords(title):
                results.append({
                    "title": title,
                    "url": article_url,
                    "source": "Investing.com",
                    "published_date": published.strftime("%Y-%m-%d %H:%M:%S"),
                    "scraped_at": datetime.utcnow().isoformat()
                })

    except Exception as e:
        logger.exception(f"Scraping failed: {e}")
    finally:
        driver.quit()

    return results


def run_scraper():
    news = scrape_investing_news()

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(news, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(news)} filtered articles to {OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Error saving Investing.com news: {e}")


if __name__ == "__main__":
    run_scraper()
    