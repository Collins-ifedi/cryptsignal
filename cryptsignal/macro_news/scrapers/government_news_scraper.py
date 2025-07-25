import os
import json
import time
import sys
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("GovNewsScraper")

OUTPUT_PATH = "data/macro/government_news.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

GOV_SITES = {
    "USA": "https://www.whitehouse.gov/briefing-room/",
    "China": "http://english.www.gov.cn/news/",
    "India": "https://pib.gov.in/PressReleasePage.aspx",
    "Canada": "https://www.canada.ca/en/news.html",
    "Japan": "https://japan.kantei.go.jp/",
    "Russia": "http://government.ru/en/news/",
    "UK": "https://www.gov.uk/government/announcements"
}

def create_session():
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session

session = create_session()

def extract_articles(soup, country, base_url):
    articles = []

    try:
        if country == "USA":
            for a in soup.select("article a.card-title")[:10]:
                articles.append((a.text.strip(), a["href"]))

        elif country == "China":
            for a in soup.select(".news_box .list_item h4 a")[:10]:
                articles.append((a.text.strip(), urljoin(base_url, a["href"])))

        elif country == "India":
            for a in soup.select(".content-area .col-sm-9 a")[:10]:
                articles.append((a.text.strip(), urljoin(base_url, a["href"])))

        elif country == "Canada":
            for a in soup.select(".wb-feeds a")[:10]:
                articles.append((a.text.strip(), a["href"]))

        elif country == "Japan":
            for a in soup.select(".title a")[:10]:
                articles.append((a.text.strip(), urljoin(base_url, a["href"])))

        elif country == "Russia":
            for a in soup.select(".block-news a")[:10]:
                articles.append((a.text.strip(), urljoin(base_url, a["href"])))

        elif country == "UK":
            for item in soup.select(".gem-c-document-list__item-title")[:10]:
                parent_link = item.find_parent("a")
                link = urljoin(base_url, parent_link["href"]) if parent_link else base_url
                articles.append((item.text.strip(), link))

    except Exception as e:
        logger.exception(f"Failed to parse articles for {country}: {e}")

    return [
        {
            "title": title,
            "url": url,
            "country": country,
            "source": base_url,
            "scraped_at": datetime.utcnow().isoformat()
        } for title, url in articles
    ]

def scrape_government_news():
    """Scrape the latest government news from various countries."""
    logger.info("Starting government news scraping...")
    all_news = []

    for country, url in GOV_SITES.items():
        try:
            logger.info(f"Fetching from {country}: {url}")
            response = session.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            articles = extract_articles(soup, country, url)
            all_news.extend(articles)
        except Exception as e:
            logger.error(f"Error scraping {country}: {e}")

    logger.info(f"Collected {len(all_news)} articles from all sources.")
    return all_news

def save_news(data):
    """Save the scraped government news to a JSON file."""
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} government news items to {OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Failed to save government news: {e}")

def run_scraper():
    """Run the scraper and save the results."""
    data = scrape_government_news()
    save_news(data)

if __name__ == "__main__":
    run_scraper()