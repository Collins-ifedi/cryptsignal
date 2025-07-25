import os
import json
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set up the logger for this scraper
logger = get_logger("IMFWorldBankScraper")

# Define headers for requests to avoid blocks by the website
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Define the sources for IMF and World Bank
SOURCES = {
    "IMF": "https://www.imf.org/en/News",
    "World Bank": "https://www.worldbank.org/en/news/all"
}

# Define the output path for the scraped data
OUTPUT_PATH = "data/macro/imf_worldbank_news.json"

def scrape_imf_news(max_results=20):
    """Scrape recent news from the IMF website."""
    logger.info("Scraping IMF news...")
    results = []
    
    try:
        # Make a request to the IMF news page
        response = requests.get(SOURCES["IMF"], headers=HEADERS, timeout=10)
        response.raise_for_status()  # Ensure we get a successful response
        soup = BeautifulSoup(response.content, "html.parser")

        # Select the news article elements
        articles = soup.select("div.media-body h3 a")
        
        for article in articles[:max_results]:
            title = article.get_text(strip=True)
            url = article.get("href")
            if not url.startswith("http"):
                url = "https://www.imf.org" + url  # Ensure full URL

            # Append the article data
            results.append({
                "title": title,
                "url": url,
                "source": "IMF",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        logger.exception(f"Failed to scrape IMF news: {e}")

    return results

def scrape_world_bank_news(max_results=20):
    """Scrape recent news from the World Bank website."""
    logger.info("Scraping World Bank news...")
    results = []

    try:
        # Make a request to the World Bank news page
        response = requests.get(SOURCES["World Bank"], headers=HEADERS, timeout=10)
        response.raise_for_status()  # Ensure we get a successful response
        soup = BeautifulSoup(response.content, "html.parser")

        # Select the news article elements
        articles = soup.select("div.headline a")
        
        for article in articles[:max_results]:
            title = article.get_text(strip=True)
            url = article.get("href")
            if not url.startswith("http"):
                url = "https://www.worldbank.org" + url  # Ensure full URL

            # Append the article data
            results.append({
                "title": title,
                "url": url,
                "source": "World Bank",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        logger.exception(f"Failed to scrape World Bank news: {e}")

    return results

def fetch_imf_worldbank_reports(max_results=20):
    """Fetch and save IMF & World Bank news without filtering by coin."""
    logger.info("Fetching IMF & World Bank reports...")
    news = scrape_imf_news(max_results) + scrape_world_bank_news(max_results)

    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        # Save the collected news to the output JSON file
        with open(OUTPUT_PATH, "w") as f:
            json.dump(news, f, indent=2)
        
        logger.info(f"Saved {len(news)} articles to {OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Error saving IMF/World Bank news: {e}")

    return news

# Optional direct run: If executed directly, the script will fetch and save the reports
if __name__ == "__main__":
    fetch_imf_worldbank_reports()
