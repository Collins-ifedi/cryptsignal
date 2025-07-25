import os
import sys
import json
import yaml
import requests
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("MacroNewsScraper")

# Constants
CREDENTIALS_PATH = Path(PROJECT_ROOT)/ "config"/ "credentials.yaml"
OUTPUT_DIR = "data/macro"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
TOPICS = [
    "global economy", "geopolitical tensions", "crypto adoption", "crypto ban",
    "crypto regulation", "SEC ruling crypto", "court decision cryptocurrency",
    "gold price", "central bank gold reserve", "trade tariffs", "economic sanctions"
]

def load_serpapi_key():
    """Load SerpAPI key from YAML credentials file."""
    try:
        with open(CREDENTIALS_PATH, "r") as f:
            creds = yaml.safe_load(f)
            api_key = creds.get("serpapi_key")
            if not api_key:
                logger.error("Missing SerpAPI key in credentials.")
                sys.exit(1)
            return api_key
    except Exception as e:
        logger.exception(f"Failed to load SerpAPI key: {e}")
        sys.exit(1)

SERPAPI_KEY = load_serpapi_key()

def fetch_news_for_topic(topic):
    """Fetch news articles from SerpAPI for a specific topic."""
    try:
        logger.info(f"Fetching news for topic: {topic}")
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": topic,
            "api_key": SERPAPI_KEY,
            "num": 10,
            "hl": "en",
            "gl": "us"
        }

        response = requests.get(url, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()

        results = response.json().get("organic_results", [])
        articles = []
        for result in results:
            link = result.get("link", "")
            if "reuters.com" in link:  # Skip Reuters
                continue

            articles.append({
                "topic": topic,
                "title": result.get("title", "No Title"),
                "url": link,
                "snippet": result.get("snippet", ""),
                "source": result.get("source", "Unknown"),
                "timestamp": datetime.utcnow().isoformat()
            })

        return articles

    except Exception as e:
        logger.error(f"Error fetching news for topic '{topic}': {e}")
        return []

def fetch_all_macro_news():
    """Fetch macroeconomic and geopolitical news for all defined topics."""
    all_articles = []
    for topic in TOPICS:
        topic_articles = fetch_news_for_topic(topic)
        all_articles.extend(topic_articles)
    return all_articles

def save_articles_to_file(articles):
    """Save articles to a JSON file with timestamped filename."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"macro_news_serpapi_{timestamp}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(articles)} articles to {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save articles: {e}")

def main():
    logger.info("Starting SerpAPI macro news scraper...")
    articles = fetch_all_macro_news()
    save_articles_to_file(articles)
    logger.info("Scraper finished.")

if __name__ == "__main__":
    main()

