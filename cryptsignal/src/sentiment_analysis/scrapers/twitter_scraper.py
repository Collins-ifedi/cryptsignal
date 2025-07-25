# File: src/sentiment_analysis/twitter_scraper.py

import sys
import os
import time
import logging
import requests
from pathlib import Path

# Access cryptsignal project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.credentials_loader import load_credentials

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TwitterScraper:
    def __init__(self, config_path='config/credentials.yaml'):
        self.config_path = config_path
        self.api_key = self._load_api_key()

        if not self.api_key:
            raise ValueError("Missing SerpAPI key in credentials.yaml under ['twitter']['serpapi_key']")

    def _load_api_key(self):
        try:
            credentials = load_credentials(self.config_path)
            if not isinstance(credentials, dict):
                raise ValueError("Invalid format in credentials.yaml")
            return credentials.get("twitter", {}).get("serpapi_key")
        except Exception as e:
            logger.error(f"Failed to load SerpAPI key: {e}")
            return None

    def scrape(self, coin_name: str, max_results: int = 30, retries: int = 3, delay: int = 15):
        """
        Scrape Twitter posts via SerpAPI for sentiment analysis.

        Args:
            coin_name (str): Cryptocurrency name
            max_results (int): Max results to fetch
            retries (int): Retry attempts
            delay (int): Delay between retries

        Returns:
            list[dict]: List of tweet-like entries with 'content'
        """
        query = f"{coin_name} crypto site:twitter.com"
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": max_results,
            "hl": "en",
            "gl": "us"
        }

        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"Scraping Twitter for '{coin_name}' (attempt {attempt + 1})...")
                response = requests.get("https://serpapi.com/search", params=params)

                if response.status_code == 429:
                    logger.warning("Rate limit hit (429). Retrying...")
                    time.sleep(delay)
                    attempt += 1
                    continue

                response.raise_for_status()
                data = response.json()

                posts = []
                for entry in data.get("organic_results", []):
                    title = entry.get("title", "")
                    snippet = entry.get("snippet", "")
                    content = f"{title} {snippet}".strip()
                    if content:
                        posts.append({"content": content})

                logger.info(f"Retrieved {len(posts)} Twitter posts for {coin_name}.")
                return posts

            except Exception as e:
                logger.exception(f"Error scraping Twitter (attempt {attempt + 1}): {e}")
                time.sleep(delay)
                attempt += 1

        logger.error(f"Failed to scrape Twitter after {retries} attempts.")
        return []

if __name__ == "__main__":
    try:
        scraper = TwitterScraper(config_path= Path(PROJECT_ROOT)/ "config"/ "credentials.yaml")
        results = scraper.scrape("Ethereum", max_results=10)
        for i, post in enumerate(results, 1):
            print(f"{i}. {post['content']}")
    except Exception as e:
        logger.exception("Test run failed")