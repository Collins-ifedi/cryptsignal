import sys
import os
import requests
import logging
from datetime import datetime
from pathlib import Path

# Set up path to access project modules
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

class RedditScraper:
    """
    Uses SerpAPI to scrape Reddit posts related to a specific cryptocurrency.
    """

    def __init__(self, *, subreddit: str = "CryptoCurrency", config_path: str = Path(PROJECT_ROOT)/ "config"/"credentials.yaml"):
        self.subreddit = subreddit
        self.config_path = config_path
        self.api_key = self._load_serpapi_key()
        if not self.api_key:
            raise ValueError("Missing SerpAPI key in credentials.yaml under ['reddit']['serpapi_key']")

    def _load_serpapi_key(self):
        try:
            credentials = load_credentials(self.config_path)
            if not isinstance(credentials, dict):
                raise ValueError("Invalid format in credentials.yaml.")
            api_key = credentials.get("reddit", {}).get("serpapi_key")
            if not api_key:
                raise ValueError("SerpAPI key not found under 'reddit' in credentials.")
            return api_key
        except Exception as e:
            logger.error(f"Failed to load SerpAPI key: {e}")
            return None

    def scrape(self, coin_name: str, days_back: int = 1, max_results: int = 50):
        """
        Scrapes Reddit using SerpAPI for mentions of a specific coin.

        Args:
            coin_name (str): Cryptocurrency name to search for.
            days_back (int): How many days of history to consider (not used in search query but available for future extension).
            max_results (int): Max number of posts to fetch.

        Returns:
            list[dict]: List of posts with 'content' fields.
        """
        try:
            logger.info(f"Scraping r/{self.subreddit} for '{coin_name}', days_back={days_back}")

            params = {
                "engine": "google",
                "q": f"{coin_name} site:reddit.com/r/{self.subreddit}",
                "api_key": self.api_key,
                "num": max_results,
                "hl": "en",
                "gl": "us",
            }

            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            results = response.json()

            posts = []
            for entry in results.get("organic_results", []):
                title = entry.get("title", "")
                snippet = entry.get("snippet", "")
                content = f"{title} {snippet}".strip()
                if content:
                    posts.append({
                        "content": content,
                        "source": "reddit",
                        "timestamp": datetime.utcnow().isoformat()
                    })

            logger.info(f"Retrieved {len(posts)} Reddit posts for '{coin_name}' from r/{self.subreddit}")
            return posts

        except Exception as e:
            logger.exception(f"Reddit scraping failed for '{coin_name}': {e}")
            return []

# Optional test run
if __name__ == "__main__":
    try:
        scraper = RedditScraper(subreddit="CryptoCurrency", config_path= Path(PROJECT_ROOT)/ "config"/ "credentials.yaml")
        results = scraper.scrape("Solana", days_back=1, max_results=10)
        if results:
            for i, post in enumerate(results, 1):
                print(f"{i}. {post['content']}")
        else:
            print("No posts found.")
    except Exception as e:
        logger.exception("Test run failed")

        