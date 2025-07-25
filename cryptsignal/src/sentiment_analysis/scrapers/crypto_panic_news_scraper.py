import os
import sys
import logging
import requests
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CryptoPanicNewsScraper:
    """
    Scraper class to fetch recent cryptocurrency news from CryptoPanic API,
    filtered optionally by a given coin name.
    """
    
    def __init__(self, config_path= Path(PROJECT_ROOT)/ "config/credentials.yaml"):
        self.api_key = self._load_api_key(config_path)
        if not self.api_key:
            raise ValueError("CryptoPanic API key not found in credentials file.")
        
        self.base_url = "https://cryptopanic.com/api/v1/posts/"
        self.params = {
            "auth_token": self.api_key,
            "public": "true",
            "kind": "news"
        }

    def _load_api_key(self, filepath):
        try:
            with open(filepath, "r") as f:
                credentials = yaml.safe_load(f)
            return credentials.get("cryptopanic", {}).get("api_key")
        except FileNotFoundError:
            logger.error(f"Credentials file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load CryptoPanic API key: {e}")
        return None

    def fetch_news(self, coin_name=None, limit=10):
        """
        Fetches and filters news articles from CryptoPanic based on a coin name.

        Args:
            coin_name (str): Name of the coin to filter news (e.g. 'Bitcoin').
            limit (int): Max number of articles to return.

        Returns:
            List[dict]: Filtered news articles with metadata.
        """
        try:
            response = requests.get(self.base_url, params=self.params)
            response.raise_for_status()
            data = response.json()
            articles = data.get("results", [])
            
            if not articles:
                logger.warning("No news articles found from CryptoPanic.")
                return []

            filtered_news = []
            for article in articles:
                title = article.get("title", "")
                summary = article.get("slug", "")
                
                if coin_name:
                    if coin_name.lower() not in title.lower() and coin_name.lower() not in summary.lower():
                        continue

                filtered_news.append({
                    "title": title,
                    "url": article.get("url"),
                    "published_at": article.get("published_at"),
                    "source": article.get("source", {}).get("title", "Unknown"),
                    "summary": summary
                })

                if len(filtered_news) >= limit:
                    break

            logger.info(f"Fetched {len(filtered_news)} news articles from CryptoPanic for '{coin_name}'.")
            return filtered_news

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from CryptoPanic: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during news fetch: {e}")
            return []

# Optional test run
if __name__ == "__main__":
    try:
        scraper = CryptoPanicNewsScraper(
            config_path="/storage/emulated/0/Android/data/ru.iiec.pydroid3/files/cryptsignal/config/credentials.yaml"
        )
        coin = "Bitcoin"
        news = scraper.fetch_news(coin_name=coin, limit=5)
        if news:
            print(f"\nLatest News for {coin}:\n")
            for i, item in enumerate(news, 1):
                print(f"{i}. {item['title']} ({item['published_at']})")
                print(f"   Source: {item['source']}")
                print(f"   URL: {item['url']}\n")
        else:
            print(f"No relevant news found for {coin}.")
    except Exception as e:
        logger.exception("Test run failed.")