import requests
import yaml
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SerpAPIFallback")

# Load credentials
def load_serpapi_key():
    try:
        with open(Path(PROJECT_ROOT)/ "config"/ "credentials.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config.get("serpapi_key")
    except Exception as e:
        logger.error(f"Error loading SerpAPI key: {e}")
        return None

# Run SerpAPI Google search for fallback
def fetch_fallback_data(indicator_title, coin_name="Bitcoin"):
    serpapi_key = load_serpapi_key()
    if not serpapi_key:
        return {
            "text": f"Unable to fetch fallback data for {indicator_title} (missing API key).",
            "sentiment": 0
        }

    try:
        query = f"{indicator_title} trend United States site:tradingeconomics.com OR site:investing.com OR site:cnbc.com"
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 5
        }

        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        results = response.json()

        organic_results = results.get("organic_results", [])
        if not organic_results:
            return {
                "text": f"Could not retrieve fallback information for {indicator_title}.",
                "sentiment": 0
            }

        top_summary = organic_results[0].get("snippet") or organic_results[0].get("title", "")
        direction = "changing"
        if "increase" in top_summary.lower() or "rise" in top_summary.lower() or "growth" in top_summary.lower():
            direction = "increasing"
        elif "decrease" in top_summary.lower() or "fall" in top_summary.lower() or "decline" in top_summary.lower():
            direction = "decreasing"

        # Rule-based sentiment logic
        bearish_keywords = ["unemployment", "jobless", "inflation", "interest rate"]
        bullish_keywords = ["gdp", "retail", "payroll", "confidence", "manufacturing", "sales"]

        sentiment = 0
        if any(k in indicator_title.lower() for k in bearish_keywords):
            sentiment = -1 if direction == "increasing" else 1
        elif any(k in indicator_title.lower() for k in bullish_keywords):
            sentiment = 1 if direction == "increasing" else -1

        text = f"The {indicator_title} appears to be {direction} (fallback source), which may affect {coin_name} sentiment."

        return {
            "text": text,
            "sentiment": sentiment
        }

    except Exception as e:
        logger.error(f"SerpAPI fallback failed for {indicator_title}: {e}")
        return {
            "text": f"Error analyzing {indicator_title} using fallback source: {str(e)}",
            "sentiment": 0
        }