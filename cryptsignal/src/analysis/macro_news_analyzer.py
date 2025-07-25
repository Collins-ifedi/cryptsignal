import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional # Added for type hinting

# Add root directory for local imports
# Ensure this path is correct for your Pydroid environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom logger
try:
    from utils.logger import get_logger # Assuming this is the project's standard logger
    logger = get_logger("MacroNewsAnalyzer")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MacroNewsAnalyzer_Fallback")
    logger.warning("Custom logger not found, using basic logging for MacroNewsAnalyzer.")


# Import individual scrapers
try:
    from macro_news.scrapers.cnbc_scraper import fetch_cnbc_news
    from macro_news.scrapers.yahoo_finance_scraper import fetch_yahoo_finance_reports
    from macro_news.scrapers.imf_worldbank_scraper import fetch_imf_worldbank_reports
    from macro_news.scrapers.federal_reserve_scraper import scrape_federal_reserve_news
    from macro_news.scrapers.government_news_scraper import scrape_government_news
    from macro_news.scrapers.serpapi_news_scraper import fetch_all_macro_news
except ImportError as e:
    logger.error(f"Failed to import one or more scrapers: {e}. Some news sources may be unavailable.", exc_info=True)
    # Define dummy functions if scrapers are critical for the script to load, though it's better to fix imports
    def fetch_cnbc_news(**kwargs): logger.error("fetch_cnbc_news not available"); return []
    def fetch_yahoo_finance_reports(**kwargs): logger.error("fetch_yahoo_finance_reports not available"); return []
    def fetch_imf_worldbank_reports(**kwargs): logger.error("fetch_imf_worldbank_reports not available"); return []
    def scrape_federal_reserve_news(**kwargs): logger.error("scrape_federal_reserve_news not available"); return []
    def scrape_government_news(**kwargs): logger.error("scrape_government_news not available"); return []
    def fetch_all_macro_news(**kwargs): logger.error("fetch_all_macro_news not available"); return []


# Import sentiment utility
# This will use the updated hybrid_sentiment from sentiment_utils.py
# which now returns a Dict with score, category, and error.
try:
    from src.sentiment_analysis.sentiment_utils import hybrid_sentiment
except ImportError:
    logger.critical("Failed to import hybrid_sentiment from sentiment_utils. Ensure sentiment_utils.py is in the correct path and updated.", exc_info=True)
    # Define a dummy hybrid_sentiment if it's critical for the script to load,
    # though this means sentiment analysis will not work.
    def hybrid_sentiment(text: str) -> Dict[str, Any]:
        logger.error("hybrid_sentiment function is not available due to import error.")
        return {"score": 0.0, "category": "neutral", "error": "hybrid_sentiment not imported"}


# Storage paths
HISTORY_DIR = os.path.join(PROJECT_ROOT, "macro_news/history")
CACHE_DIR = os.path.join(PROJECT_ROOT, "macro_news/cache")
HISTORY_FILE_BASENAME = "news_sentiment.json" # Basename for timestamped history files
CACHE_FILE = os.path.join(CACHE_DIR, "news_cache.json")


def analyze_sentiment(posts: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Add sentiment tags (score, category, and raw details) to news articles
    using LLM via the updated hybrid_sentiment function.
    """
    analyzed: List[Dict[str, Any]] = []
    if not posts:
        logger.info(f"[{source.upper()}] No posts to analyze.")
        return analyzed
        
    for post_idx, post in enumerate(posts):
        # Robust content extraction
        content = post.get("content") or post.get("snippet") or post.get("title", "")
        
        sentiment_details: Dict[str, Any] # Type hint for clarity

        if not isinstance(content, str) or not content.strip():
            logger.warning(f"[{source.upper()}] Post {post_idx+1} (ID/Link: {post.get('id', post.get('link', 'N/A'))}) has invalid or empty content. Assigning neutral sentiment.")
            sentiment_details = {"score": 0.0, "category": "neutral", "error": "Invalid or empty content for analysis"}
        else:
            # hybrid_sentiment is expected to return a dict like:
            # {"score": 0.75, "category": "bullish", "error": None}
            # or {"score": 0.0, "category": "neutral", "error": "Some error message"}
            sentiment_details = hybrid_sentiment(content)
            
            if sentiment_details.get("error"):
                logger.warning(
                    f"[{source.upper()}] Sentiment analysis for post {post_idx+1} "
                    f"(Source content preview: '{content[:70]}...') "
                    f"returned an error: {sentiment_details['error']}. Using fallback neutral sentiment."
                )
        
        # Update post with detailed sentiment information
        post.update({
            "sentiment": sentiment_details.get("score", 0.0),  # Retained for potential backward compatibility, stores the score.
            "sentiment_score": sentiment_details.get("score", 0.0),
            "sentiment_category": str(sentiment_details.get("category", "neutral")).lower(), # Ensure category is string and lowercase
            "sentiment_analysis_details": sentiment_details, # Stores the full dict {"score", "category", "error"}
            "source": source,
            # "analysis_timestamp": datetime.utcnow().isoformat() # Optional: timestamp per item analysis
        })
        analyzed.append(post)
        logger.debug(f"[{source.upper()}] Post {post_idx+1} analyzed. Score: {post.get('sentiment_score')}, Category: '{post.get('sentiment_category')}'")

    logger.info(f"[{source.upper()}] Processed and analyzed {len(analyzed)} articles with new sentiment structure.")
    return analyzed


def save_to_history_file(news_data: List[Dict[str, Any]], timestamp: str) -> None:
    """Persist timestamped sentiment-tagged news into a new history snapshot file."""
    if not news_data:
        logger.info("No news data to save to history.")
        return
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        output_filename = HISTORY_FILE_BASENAME.replace('.json', f'_{timestamp}.json')
        output_file_path = os.path.join(HISTORY_DIR, output_filename)
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
        logger.info(f"News history snapshot stored at {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save news history snapshot: {e}", exc_info=True)


def save_cache(data: List[Dict[str, Any]]) -> None:
    """Save recent sentiment-analyzed news to the cache file (overwrites)."""
    if not data:
        logger.info("No data to save to cache.")
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Latest news cache updated at {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Cache update failed: {e}", exc_info=True)


def fetch_and_analyze_news_sources(coin_name: Optional[str] = None, max_results_per_source: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches and analyzes news from all integrated sources.
    Handles errors from individual sources to ensure partial data collection.
    """
    all_results: List[Dict[str, Any]] = []
    
    # Ensure all scraper functions are callable, even if defined as dummies due to import errors.
    scraper_configs = [
        {"name": "CNBC", "func": fetch_cnbc_news, "params": {"coin_name": coin_name, "max_results": max_results_per_source}},
        {"name": "Yahoo Finance", "func": fetch_yahoo_finance_reports, "params": {"max_results": max_results_per_source}},
        {"name": "IMF & World Bank", "func": fetch_imf_worldbank_reports, "params": {"max_results": max_results_per_source}},
        {"name": "Federal Reserve", "func": scrape_federal_reserve_news, "params": {"max_results": 20}},
        {"name": "Global Government", "func": scrape_government_news, "params": {}}, # Might need max_results too
        {"name": "SerpAPI Macro", "func": fetch_all_macro_news, "params": {}} # Might need max_results too
    ]

    for config in scraper_configs:
        try:
            if not callable(config["func"]):
                logger.error(f"Scraper function for {config['name']} is not callable. Skipping.")
                continue

            logger.info(f"Fetching {config['name']} news...")
            news_items = config["func"](**config['params']) # Call scraper

            # Ensure news_items is a list before passing to analyze_sentiment
            if not isinstance(news_items, list):
                logger.warning(f"{config['name']} scraper did not return a list (got {type(news_items)}). Skipping analysis for this source.")
                news_items = [] # Default to empty list to avoid error in extend

            # Analyze sentiment for the fetched items
            analyzed_items = analyze_sentiment(news_items, config['name'].lower().replace(" & ", "_").replace(" ", "_"))
            all_results.extend(analyzed_items)

        except Exception as e:
            logger.error(f"Failed to fetch or analyze {config['name']} news: {e}", exc_info=True)
            
    if all_results:
        current_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        save_to_history_file(all_results, current_timestamp)
    else:
        logger.warning("No news data was collected from any source. History file not created for this run.")
        
    return all_results


def fetch_and_analyze_macro_news(coin_name: Optional[str] = None, max_results: int = 30) -> List[Dict[str, Any]]:
    """Main pipeline for macro news fetching, sentiment tagging, and caching."""
    try:
        logger.info(f"Starting macroeconomic sentiment pipeline for coin: {coin_name if coin_name else 'General'} (max_results per source: {max_results})...")
        news_data = fetch_and_analyze_news_sources(coin_name=coin_name, max_results_per_source=max_results)
        
        if news_data:
            save_cache(news_data)
            logger.info(f"Macro news pipeline completed. Total articles processed: {len(news_data)}.")
        else:
            logger.warning("Pipeline completed but no news data was collected or analyzed. Cache not updated.")
            
        return news_data
    except Exception as e:
        logger.error(f"Macro news pipeline failed critically: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    logger.info("--- Macro News Analyzer Script Started (Main Test) ---")
    
    # --- IMPORTANT ---
    # For this example to work with LLM sentiment, you MUST ensure:
    # 1. sentiment_utils.py is updated to use Llama 4 Maverick.
    # 2. The OPENROUTER_API_KEY environment variable is set for sentiment_utils.py to access.
    # Example (in your terminal before running the script):
    # export OPENROUTER_API_KEY="your_actual_openrouter_api_key_here"
    # -----------------

    # Test if hybrid_sentiment is available (basic check after imports)
    if not callable(hybrid_sentiment) or hybrid_sentiment("test").get("error") == "hybrid_sentiment not imported":
        logger.critical("Dummy hybrid_sentiment is in use. LLM sentiment analysis will NOT work. Please check imports.")
    else:
        logger.info("hybrid_sentiment appears to be imported correctly.")


    example_coin_name = "bitcoin"  # Example for crypto relevance
    # example_coin_name = None # To test general macro news (uses default coin context in scrapers or general queries)
    
    # Reduced max_results for faster test; increase for more comprehensive data
    results = fetch_and_analyze_macro_news(coin_name=example_coin_name, max_results=2) 
    
    if results:
        logger.info(f"Successfully fetched and analyzed {len(results)} macro news items in total.")
        logger.info("--- Example of first result's sentiment data ---")
        try:
            # Print details of the first result for inspection if results exist
            first_result = results[0]
            print(json.dumps({
                "title_preview": first_result.get("title", "N/A")[:50] + "...",
                "source": first_result.get("source"),
                "sentiment_score": first_result.get("sentiment_score"),
                "sentiment_category": first_result.get("sentiment_category"),
                "sentiment_analysis_details": first_result.get("sentiment_analysis_details")
            }, indent=2))
        except IndexError:
            logger.info("Results list was unexpectedly empty after check, cannot display first result.")
        except Exception as e:
            logger.error(f"Error displaying first result example: {e}")

    else:
        logger.warning("No macro news items were processed in the test run.")
    logger.info("--- Macro News Analyzer Script Finished ---")