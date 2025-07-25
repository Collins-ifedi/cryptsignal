import os
import json
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Pydroid-specific path insertion (ensure this is correct for your environment)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Scrapers ---
# It's assumed these scrapers are correctly set up and importable.
# Their internal configuration (like API keys) should be handled within the scraper classes.
try:
    from src.sentiment_analysis.scrapers.reddit_scraper import RedditScraper
    from src.sentiment_analysis.scrapers.twitter_scraper import TwitterScraper
    from src.sentiment_analysis.scrapers.crypto_panic_news_scraper import CryptoPanicNewsScraper
except ImportError as e:
    # Fallback basic logging if get_logger is not found yet
    import logging as temp_logging
    temp_logging.basicConfig(level=temp_logging.ERROR)
    temp_logging.error(f"Critical import error for scrapers: {e}. Some functionalities might be unavailable.")
    
# --- Sentiment Utilities ---
# This will import the hybrid_sentiment that now uses LLM and returns a Dict.
try:
    from src.sentiment_analysis.sentiment_utils import hybrid_sentiment
except ImportError:
    import logging as temp_logging
    temp_logging.basicConfig(level=temp_logging.ERROR)
    temp_logging.critical("Failed to import hybrid_sentiment from sentiment_utils. Sentiment analysis will not work.", exc_info=True)
    # Define a dummy hybrid_sentiment if it's critical for the script to load
    def hybrid_sentiment(text: str) -> Dict[str, Any]:
        temp_logging.error("hybrid_sentiment function is not available due to import error.")
        return {"score": 0.0, "category": "neutral", "error": "hybrid_sentiment not imported"}

# --- Analysis Modules ---
# These modules fetch external data or perform other types of analysis.
try:
    from src.analysis.fear_greed_index import get_fear_greed_index
    from src.analysis.dxy_strength import fetch_dxy_strength
    # fetch_and_analyze_macro_news now uses the updated macro_news_analyzer.py
    from src.analysis.macro_news_analyzer import fetch_and_analyze_macro_news
except ImportError as e:
    import logging as temp_logging
    temp_logging.basicConfig(level=temp_logging.ERROR)
    temp_logging.error(f"Critical import error for analysis modules: {e}. Some functionalities might be unavailable.")
    def get_fear_greed_index(*args, **kwargs): return []
    def fetch_dxy_strength(*args, **kwargs): return None
    def fetch_and_analyze_macro_news(*args, **kwargs): return []


# --- Project Utilities ---
try:
    from utils.logger import get_logger
    logger = get_logger("SentimentAnalysis")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("SentimentAnalysis_Fallback")
    logger.warning("Custom logger not found, using basic logging for SentimentAnalysis.")

# Import the new config_loader
try:
    from utils.credentials_loader import load_credentials
except ImportError:
    logger.error("Could not import config_loader. LLM API key loading from file will fail.")
    load_credentials = None # type: ignore


# --- LLM API Key Configuration (for use by this module or shared with sentiment_utils) ---
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

_llm_api_key: Optional[str] = None
_credentials_loaded: bool = False # Flag to ensure credentials are only loaded once

def _get_openrouter_api_key() -> Optional[str]:
    global _llm_api_key, _credentials_loaded
    
    if _llm_api_key is None:
        # 1. Try environment variable first
        _llm_api_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
        if _llm_api_key:
            logger.info("OpenRouter API key loaded from environment variable.")
            return _llm_api_key

        # 2. If not in environment, try loading from credentials.yaml
        if not _credentials_loaded and load_credentials: # Check if loader was imported
            credentials = load_credentials() # Load the credentials file
            if credentials:
                _llm_api_key = credentials.get('openrouter', {}).get('api_key')
                if _llm_api_key:
                    logger.info("OpenRouter API key loaded from credentials.yaml.")
                else:
                    logger.warning("OpenRouter API key not found in credentials.yaml under 'openrouter: api_key'.")
            else:
                logger.warning("Failed to load credentials from file.")
            _credentials_loaded = True # Mark as attempted to load from file

    if _llm_api_key is None:
        logger.error(f"{OPENROUTER_API_KEY_ENV_VAR} not found in environment or credentials.yaml. LLM functions may be skipped.")
    return _llm_api_key


# --- Constants ---
CONFIG_PATH = Path(PROJECT_ROOT)/ "config"/ "credentials.yaml" # Path for scrapers if they need it explicitly
SENTIMENT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sentiment")


def retry_request(func, *args, retries: int = 3, delay: int = 2, **kwargs) -> Any:
    """Helper function to retry network-bound functions."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {func.__name__} with error: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1)) # Exponential backoff for delay
            else:
                logger.error(f"Function {func.__name__} failed after {retries} attempts: {e}", exc_info=True)
                # Default return values for common failing function types
                if func.__name__ in ["scrape", "fetch_news", "get_fear_greed_index", "fetch_and_analyze_macro_news"]:
                    return [] # Empty list for functions expected to return lists
                elif func.__name__ == "fetch_dxy_strength":
                    return None # None for functions expected to return Optional[float]
                # For other types, or if re-raising is preferred, adjust here
                return None # General fallback
    return None


def analyze_posts(posts: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Analyzes a list of posts for sentiment using the updated hybrid_sentiment function.
    Each post dictionary is updated with 'sentiment_score', 'sentiment_category',
    and 'sentiment_analysis_details'.
    """
    analyzed: List[Dict[str, Any]] = []
    if not posts:
        logger.info(f"No posts from {source} to analyze.")
        return analyzed
        
    for post_idx, post in enumerate(posts): # Added index for better logging
        content = post.get("content", "") # Ensure content is a string
        if not isinstance(content, str):
            logger.warning(f"Post content from {source} (item {post_idx+1}) is not a string: {content}. Using empty string.")
            content = "" # Default to empty string if not a string
            
        sentiment_details: Dict[str, Any] # Type hint for clarity

        if not content.strip(): # If content is empty or only whitespace
            logger.warning(f"Post content from {source} (item {post_idx+1}, ID/Link: {post.get('id', post.get('link', 'N/A'))}) is effectively empty. Assigning neutral sentiment.")
            sentiment_details = {"score": 0.0, "category": "neutral", "error": "Empty content for analysis"}
        else:
            # hybrid_sentiment now returns a Dict: {"score": float, "category": str, "error": Optional[str]}
            sentiment_details = hybrid_sentiment(content) 

        # Log if the LLM call within hybrid_sentiment reported an error
        if sentiment_details.get("error"):
            logger.warning(
                f"Sentiment analysis for post {post_idx+1} from {source} "
                f"(content preview: '{content[:70]}...') "
                f"encountered an error: {sentiment_details['error']}. Using fallback neutral sentiment values."
            )
        
        # Store the detailed sentiment analysis
        post["sentiment_score"] = sentiment_details.get("score", 0.0)
        post["sentiment_category"] = str(sentiment_details.get("category", "neutral")).lower() # Ensure category is string and lowercase
        post["sentiment_analysis_details"] = sentiment_details # The full dict from hybrid_sentiment
        
        post["source"] = source # Ensure source is (re)assigned
        analyzed.append(post)
        logger.debug(f"Analyzed post {post_idx+1} from {source}. Score: {post['sentiment_score']}, Category: '{post['sentiment_category']}'")

    logger.info(f"Analyzed {len(analyzed)} posts from {source} using new sentiment structure.")
    return analyzed


def save_results(data: Dict[str, Any], output_path: str) -> None:
    """Saves the analysis results to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Sentiment results saved to: {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save sentiment results to {output_path}: {e}")


async def fetch_social_and_news_posts(coin_name: str, config_path_for_scrapers: str) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Fetches posts from Reddit, Twitter, and CryptoPanic concurrently."""
    loop = asyncio.get_running_loop()

    # Initialize scrapers. They might use config_path or their own default config mechanisms.
    # This assumes scrapers can be initialized this way.
    try:
        reddit_scraper = RedditScraper(config_path=config_path_for_scrapers)
        twitter_scraper = TwitterScraper(config_path=config_path_for_scrapers)
        cryptopanic_scraper = CryptoPanicNewsScraper(config_path=config_path_for_scrapers)
    except Exception as e: # Catch potential init errors if scrapers fail with path
        logger.error(f"Error initializing scrapers with config_path '{config_path_for_scrapers}': {e}. Scrapers may use defaults or fail.", exc_info=True)
        # Re-init without path if that's a fallback, or let them be None if they must have path
        reddit_scraper = RedditScraper() # Assuming they can init without path or handle None
        twitter_scraper = TwitterScraper()
        cryptopanic_scraper = CryptoPanicNewsScraper()


    scraper_tasks = []
    # Using ThreadPoolExecutor for blocking I/O tasks like scraping
    # max_workers can be adjusted based on typical performance of scrapers
    with ThreadPoolExecutor(max_workers=3) as executor:
        if hasattr(reddit_scraper, 'scrape') and callable(reddit_scraper.scrape):
            scraper_tasks.append(
                loop.run_in_executor(executor, retry_request, reddit_scraper.scrape, coin_name, 20) # Args: func, coin_name, limit
            )
        else: scraper_tasks.append(asyncio.sleep(0, result=[])); logger.warning("Reddit scraper 'scrape' method not available.") # Placeholder if scraper failed init

        if hasattr(twitter_scraper, 'scrape') and callable(twitter_scraper.scrape):
            scraper_tasks.append(
                loop.run_in_executor(executor, retry_request, twitter_scraper.scrape, coin_name, 20)
            )
        else: scraper_tasks.append(asyncio.sleep(0, result=[])); logger.warning("Twitter scraper 'scrape' method not available.")

        if hasattr(cryptopanic_scraper, 'fetch_news') and callable(cryptopanic_scraper.fetch_news):
            scraper_tasks.append(
                loop.run_in_executor(executor, retry_request, cryptopanic_scraper.fetch_news, coin_name, 20) # Assuming 'fetch_news' takes coin_name and limit
            )
        else: scraper_tasks.append(asyncio.sleep(0, result=[])); logger.warning("CryptoPanic scraper 'fetch_news' method not available.")
        
        # Gather results from all tasks
        results = await asyncio.gather(*scraper_tasks, return_exceptions=True)

    # Process results, handling potential exceptions from asyncio.gather
    processed_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Scraper task {i} failed with exception: {res}", exc_info=res)
            processed_results.append([]) # Append empty list on error
        elif res is None: # If retry_request returned None (should be [] based on its logic)
            logger.warning(f"Scraper task {i} returned None, defaulting to empty list.")
            processed_results.append([])
        else:
            processed_results.append(res if isinstance(res, list) else [])
    
    # Unpack with defaults if results array is shorter than expected (though gather should return for all)
    reddit_posts_raw = processed_results[0] if len(processed_results) > 0 else []
    twitter_posts_raw = processed_results[1] if len(processed_results) > 1 else []
    cmc_articles_raw = processed_results[2] if len(processed_results) > 2 else []
    
    return reddit_posts_raw, twitter_posts_raw, cmc_articles_raw


async def get_sentiment_snapshot(symbol_or_coin_name: str) -> Dict[str, Any]:
    """
    Generates a comprehensive sentiment snapshot for a given cryptocurrency symbol or name.
    It fetches social media posts, news, macroeconomic data, market indicators,
    analyzes sentiment using an LLM (via hybrid_sentiment), and aggregates the results.
    """
    # Use lowercase for coin_name consistency in scrapers if they expect it
    coin_name_for_scrapers = symbol_or_coin_name.lower() 
    
    logger.info(f"Starting sentiment snapshot for: {symbol_or_coin_name} (using '{coin_name_for_scrapers}' for scraping)")
    start_time_total = time.time()

    try:
        # --- 1. Fetch social media and CryptoPanic news posts concurrently ---
        fetch_posts_start_time = time.time()
        # Pass CONFIG_PATH for scrapers; they can use it if needed for API keys etc.
        reddit_posts_raw, twitter_posts_raw, cmc_articles_raw = await fetch_social_and_news_posts(coin_name_for_scrapers, CONFIG_PATH)
        logger.info(f"Fetched social/news posts for {symbol_or_coin_name} in {time.time() - fetch_posts_start_time:.2f}s. "
                    f"Reddit: {len(reddit_posts_raw)}, Twitter: {len(twitter_posts_raw)}, CryptoPanic: {len(cmc_articles_raw)}")

        # --- 2. Fetch macroeconomic data and market indicators concurrently ---
        logger.info("Fetching macroeconomic data and market indicators...")
        macro_market_start_time = time.time()
        loop = asyncio.get_running_loop()
        # Using ThreadPoolExecutor for potentially blocking I/O operations
        with ThreadPoolExecutor(max_workers=3) as macro_executor:
            # Pass coin_name_for_scrapers for context if the macro news analyzer uses it
            macro_news_future = loop.run_in_executor(macro_executor, retry_request, fetch_and_analyze_macro_news, coin_name_for_scrapers)
            fear_greed_future = loop.run_in_executor(macro_executor, retry_request, get_fear_greed_index) # Assumes default limit=1
            dxy_strength_future = loop.run_in_executor(macro_executor, retry_request, fetch_dxy_strength)

            # Await results
            macro_news_data = await macro_news_future
            fear_greed_data = await fear_greed_future
            dxy_strength_value = await dxy_strength_future
        
        # Ensure results are of expected types or default if retry_request failed badly
        macro_news_data = macro_news_data if isinstance(macro_news_data, list) else []
        fear_greed_data = fear_greed_data if isinstance(fear_greed_data, list) and len(fear_greed_data) > 0 else []
        # dxy_strength_value can be None if fetch fails, which is handled.

        logger.info(f"Fetched macro data & market indicators in {time.time() - macro_market_start_time:.2f}s.")
        logger.debug(f"Fear&Greed data (first item if any): {fear_greed_data[0] if fear_greed_data else 'N/A'}")
        logger.debug(f"DXY value: {dxy_strength_value}")
        # Macro news items already have new sentiment structure from macro_news_analyzer.py
        logger.debug(f"Macro news articles fetched: {len(macro_news_data)}. First item sentiment (if any): {macro_news_data[0].get('sentiment_analysis_details') if macro_news_data else 'N/A'}")


        # --- 3. Prepare and Analyze Sentiment for Social Media Posts ---
        logger.info("Analyzing sentiments for social media and CryptoPanic posts...")
        analyzed_posts_combined: List[Dict[str, Any]] = []

        analyzed_posts_combined.extend(analyze_posts(reddit_posts_raw, "reddit"))
        analyzed_posts_combined.extend(analyze_posts(twitter_posts_raw, "twitter"))
        
        # Process CryptoPanic articles: create a 'content' field from title & summary/slug
        processed_cmc_articles_for_sentiment: List[Dict[str, Any]] = []
        for article in cmc_articles_raw:
            title = article.get("title", "")
            # 'summary' from CryptoPanicNewsScraper is often the 'slug' or a brief text
            summary_or_slug = article.get("summary", article.get("slug", "")) 
            # Combine title and summary/slug for a richer content string for sentiment analysis
            article_content_for_sentiment = f"{title}. {summary_or_slug}".strip()
            
            article_copy = article.copy() # Avoid modifying original dict from scraper if it's reused
            article_copy["content"] = article_content_for_sentiment # Add the combined content field
            processed_cmc_articles_for_sentiment.append(article_copy)
        analyzed_posts_combined.extend(analyze_posts(processed_cmc_articles_for_sentiment, "cryptopanic"))
        
        # Calculate average social sentiment score using the 'sentiment_score' field
        valid_social_scores = []
        for post_item in analyzed_posts_combined:
            score = post_item.get("sentiment_score")
            if isinstance(score, (int, float)):
                valid_social_scores.append(score)
            else:
                logger.warning(f"Invalid or missing 'sentiment_score' in post from {post_item.get('source')}: {score}. Skipping for average calculation.")
        
        average_social_sentiment_score = sum(valid_social_scores) / len(valid_social_scores) if valid_social_scores else 0.0

        # Determine overall social sentiment category based on the average score
        overall_social_category = "neutral"
        if average_social_sentiment_score > 0.5: overall_social_category = "bullish"
        elif average_social_sentiment_score > 0.1: overall_social_category = "mini bullish"
        elif average_social_sentiment_score < -0.5: overall_social_category = "bearish"
        elif average_social_sentiment_score < -0.1: overall_social_category = "mini bearish"

        # --- 4. Construct the final result ---
        # Ensure Fear & Greed data is structured as expected (usually a list, take first item)
        current_fear_greed_index_details = None
        if fear_greed_data and isinstance(fear_greed_data, list) and len(fear_greed_data) > 0:
            if isinstance(fear_greed_data[0], dict):
                current_fear_greed_index_details = fear_greed_data[0]
            else:
                logger.warning(f"Fear & Greed data item is not a dict: {fear_greed_data[0]}")
        elif fear_greed_data: # If it's not a list or empty list
            logger.warning(f"Unexpected Fear & Greed data format: {fear_greed_data}")


        result = {
            "snapshot_timestamp_utc": datetime.utcnow().isoformat(),
            "symbol_analyzed": symbol_or_coin_name,
            "average_social_sentiment_score": round(average_social_sentiment_score, 4),
            "overall_social_sentiment_category": overall_social_category,
            "fear_greed_index": current_fear_greed_index_details, # Stores the first dict from F&G list or None
            "dxy_strength_score": dxy_strength_value if isinstance(dxy_strength_value, (float, int)) else None, # Ensure it's float/int or None
            "macro_economic_news": macro_news_data, # This list already contains items with new sentiment structure
            "total_social_posts_analyzed": len(analyzed_posts_combined), # Posts from Reddit, Twitter, CryptoPanic for sentiment
            "source_breakdown": {
                "reddit_posts_fetched": len(reddit_posts_raw),
                "twitter_tweets_fetched": len(twitter_posts_raw),
                "cryptopanic_articles_fetched": len(cmc_articles_raw)
            },
            # "raw_social_posts_analyzed": analyzed_posts_combined # Optionally include all analyzed social posts with their sentiment details
        }

        # --- 5. Save results to a file ---
        output_filename = f"{symbol_or_coin_name.replace('/', '_')}_sentiment_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(SENTIMENT_DATA_DIR, output_filename)
        save_results(result, output_path)

        logger.info(f"Sentiment snapshot for {symbol_or_coin_name} completed in {time.time() - start_time_total:.2f} seconds.")
        return result

    except Exception as e:
        logger.exception(f"Error generating sentiment snapshot for {symbol_or_coin_name}: {e}")
        return {
            "error": str(e),
            "symbol_analyzed": symbol_or_coin_name,
            "snapshot_timestamp_utc": datetime.utcnow().isoformat(),
            # Provide default empty/neutral values for other keys to maintain structure on error
            "average_social_sentiment_score": 0.0,
            "overall_social_sentiment_category": "neutral",
            "fear_greed_index": None,
            "dxy_strength_score": None,
            "macro_economic_news": [],
            "total_social_posts_analyzed": 0,
            "source_breakdown": {"reddit_posts_fetched": 0, "twitter_tweets_fetched": 0, "cryptopanic_articles_fetched": 0}
        }


async def run_sentiment_analysis_for_symbol(symbol: str) -> Dict[str, Any]:
    """Helper function to run sentiment analysis for a single symbol."""
    logger.info(f"Queuing sentiment analysis for symbol: {symbol}")
    analysis_result = await get_sentiment_snapshot(symbol)
    # logger.info(f"Sentiment analysis result for {symbol}: {json.dumps(analysis_result, indent=2)}") # Can be very verbose
    return analysis_result


if __name__ == "__main__":
    # --- IMPORTANT ---
    # For this script to work correctly with LLM-based sentiment, ensure:
    # 1. sentiment_utils.py is updated to use Llama 4 Maverick.
    # 2. The OPENROUTER_API_KEY environment variable is set for sentiment_utils.py to access.
    #    Example (in your terminal before running): export OPENROUTER_API_KEY="your_key"
    # 3. Scrapers and other analysis modules are correctly configured and their dependencies (e.g., API keys) are set up.
    # -----------------

    # Example: Test with a common cryptocurrency symbol
    test_symbol_input = "BTC"  # Can be "Bitcoin", "ETH", "Ethereum", "SOL/USDT", etc.
    
    # Ensure the API key is loaded from credentials.yaml or environment variables
    loaded_api_key = _get_openrouter_api_key()
    if not loaded_api_key:
        print("\nWARNING: OpenRouter API key could not be loaded. LLM-based sentiment analysis may fail.")
        logger.warning("OpenRouter API key not available for LLM calls in sentiment_utils.")

    # Check if hybrid_sentiment is the real one or the dummy
    if not callable(hybrid_sentiment) or hybrid_sentiment("test").get("error") == "hybrid_sentiment not imported":
        logger.critical("Dummy hybrid_sentiment is in use. LLM sentiment analysis will NOT work. Please check imports in sentiment_utils.py.")
        print("\nCRITICAL: `hybrid_sentiment` could not be imported correctly. Sentiment analysis will fail or use dummy data.")
        print("Ensure `sentiment_utils.py` is updated and in the correct PYTHONPATH.")
        # Optionally, exit if this core functionality is missing
        # sys.exit(1) 
    else:
        logger.info("`hybrid_sentiment` appears to be imported correctly.")
        # Quick check for OpenRouter API Key (sentiment_utils should handle this, but a pre-check is useful)
        # This check is now redundant if _get_openrouter_api_key() is called first, but kept for clarity.
        # if not os.environ.get("OPENROUTER_API_KEY"):
        #     logger.warning("OPENROUTER_API_KEY environment variable is not set. LLM calls in sentiment_utils.py will likely fail.")
        #     print("\nWARNING: OPENROUTER_API_KEY is not set. LLM-based sentiment analysis may fail.")


    print(f"\nRunning sentiment analysis for: {test_symbol_input}...")
    
    # Use asyncio.run to execute the main async function for the test
    try:
        final_result_for_test = asyncio.run(run_sentiment_analysis_for_symbol(test_symbol_input))
        
        print("\n--- Sentiment Analysis Result (Summary) ---")
        if final_result_for_test.get("error"):
            print(f"Error: {final_result_for_test['error']}")
        else:
            print(f"Symbol Analyzed: {final_result_for_test.get('symbol_analyzed')}")
            print(f"Timestamp UTC: {final_result_for_test.get('snapshot_timestamp_utc')}")
            print(f"Avg. Social Sentiment Score: {final_result_for_test.get('average_social_sentiment_score')}")
            print(f"Overall Social Sentiment Category: {final_result_for_test.get('overall_social_sentiment_category')}")
            
            fg_index = final_result_for_test.get('fear_greed_index')
            if fg_index and isinstance(fg_index, dict):
                print(f"Fear & Greed Index: {fg_index.get('value')} ({fg_index.get('value_classification')})")
            else:
                print(f"Fear & Greed Index: Data unavailable or in unexpected format ({fg_index})")

            print(f"DXY Strength Score: {final_result_for_test.get('dxy_strength_score', 'N/A')}")
            print(f"Total Social Posts Analyzed: {final_result_for_test.get('total_social_posts_analyzed')}")
            print(f"Macro News Articles Fetched: {len(final_result_for_test.get('macro_economic_news', []))}")
            # Full result is saved to a file, can be very large to print to console.
            # print("\nFull Result (JSON):")
            # print(json.dumps(final_result_for_test, indent=2))

    except Exception as e:
        logger.critical(f"An error occurred during the __main__ test run for {test_symbol_input}: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred in the main test execution: {e}")

    print("\n--- End of Test Run ---")