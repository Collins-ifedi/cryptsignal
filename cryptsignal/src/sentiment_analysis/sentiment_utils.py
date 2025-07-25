import os
import re
import sys
import logging
import json # Added for LLM response parsing
from pathlib import Path
import requests # Added for API calls
from typing import Dict, Any, Optional # Added Dict, Any, Optional

# Pydroid path (ensure this is correct for your environment)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT)not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import project's logger, fallback to basic logging
try:
    from utils.logger import get_logger
    logger = get_logger("SentimentUtils")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("SentimentUtils_Fallback")
    logger.warning("Custom logger not found, using basic logging for SentimentUtils.")

# Import the new config_loader
try:
    from utils.credentials_loader import load_credentials
except ImportError:
    logger.error("Could not import config_loader. LLM API key loading from file will fail.")
    load_credentials = None # type: ignore


# --- LLM Configuration ---
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "meta-llama/llama-4-maverick:" # As used in TradingAssistantNLPHandler
LLM_REQUEST_TIMEOUT = 45 # Seconds

# Global variable for API key to cache it after first load
_llm_api_key: Optional[str] = None
_credentials_loaded: bool = False # Flag to ensure credentials are only loaded once

def _get_openrouter_api_key() -> Optional[str]:
    """Retrieves the OpenRouter API key from environment variables, then from credentials.yaml."""
    global _llm_api_key, _credentials_loaded
    
    if _llm_api_key is None:
        # 1. Try environment variable first
        _llm_api_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
        if _llm_api_key:
            logger.info("OpenRouter API key loaded from environment variable for SentimentUtils.")
            return _llm_api_key

        # 2. If not in environment, try loading from credentials.yaml
        if not _credentials_loaded and load_credentials: # Check if loader was imported
            credentials = load_credentials() # Load the credentials file
            if credentials:
                _llm_api_key = credentials.get('openrouter', {}).get('api_key')
                if _llm_api_key:
                    logger.info("OpenRouter API key loaded from credentials.yaml for SentimentUtils.")
                else:
                    logger.warning("OpenRouter API key not found in credentials.yaml under 'openrouter: api_key'.")
            else:
                logger.warning("Failed to load credentials from file for SentimentUtils.")
            _credentials_loaded = True # Mark as attempted to load from file

    if _llm_api_key is None:
        logger.error(f"{OPENROUTER_API_KEY_ENV_VAR} not found in environment or credentials.yaml. LLM sentiment analysis will fail.")
    return _llm_api_key

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    # Basic cleaning: remove non-alphanumeric and non-essential punctuation, keep spaces.
    # This can be adjusted if the LLM handles raw text better.
    text = re.sub(r"[^\w\s\.,'!\?\$\%\@\#\&\*]", "", text) # Kept some common punctuation
    return text.strip()

def _extract_json_from_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from the LLM's response string.
    Handles cases where the JSON is embedded within other text.
    """
    if not response_text:
        return None
    
    # Regex to find JSON block, handles ```json ... ``` or just { ... }
    match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', response_text, re.DOTALL)
    if match:
        json_str = match.group(1) if match.group(1) else match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}. JSON string was: '{json_str}'")
            return None
    else:
        logger.warning(f"No JSON object found in LLM response: '{response_text[:200]}...'")
        return None

def _get_sentiment_from_llm(text_to_analyze: str, api_key: str, model: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    """
    Calls the OpenRouter LLM to get sentiment analysis for the given text.

    Returns:
        A dictionary with 'score', 'category', and optionally 'error'.
        Example: {"score": 0.75, "category": "bullish", "error": null}
                 {"score": 0.0, "category": "neutral", "error": "API call failed"}
    """
    default_error_response = {"score": 0.0, "category": "neutral", "error": "LLM sentiment analysis failed."}

    if not text_to_analyze:
        return {"score": 0.0, "category": "neutral", "error": "Input text was empty."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Optional: Replace with your actual site URL if deployed
        "X-Title": "CryptoSignal SentimentUtils", # Optional: For OpenRouter logging
    }

    system_prompt = (
        "You are an expert sentiment analysis AI. Your task is to analyze the sentiment of the provided text. "
        "Provide a numerical score between -1.0 (extremely bearish/negative) and 1.0 (extremely bullish/positive). "
        "Also, classify the sentiment into ONE of the following categories: 'bullish', 'mini bullish', 'neutral', 'mini bearish', 'bearish'. "
        "You MUST return your response ONLY as a valid JSON object with two keys: \"score\" (a float) and \"category\" (a string from the provided list). "
        "Do not add any explanations or conversational text outside of the JSON object."
    )
    user_prompt_template = (
        "Analyze the sentiment of the following text and provide your output strictly as a JSON object with \"score\" and \"category\" keys:\n\n"
        "Text: \"{text}\""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format(text=text_to_analyze)}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 150,  # Sufficient for a JSON response with score and category
        "temperature": 0.2, # Lower temperature for more deterministic classification
        "top_p": 0.9,
        # "response_format": {"type": "json_object"}, # If supported by specific model on OpenRouter
    }

    try:
        logger.debug(f"Sending sentiment request to LLM for text: '{text_to_analyze[:100]}...'")
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=LLM_REQUEST_TIMEOUT
        )
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)

        llm_response_data = response.json()
        
        if llm_response_data.get("choices") and llm_response_data["choices"][0].get("message"):
            generated_text_content = llm_response_data["choices"][0]["message"]["content"].strip()
            logger.debug(f"Raw LLM response content: {generated_text_content}")
            
            parsed_json = _extract_json_from_llm_response(generated_text_content)

            if parsed_json and isinstance(parsed_json.get("score"), (int, float)) and isinstance(parsed_json.get("category"), str):
                # Validate score range and category
                score = float(parsed_json["score"])
                category = parsed_json["category"].lower()
                valid_categories = ['bullish', 'mini bullish', 'neutral', 'mini bearish', 'bearish']

                if not (-1.0 <= score <= 1.0):
                    logger.warning(f"LLM returned score out of range ({score}). Clamping to [-1, 1].")
                    score = max(-1.0, min(1.0, score))
                
                if category not in valid_categories:
                    logger.warning(f"LLM returned an invalid category '{category}'. Defaulting to 'neutral'. Original LLM output: {generated_text_content}")
                    # Attempt to infer category from score if LLM fails category, or default
                    if score > 0.5: category = "bullish"
                    elif score > 0.1: category = "mini bullish"
                    elif score < -0.5: category = "bearish"
                    elif score < -0.1: category = "mini bearish"
                    else: category = "neutral"
                    
                return {"score": score, "category": category, "error": None}
            else:
                logger.error(f"LLM response did not contain valid 'score' and 'category' in expected JSON format. Raw content: {generated_text_content}")
                error_msg = "LLM response parsing failed or missing required fields."
                if parsed_json: # if JSON was parsed but fields were wrong
                     error_msg += f" Parsed JSON: {parsed_json}"
                return {**default_error_response, "error": error_msg}
        else:
            logger.error(f"Unexpected LLM API response structure: {llm_response_data}")
            return {**default_error_response, "error": "LLM API response structure invalid."}

    except requests.exceptions.Timeout:
        logger.error(f"LLM API request timed out after {LLM_REQUEST_TIMEOUT} seconds.")
        return {**default_error_response, "error": "LLM request timed out."}
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        error_detail = http_err.response.text[:200] if hasattr(http_err.response, 'text') else str(http_err)
        logger.error(f"HTTP error calling LLM API (Status {status_code}): {error_detail}", exc_info=True)
        return {**default_error_response, "error": f"LLM API HTTP error {status_code}: {error_detail}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network or Request error calling LLM API: {e}", exc_info=True)
        return {**default_error_response, "error": f"LLM API Request error: {str(e)[:100]}"}
    except Exception as e:
        logger.error(f"Unexpected error during LLM sentiment call: {e}", exc_info=True)
        return {**default_error_response, "error": f"Unexpected error: {str(e)[:100]}"}


def hybrid_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of the given text using Llama 4 Maverick via OpenRouter.

    Args:
        text: The input text to analyze.

    Returns:
        A dictionary containing the sentiment 'score' (float between -1.0 and 1.0),
        'category' (string: 'bullish', 'mini bullish', 'neutral', 'mini bearish', 'bearish'),
        and 'error' (string, null if no error).
    """
    api_key = _get_openrouter_api_key()
    if not api_key:
        logger.error("OpenRouter API key is not configured. Cannot perform LLM sentiment analysis.")
        return {"score": 0.0, "category": "neutral", "error": "API key not configured."}

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return {"score": 0.0, "category": "neutral", "error": "Input text was empty after cleaning."}

    # Max length check (optional, depends on typical text length and LLM limits)
    # Max context for Llama-4-Maverick is not explicitly stated as "free" model, but generally high.
    # For very long texts, consider truncation or summarization first.
    # For now, sending as is. Add truncation if needed:
    # MAX_TEXT_LENGTH = 2000 # Example
    # if len(cleaned_text) > MAX_TEXT_LENGTH:
    #    logger.warning(f"Text length ({len(cleaned_text)}) exceeds max ({MAX_TEXT_LENGTH}), truncating.")
    #    cleaned_text = cleaned_text[:MAX_TEXT_LENGTH]
        
    sentiment_result = _get_sentiment_from_llm(cleaned_text, api_key)
    
    return sentiment_result


# --- Scraper Imports (Kept for SentimentAggregator if still used) ---
# These are not directly used by hybrid_sentiment anymore but might be by SentimentAggregator
try:
    from src.sentiment_analysis.scrapers.reddit_scraper import RedditScraper
    from src.sentiment_analysis.scrapers.twitter_scraper import TwitterScraper
    from src.sentiment_analysis.scrapers.crypto_panic_news_scraper import CryptoPanicNewsScraper
except ImportError as e:
    logger.warning(f"Failed to import one or more scrapers for SentimentAggregator: {e}. SentimentAggregator might not work as expected.")
    RedditScraper, TwitterScraper, CryptoPanicNewsScraper = None, None, None


class SentimentAggregator:
    def __init__(self, config_path: Optional[str] = None): # Made config_path optional
        self.reddit_scraper = None
        self.twitter_scraper = None
        self.news_scraper = None
        
        # Configuration for scrapers is often handled by the scrapers themselves looking for a default path or env vars.
        # If config_path is provided and scrapers expect it, it can be passed.
        # For now, assuming scrapers can initialize without a path or use their own defaults.
        # This part might need adjustment based on how RedditScraper etc. are designed.
        # If they strictly require config_path, this __init__ needs error handling or a mandatory path.

        try:
            if RedditScraper:
                self.reddit_scraper = RedditScraper(config_path=config_path) if config_path else RedditScraper()
            if TwitterScraper:
                self.twitter_scraper = TwitterScraper(config_path=config_path) if config_path else TwitterScraper()
            if CryptoPanicNewsScraper:
                self.news_scraper = CryptoPanicNewsScraper(config_path=config_path) if config_path else CryptoPanicNewsScraper()
            logger.info("SentimentAggregator initialized (scrapers might be None if imports failed or config missing).")
        except Exception as e:
            logger.exception(f"Failed to initialize SentimentAggregator scrapers with config_path='{config_path}'. Scrapers might be unavailable.")
            # Individual scrapers will remain None if their init fails

    def get_sentiment_data(self, coin_name: str, limit: int = 10) -> dict:
        logger.info(f"Fetching sentiment data for: {coin_name} with limit: {limit}")
        results = {"coin": coin_name, "reddit": [], "twitter": [], "news": [], "error": None}
        
        try:
            if self.reddit_scraper:
                reddit_data = self.reddit_scraper.scrape(coin_name=coin_name, max_results=limit)
                logger.debug(f"Reddit Data for {coin_name}: {reddit_data}")
                results["reddit"] = self._analyze_items(reddit_data, "content")
            else: logger.warning("Reddit scraper not available for SentimentAggregator.")
            
            if self.twitter_scraper:
                twitter_data = self.twitter_scraper.scrape(coin_name=coin_name, max_results=limit)
                logger.debug(f"Twitter Data for {coin_name}: {twitter_data}")
                results["twitter"] = self._analyze_items(twitter_data, "content")
            else: logger.warning("Twitter scraper not available for SentimentAggregator.")

            if self.news_scraper:
                news_data = self.news_scraper.fetch_news(coin_name=coin_name, limit=limit) # Assuming 'limit' param name
                logger.debug(f"News Data for {coin_name}: {news_data}")
                results["news"] = self._analyze_items(news_data, "title") 
            else: logger.warning("News scraper not available for SentimentAggregator.")

        except Exception as e:
            logger.exception(f"Error during sentiment aggregation for {coin_name}: {e}")
            results["error"] = str(e)
            
        return results

    def _analyze_items(self, items: list, field: str) -> list:
        analyzed = []
        if not items:
            return analyzed
            
        for item_idx, item in enumerate(items):
            text_to_analyze = ""
            if isinstance(item, dict):
                 text_to_analyze = item.get(field, "")
            elif isinstance(item, str):
                 text_to_analyze = item
            
            # hybrid_sentiment now returns a dict: {"score": float, "category": str, "error": Optional[str]}
            sentiment_details = hybrid_sentiment(text_to_analyze)
            
            if isinstance(item, dict):
                # Store the entire sentiment details dictionary
                item["sentiment_analysis"] = sentiment_details 
                # For backward compatibility, also store score if needed, but prefer the full dict
                item["sentiment_score"] = sentiment_details.get("score", 0.0) 
                item["sentiment_category"] = sentiment_details.get("category", "neutral")
                analyzed.append(item)
            else: # If item was a string (less likely for current scrapers)
                analyzed.append({
                    "original_text": item, 
                    "sentiment_analysis": sentiment_details,
                    "sentiment_score": sentiment_details.get("score", 0.0),
                    "sentiment_category": sentiment_details.get("category", "neutral"),
                    "field_used": field
                })
            logger.debug(f"Analyzed item {item_idx + 1} for field '{field}'. Sentiment: {sentiment_details}")
                
        return analyzed

# Example usage
if __name__ == "__main__":
    # --- IMPORTANT ---
    # For this example to work, you MUST set the OPENROUTER_API_KEY environment variable.
    # Example (in your terminal before running the script):
    # export OPENROUTER_API_KEY="your_actual_openrouter_api_key_here"
    # -----------------

    api_key_is_set = _get_openrouter_api_key() # This also attempts to load it

    if not api_key_is_set:
        print("\n" + "="*60)
        print("ERROR: OPENROUTER_API_KEY could not be loaded from environment or credentials.yaml.")
        print("Please ensure it's set in either place before running this example.")
        print("Example (environment): export OPENROUTER_API_KEY='your_key_here'")
        print("Example (credentials.yaml): openrouter: api_key: 'your_key_here'")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("OpenRouter API Key found. Proceeding with example.")
        print("="*60 + "\n")

        test_texts = [
            ("This is a great day for Bitcoin, price is soaring!", "Expected: bullish"),
            ("I think BTC might go up a bit more soon.", "Expected: mini bullish"),
            ("The market is stable, nothing much happening with Ethereum.", "Expected: neutral"),
            ("Feeling a bit worried about the short-term dip in Solana.", "Expected: mini bearish"),
            ("Complete disaster for altcoins, everything is crashing hard!", "Expected: bearish"),
            ("Crypto is the future, to the moon! 🚀🚀", "Expected: bullish"),
            ("Not sure what to think about the market right now.", "Expected: neutral"),
            ("", "Expected: neutral (empty string)"),
            ("This is an extremely positive outlook for the entire crypto space, expecting massive gains across the board!", "Expected: bullish"),
            ("Slightly negative sentiment observed due to recent FUD, but fundamentals remain okay.", "Expected: mini bearish")
        ]

        for i, (text, expected) in enumerate(test_texts):
            print(f"--- Test Case {i+1} ---")
            print(f"Input Text: \"{text}\" (Expected sentiment hint: {expected})")
            result = hybrid_sentiment(text)
            print(f"Sentiment Result: Score = {result.get('score')}, Category = '{result.get('category')}', Error = {result.get('error')}")
            print("-" * 20 + "\n")

        # Example with SentimentAggregator (will be limited if scrapers are not fully configured/working)
        print("\n--- SentimentAggregator Example (limited functionality without full scraper setup) ---")
        # Ensure the config path is correct for your environment if scrapers need it.
        # For this test, we'll assume scrapers might not initialize fully if their own configs are missing.
        # The main test here is that `hybrid_sentiment` is called correctly.
        # Default path for config if scrapers need it, they might have their own defaults too.
        aggregator_config_path = Path(PROJECT_ROOT)/ "config"/ "credentials.yaml"
        if not os.path.exists(aggregator_config_path):
            logger.warning(f"Aggregator config file not found at: {aggregator_config_path}. Scrapers in SentimentAggregator might use defaults or fail if they strictly require it.")
            aggregator_config_path = None # Pass None if file not found to let scrapers handle it

        try:
            # Initialize aggregator, scrapers might be None if their imports/init failed earlier
            aggregator = SentimentAggregator(config_path=aggregator_config_path) 
            
            # We can't actually scrape without full setup, so we'll test _analyze_items with mock data
            print("Testing SentimentAggregator._analyze_items with mock data...")
            mock_news_items = [
                {"title": "Bitcoin reaches new all-time high!", "source": "mock_news"},
                {"title": "Market shows slight downturn, investors cautious.", "source": "mock_news"}
            ]
            analyzed_mock_news = aggregator._analyze_items(mock_news_items, "title")
            print("Analyzed Mock News Items:")
            for item_news in analyzed_mock_news:
                print(f"  Title: {item_news.get('title')}, Sentiment: Score={item_news.get('sentiment_score')}, Category='{item_news.get('sentiment_category')}'")

        except Exception as e:
            logger.exception(f"An error occurred during the SentimentAggregator test: {e}")