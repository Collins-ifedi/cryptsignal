import sys
import os
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import re # Make sure 're' is imported if used in _extract_json_from_llm_response

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- PROJECT IMPORTS ---
try:
    from utils.logger import get_logger
    logger = get_logger("MacroEconomyIndicator")
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MacroEconomyIndicator_Fallback")
    logger.error(f"Custom logger import error: {e}. Using fallback basic logging.")

# Import the new config_loader
try:
    from utils.credentials_loader import load_credentials
except ImportError:
    logger.error("Could not import config_loader. LLM API key loading from file will fail.")
    load_credentials = None # type: ignore

# --- LLM Configuration ---
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "meta-llama/llama-4-maverick"
LLM_REQUEST_TIMEOUT = 60 # Seconds

# Global variable for API key to cache it after first load
_llm_api_key: Optional[str] = None
_credentials_loaded: bool = False # Flag to ensure credentials are only loaded once

def _get_openrouter_api_key() -> Optional[str]:
    """
    Retrieves the OpenRouter API key from environment variables, then from credentials.yaml.
    """
    global _llm_api_key, _credentials_loaded
    
    if _llm_api_key is None:
        # 1. Try environment variable first
        _llm_api_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
        if _llm_api_key:
            logger.info("OpenRouter API key loaded from environment variable for MacroEconomyIndicator.")
            return _llm_api_key

        # 2. If not in environment, try loading from credentials.yaml
        if not _credentials_loaded and load_credentials: # Check if loader was imported
            credentials = load_credentials() # Load the credentials file
            if credentials:
                _llm_api_key = credentials.get('openrouter', {}).get('api_key')
                if _llm_api_key:
                    logger.info("OpenRouter API key loaded from credentials.yaml for MacroEconomyIndicator.")
                else:
                    logger.warning("OpenRouter API key not found in credentials.yaml under 'openrouter: api_key'.")
            else:
                logger.warning("Failed to load credentials from file for MacroEconomyIndicator.")
            _credentials_loaded = True # Mark as attempted to load from file

    if _llm_api_key is None:
        logger.error(f"{OPENROUTER_API_KEY_ENV_VAR} not found in environment or credentials.yaml. LLM analysis will fail.")
    return _llm_api_key

def _extract_json_from_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extracts a JSON object from the LLM's response string."""
    if not response_text:
        return None
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

def _analyze_indicator_with_llm(indicator_title: str, coin_name: str, api_key: str, model: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    """
    Analyzes a macroeconomic indicator using an LLM.

    Returns:
        A dictionary with "text" (analysis), "sentiment" (score), "category", and "error".
    """
    default_error_response = {
        "text": f"LLM analysis unavailable for {indicator_title}.",
        "sentiment": 0.0,
        "category": "neutral",
        "error": "LLM analysis failed or API key missing."
    }
    if not api_key:
        logger.error("OpenRouter API key is not available for LLM analysis.")
        return default_error_response

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Optional
        "X-Title": "CryptoSignal MacroIndicator", # Optional
    }

    system_prompt = (
        "You are an expert macroeconomic analyst AI. Your task is to analyze the provided macroeconomic indicator "
        "and its potential implications for the specified cryptocurrency (e.g., Bitcoin)."
    )
    user_prompt_template = (
        "Analyze the macroeconomic indicator: \"{indicator_title}\" and its potential implications for {coin_name}. "
        "Provide a concise text analysis (around 50-100 words). "
        "Also, provide a sentiment score reflecting this indicator's typical impact on {coin_name} (ranging from -1.0 for very negative impact to 1.0 for very positive impact), "
        "and a sentiment category (choose one from: 'positive', 'mini positive', 'neutral', 'mini negative', 'negative').\n"
        "Return your response ONLY as a valid JSON object with three keys: \"text_analysis\" (string), \"sentiment_score\" (float), and \"sentiment_category\" (string)."
        "\nExample for 'Interest Rate' and 'Bitcoin':\n"
        "{{ \"text_analysis\": \"Higher interest rates often make borrowing more expensive, potentially reducing liquidity in speculative markets like Bitcoin. It can also strengthen traditional currencies, making crypto relatively less attractive as an alternative store of value in the short term.\", \"sentiment_score\": -0.4, \"sentiment_category\": \"mini negative\" }}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format(indicator_title=indicator_title, coin_name=coin_name)}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 250, # Adjusted for potentially longer analysis text
        "temperature": 0.5, # Balanced temperature
    }

    try:
        logger.debug(f"Sending LLM request for indicator: '{indicator_title}', context: '{coin_name}'")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        llm_response_data = response.json()

        if llm_response_data.get("choices") and llm_response_data["choices"][0].get("message"):
            generated_text_content = llm_response_data["choices"][0]["message"]["content"].strip()
            logger.debug(f"Raw LLM response for '{indicator_title}': {generated_text_content}")
            
            parsed_json = _extract_json_from_llm_response(generated_text_content)

            if parsed_json and \
               isinstance(parsed_json.get("text_analysis"), str) and \
               isinstance(parsed_json.get("sentiment_score"), (int, float)) and \
               isinstance(parsed_json.get("sentiment_category"), str):
                
                score = float(parsed_json["sentiment_score"])
                category = parsed_json["sentiment_category"].lower()
                valid_categories = ['positive', 'mini positive', 'neutral', 'mini negative', 'negative']

                if not (-1.0 <= score <= 1.0):
                    logger.warning(f"LLM returned score {score} out of range for '{indicator_title}'. Clamping.")
                    score = max(-1.0, min(1.0, score))
                
                if category not in valid_categories:
                    logger.warning(f"LLM returned invalid category '{category}' for '{indicator_title}'. Defaulting or inferring. Raw: {generated_text_content}")
                    # Simple inference or default to neutral
                    if score > 0.5: category = "positive"
                    elif score > 0.1: category = "mini positive"
                    elif score < -0.5: category = "negative"
                    elif score < -0.1: category = "mini negative"
                    else: category = "neutral"

                return {
                    "title": indicator_title, # Added to match analyze_macro_indicators expected dict structure
                    "text": parsed_json["text_analysis"],
                    "sentiment": score,
                    "category": category,
                    "error": None
                }
            else:
                logger.error(f"LLM response for '{indicator_title}' did not contain valid JSON structure or fields. Raw content: {generated_text_content}")
                return {**default_error_response, "error": f"LLM response parsing failed for {indicator_title}. Content: {generated_text_content[:100]}...", "text": f"LLM response parsing failed for {indicator_title}."}
        else:
            logger.error(f"Unexpected LLM API response structure for '{indicator_title}': {llm_response_data}")
            return {**default_error_response, "error": f"LLM API response structure invalid for {indicator_title}."}

    except requests.exceptions.Timeout:
        logger.error(f"LLM API request timed out for '{indicator_title}'.")
        return {**default_error_response, "error": f"LLM request timed out for {indicator_title}."}
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error for '{indicator_title}' (Status {http_err.response.status_code}): {http_err.response.text[:200]}", exc_info=False)
        return {**default_error_response, "error": f"LLM API HTTP error for {indicator_title}."}
    except Exception as e:
        logger.error(f"Unexpected error during LLM call for '{indicator_title}': {e}", exc_info=True)
        return {**default_error_response, "error": f"Unexpected error for {indicator_title}: {str(e)}"}

# --- CONSTANTS ---
MACRO_INDICATORS: List[str] = [
    "GDP Growth Rate",
    "Unemployment Rate",
    "CPI Inflation",
    "Interest Rate",
    "Consumer Confidence Index",
    "Retail Sales",
    "ISM Manufacturing PMI",
    "Initial Jobless Claims",
    "Non-Farm Payrolls"
]
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "macro")

def analyze_macro_indicators(indicators: List[str], coin_name: str = "Bitcoin") -> List[Dict[str, Any]]:
    """
    Analyzes a list of macroeconomic indicators using an LLM.

    Args:
        indicators: A list of macro indicator titles (strings) to analyze.
        coin_name: The name of the coin for context in analysis (default is "Bitcoin").

    Returns:
        A list of dictionaries, where each dictionary contains the analysis
        (title, text, sentiment score, sentiment category, error) for an indicator.
    """
    analysis_results: List[Dict[str, Any]] = []
    api_key = _get_openrouter_api_key()

    if not api_key:
        logger.critical("OpenRouter API Key not found. Cannot perform LLM-based macro analysis.")
        # Return error state for all indicators if API key is missing
        for title in indicators:
            analysis_results.append({
                "title": title,
                "text": "Error: LLM analysis service unavailable (API Key missing).",
                "sentiment": 0.0,
                "category": "neutral",
                "error": "API Key missing for LLM service"
            })
        return analysis_results

    for title in indicators:
        try:
            logger.info(f"Fetching LLM analysis for macro indicator: '{title}' with context '{coin_name}'")
            # Call the new LLM analysis function
            result = _analyze_indicator_with_llm(title, coin_name, api_key)
            
            analysis_results.append({
                "title": title,
                "text": result.get("text", f"Analysis for '{title}' failed."),
                "sentiment": result.get("sentiment", 0.0), # Numerical score
                "category": result.get("category", "neutral"), # Sentiment category
                "error": result.get("error") # Include error message if any
            })
        except Exception as e: # Catch any unexpected error during the loop for a specific indicator
            logger.error(f"Failed to process or analyze LLM result for '{title}': {e}", exc_info=True)
            analysis_results.append({
                "title": title,
                "text": f"Error analyzing '{title}' with LLM: {str(e)}",
                "sentiment": 0.0,
                "category": "neutral",
                "error": str(e)
            })
    return analysis_results

def save_analysis_results(results: List[Dict[str, Any]], directory: str = DATA_DIR) -> None:
    """
    Saves the analysis results to a JSON file in the specified directory.
    The filename will include a timestamp.
    """
    if not results:
        logger.info("No analysis results to save.")
        return

    try:
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(directory, f"macro_analysis_llm_{timestamp}.json")
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved LLM-based macroeconomic analysis results to: {file_path}")
    except IOError as e:
        logger.exception(f"IOError saving analysis results to {file_path}: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while saving analysis results: {e}")


def main() -> None:
    """
    Main function to run the macroeconomic analysis using LLM and save the results.
    """
    # --- IMPORTANT ---
    # For this script to work with LLM analysis, you MUST set the
    # OPENROUTER_API_KEY environment variable.
    # Example (in your terminal before running):
    # export OPENROUTER_API_KEY="your_actual_openrouter_api_key_here"
    # -----------------
    if not _get_openrouter_api_key(): # Initial check and attempt to load
        print("CRITICAL: OPENROUTER_API_KEY environment variable is not set.")
        print("LLM-based macroeconomic analysis cannot proceed without the API key.")
        print("Please set the environment variable or ensure it's in credentials.yaml and try again.")
        return

    try:
        logger.info("Starting LLM-based macroeconomic analysis...")
        results = analyze_macro_indicators(MACRO_INDICATORS, coin_name="Bitcoin")
        
        if results:
            logger.info("--- LLM-Based Macroeconomic Analysis Summary ---")
            for r in results:
                title = r.get('title', 'N/A')
                text_summary = r.get('text', 'No text available')
                sentiment_score = r.get('sentiment', 'N/A')
                sentiment_category = r.get('category', 'N/A')
                error_info = f" (Error: {r['error']})" if r.get('error') else ""
                
                text_preview = (text_summary[:100] + '...') if len(text_summary) > 100 else text_summary
                
                print(f"Indicator: {title}\n  Sentiment Score: {sentiment_score}\n  Sentiment Category: {sentiment_category}\n  Summary: {text_preview}{error_info}\n---")
            save_analysis_results(results)
        else:
            logger.warning("No LLM-based analysis results were generated.")
            
    except Exception as e:
        logger.critical(f"A fatal error occurred in the main execution block: {e}", exc_info=True)
    finally:
        logger.info("LLM-based macroeconomic analysis process finished.")

if __name__ == "__main__":
    main()