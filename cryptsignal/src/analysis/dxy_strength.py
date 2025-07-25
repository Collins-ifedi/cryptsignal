import requests
import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any # Added for type hinting

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("DXYStrengthFetcher")

CREDENTIALS_PATH = Path(PROJECT_ROOT)/ "config"/ "credentials.yaml"
OUTPUT_DIR = Path(PROJECT_ROOT)/ "data"/ "macro"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dxy_strength_history.json")
MAX_DXY_HISTORY_ENTRIES = 365 * 5 # Approx 5 years of daily data, set to 0 or None for unlimited

def get_alphavantage_key(file_path: str = CREDENTIALS_PATH) -> Optional[str]:
    """Loads Alpha Vantage API key from the credentials YAML file."""
    try:
        with open(file_path, "r") as file:
            creds = yaml.safe_load(file)
        api_key = creds.get("alphavantage_key")
        if not api_key:
            logger.error(f"Alpha Vantage API key not found or empty in {file_path} under 'alphavantage_key'.")
            return None
        return api_key
    except FileNotFoundError:
        logger.error(f"Credentials file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing credentials file {file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading API key from {file_path}: {e}")
    return None

def fetch_dxy_strength() -> Optional[float]:
    """
    Fetches the latest UUP ETF closing price (as a proxy for DXY strength)
    from Alpha Vantage, updates a history file, and returns the closing price.
    Returns None if fetching or processing fails.
    """
    api_key = get_alphavantage_key()
    if not api_key:
        logger.error("Cannot fetch DXY strength: Alpha Vantage API key is missing.")
        return None

    # Use UUP ETF to approximate DXY movement
    # API URL uses the provided API key
    dxy_api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=UUP&apikey={api_key}&outputsize=compact"
    logger.info(f"Fetching DXY (UUP proxy) strength from Alpha Vantage: {dxy_api_url.replace(api_key, 'REDACTED_API_KEY')}")

    try:
        response = requests.get(dxy_api_url, timeout=15) # Increased timeout slightly
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        data = response.json()

        if "Note" in data: # Handles API rate limit messages
            logger.error(f"Alpha Vantage API Note (likely rate limit or invalid call): {data['Note']}")
            return None
        if "Error Message" in data: # Handles other API error messages
             logger.error(f"Alpha Vantage API Error Message: {data['Error Message']}")
             return None
        if "Time Series (Daily)" not in data or not data["Time Series (Daily)"]:
            logger.error(f"Unexpected response structure or empty time series from Alpha Vantage. Data: {data}")
            return None

        time_series = data["Time Series (Daily)"]
        latest_data_date = sorted(time_series.keys())[-1] # Get the most recent date string
        latest_day_data = time_series[latest_data_date]
        
        if "4. close" not in latest_day_data:
            logger.error(f"Could not find '4. close' in latest data for {latest_data_date}. Data: {latest_day_data}")
            return None
            
        uup_close_price_str = latest_day_data["4. close"]
        uup_close_price = float(uup_close_price_str)

        new_record = {
            "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_date": latest_data_date,
            "uup_close_price": uup_close_price
        }

        # Manage history file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        history_data: List[Dict[str, Any]] = []
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r') as file:
                    history_data = json.load(file)
                if not isinstance(history_data, list):
                    logger.warning(f"History file {OUTPUT_FILE} was not a list. Resetting.")
                    history_data = []
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {OUTPUT_FILE}. Resetting history.")
                history_data = []
            except Exception as e: # Catch other potential errors during history load
                logger.warning(f"Failed to load history from {OUTPUT_FILE}: {e}. Resetting.")
                history_data = []
        
        history_map: Dict[str, Dict[str, Any]] = {
            record.get("data_date"): record 
            for record in history_data 
            if isinstance(record, dict) and "data_date" in record
        }

        log_msg_prefix = f"Data for date {latest_data_date} (UUP Close: {uup_close_price}):"
        if latest_data_date in history_map and \
           abs(history_map[latest_data_date].get("uup_close_price", float('nan')) - uup_close_price) < 1e-9 : # Comparing floats
            logger.info(f"{log_msg_prefix} already exists and is unchanged. History file not modified with new entry for this date.")
            # Optionally update fetch_timestamp if desired even if data is same:
            # history_map[latest_data_date]["fetch_timestamp"] = new_record["fetch_timestamp"]
        else:
            if latest_data_date in history_map:
                logger.info(f"{log_msg_prefix} updating existing entry.")
            else:
                logger.info(f"{log_msg_prefix} adding new entry.")
            history_map[latest_data_date] = new_record

        updated_history = sorted(list(history_map.values()), key=lambda x: x.get("data_date", ""), reverse=True)

        if MAX_DXY_HISTORY_ENTRIES and MAX_DXY_HISTORY_ENTRIES > 0 and len(updated_history) > MAX_DXY_HISTORY_ENTRIES:
            updated_history = updated_history[:MAX_DXY_HISTORY_ENTRIES]
            logger.info(f"DXY history truncated to the latest {MAX_DXY_HISTORY_ENTRIES} entries.")

        try:
            with open(OUTPUT_FILE, 'w') as file:
                json.dump(updated_history, file, indent=2)
            logger.info(f"Saved {len(updated_history)} entries to DXY strength history: {OUTPUT_FILE}")
        except IOError as e:
            logger.error(f"Failed to write DXY history to {OUTPUT_FILE}: {e}")
            # Decide if this error should prevent returning the fetched value. For now, it won't.

        logger.info(f"Successfully fetched UUP (DXY proxy) data: {uup_close_price} on {latest_data_date}")
        return uup_close_price

    except requests.exceptions.Timeout:
        logger.error("Timeout error fetching DXY strength from Alpha Vantage.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching DXY strength: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching DXY strength: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from Alpha Vantage for DXY: {e}")
    except (ValueError, KeyError) as e: # Catch errors from data processing (e.g. float conversion, missing keys)
        logger.error(f"Error processing data for DXY strength: {e}")
    except Exception as e: # Catch-all for any other unexpected errors
        logger.error(f"An unexpected error occurred while fetching DXY strength: {e}", exc_info=True)
    
    return None # Return None on any failure

def main():
    logger.info("--- DXY Strength Fetcher ---")
    dxy_value = fetch_dxy_strength()
    if dxy_value is not None:
        logger.info(f"Latest DXY (UUP proxy) strength value: {dxy_value}")
    else:
        logger.warning("Failed to retrieve DXY (UUP proxy) strength value.")
    logger.info("--- DXY Strength Fetcher Finished ---")

if __name__ == "__main__":
    main()