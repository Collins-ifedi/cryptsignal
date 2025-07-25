import os
import json
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any # Added for type hinting

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("FearGreedIndex")

# Define constants for file paths and cache duration
BASE_DATA_DIR = Path(PROJECT_ROOT)/ "data"/ "macro"
CACHE_FILE = os.path.join(BASE_DATA_DIR, "fear_greed_cache.json")
HISTORY_FILE = os.path.join(BASE_DATA_DIR, "fear_greed_history.json")
CACHE_DURATION = timedelta(minutes=30)
MAX_HISTORY_ENTRIES = 365 # Optional: Limit the number of entries in the history file (e.g., one year of daily data)

def load_cached_data() -> Optional[List[Dict[str, Any]]]:
    """Loads Fear & Greed index data from cache if available and not stale."""
    if not os.path.exists(CACHE_FILE):
        logger.debug(f"Cache file not found: {CACHE_FILE}")
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        # Ensure fetched_at and index_data are present
        if "fetched_at" in data and "index_data" in data:
            fetched_at = datetime.fromisoformat(data["fetched_at"])
            if fetched_at > datetime.utcnow() - CACHE_DURATION:
                logger.debug("Returning Fear & Greed data from fresh cache.")
                return data["index_data"]
            else:
                logger.debug("Fear & Greed cache is stale.")
        else:
            logger.warning(f"Cache file {CACHE_FILE} is malformed or missing required keys.")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to load or parse Fear & Greed cache from {CACHE_FILE}: {e}")
    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
        logger.warning(f"Cache file {CACHE_FILE} disappeared during read attempt.")
    return None

def save_data_to_cache(index_data: List[Dict[str, Any]]) -> None:
    """Saves Fear & Greed index data to a cache file."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        cache_content = {
            "fetched_at": datetime.utcnow().isoformat(),
            "index_data": index_data
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_content, f, indent=2)
        logger.debug(f"Fear & Greed data saved to cache: {CACHE_FILE}")
    except IOError as e:
        logger.warning(f"Failed to save Fear & Greed cache to {CACHE_FILE}: {e}")

def save_history(new_index_data: List[Dict[str, Any]]) -> None:
    """Loads existing history, appends new non-duplicate entries, sorts, limits, and saves it back."""
    if not new_index_data:
        logger.info("No new data provided to save_history.")
        return

    current_history: List[Dict[str, Any]] = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                current_history = json.load(f)
            if not isinstance(current_history, list):
                logger.warning(f"History file {HISTORY_FILE} did not contain a list. Resetting history.")
                current_history = []
        except json.JSONDecodeError:
            logger.warning(f"Could not decode existing history from {HISTORY_FILE}. Resetting history.")
            current_history = []
        except Exception as e: # Catch any other unexpected error during load
            logger.warning(f"Failed to load history from {HISTORY_FILE} due to an unexpected error: {e}. Resetting history.")
            current_history = []

    # Use a dictionary for efficient addition/updating of entries by unique timestamp_unix
    history_map: Dict[int, Dict[str, Any]] = {
        entry["timestamp_unix"]: entry 
        for entry in current_history 
        if isinstance(entry, dict) and "timestamp_unix" in entry
    }

    updated_count = 0
    added_count = 0
    for entry in new_index_data:
        if isinstance(entry, dict) and "timestamp_unix" in entry:
            ts = entry["timestamp_unix"]
            if ts not in history_map:
                added_count += 1
            elif history_map[ts] != entry: # Content changed for existing timestamp
                updated_count +=1
            history_map[ts] = entry # Add new or update existing
        else:
            logger.warning(f"Skipping malformed entry during history update: {entry}")
    
    if added_count > 0 or updated_count > 0:
        logger.info(f"History update: {added_count} entries added, {updated_count} entries updated.")
    else:
        logger.info("No new or changed entries to update in history.")

    # Convert map back to list and sort (most recent first by timestamp_unix)
    updated_history = sorted(list(history_map.values()), key=lambda x: x.get("timestamp_unix", 0), reverse=True)
    
    # Limit the number of entries in history if MAX_HISTORY_ENTRIES is set
    if MAX_HISTORY_ENTRIES > 0 and len(updated_history) > MAX_HISTORY_ENTRIES:
        updated_history = updated_history[:MAX_HISTORY_ENTRIES]
        logger.info(f"History truncated to the latest {MAX_HISTORY_ENTRIES} entries.")

    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(updated_history, f, indent=2)
        logger.info(f"Saved {len(updated_history)} entries to Fear & Greed history: {HISTORY_FILE}")
    except IOError as e:
        logger.error(f"Error saving Fear & Greed history to {HISTORY_FILE}: {e}")


def fetch_historical_index(limit: int = 30) -> List[Dict[str, Any]]:
    """Fetches historical Fear & Greed Index data from alternative.me API."""
    try:
        # The API endpoint for limit=0 or limit=1 gives current, for limit > 1 gives historical days.
        # The API sorts latest first by default.
        api_url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        logger.debug(f"Fetching Fear & Greed Index from: {api_url}")
        response = requests.get(api_url, timeout=10) # Standard 10-second timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        
        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            logger.error(f"Unexpected response structure from Fear & Greed API. Full response: {data}")
            return []

        historical_data: List[Dict[str, Any]] = []
        for entry in data["data"]:
            try:
                # Validate and convert data types
                ts_unix = int(entry["timestamp"])
                historical_data.append({
                    "value": int(entry["value"]),
                    "classification": str(entry["value_classification"]),
                    "timestamp_unix": ts_unix,
                    "datetime": datetime.utcfromtimestamp(ts_unix).isoformat() # Convert UNIX timestamp to ISO 8601
                })
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed entry from API data: {entry}. Error: {e}")
                continue # Skip this entry and proceed with the next
        
        logger.info(f"Successfully fetched {len(historical_data)} historical Fear & Greed Index entries.")
        return historical_data

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching historical Fear & Greed Index from {api_url}.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching historical Fear & Greed Index: {e.status_code} {e.response.text}")
    except requests.exceptions.RequestException as e: # Catch other requests-related errors
        logger.error(f"Network error fetching historical Fear & Greed Index: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from Fear & Greed API: {e}")
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in fetch_historical_index: {e}", exc_info=True) # exc_info for traceback
    return [] # Return empty list on any failure

def get_fear_greed_index(limit: int = 1, use_cache: bool = True, save_to_history_file: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieves the Fear & Greed Index.
    Uses cache for single entry requests (limit=1) if use_cache is True.
    Saves fetched data to a historical log if save_to_history_file is True.
    """
    logger.info(f"Getting Fear & Greed Index: limit={limit}, use_cache={use_cache}, save_to_history={save_to_history_file}")
    
    if use_cache and limit == 1:
        cached_data = load_cached_data()
        if cached_data is not None: # load_cached_data returns None or List
            # Ensure cache returns a list of the correct length
            return cached_data[:limit] if isinstance(cached_data, list) else []


    # Fetch new data if cache not used, not usable, or for limits > 1
    index_data = fetch_historical_index(limit=limit)

    if index_data: # Only save if data was actually fetched successfully
        if limit == 1: # Always cache the latest single entry if fetched for limit=1 request
            save_data_to_cache(index_data[:1]) # Ensure only the relevant entry (or entries if limit was small)
        
        if save_to_history_file:
            save_history(index_data) # Save all fetched data to history
    else:
        logger.warning("No Fear & Greed data fetched; cache and history were not updated with new data from this fetch.")
            
    return index_data

# Example usage / Test run
if __name__ == "__main__":
    logger.info("--- Starting Fear & Greed Index Update Script (Main Test) ---")
    
    # Test 1: Fetch a larger set for history (simulates a batch update)
    logger.info("Test 1: Fetching 30 days of data for history...")
    fetched_data_batch = get_fear_greed_index(limit=30, use_cache=False, save_to_history_file=True)
    if fetched_data_batch:
        logger.info(f"Test 1: Successfully fetched {len(fetched_data_batch)} entries.")
        # logger.info(f"Test 1: Latest entry from batch: {fetched_data_batch[0] if fetched_data_batch else 'N/A'}")
    else:
        logger.warning("Test 1: No data fetched for batch update.")
    logger.info("-" * 30)

    # Test 2: Simulate a call for the latest value (e.g., from another script, expecting cache usage)
    logger.info("Test 2: Simulating a call for the latest value (limit=1, use_cache=True)...")
    latest_value_data = get_fear_greed_index(limit=1, use_cache=True, save_to_history_file=True)
    if latest_value_data:
        logger.info(f"Test 2: Latest value from get_fear_greed_index (limit=1): {latest_value_data[0]}")
    else:
        logger.warning("Test 2: Could not retrieve latest Fear & Greed Index value.")
    logger.info("-" * 30)

    # Test 3: Call again for latest value, should ideally hit cache if within CACHE_DURATION
    logger.info("Test 3: Calling for latest value again (should hit cache if recent)...")
    latest_value_data_cached = get_fear_greed_index(limit=1, use_cache=True, save_to_history_file=False) # Not saving to history this time
    if latest_value_data_cached:
        logger.info(f"Test 3: Latest value (cached?): {latest_value_data_cached[0]}")
    else:
        logger.warning("Test 3: Could not retrieve latest Fear & Greed Index value (cached attempt).")
    
    logger.info("--- Fear & Greed Index Script Test Finished ---")