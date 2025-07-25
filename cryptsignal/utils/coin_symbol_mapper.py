# coin_symbol_mapper.py
import requests
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

# --- Basic Configuration ---
# Use a more robust way to define project root if this file moves
# For now, we assume this file is in a `utils` subdirectory of `src`.
try:
    # This assumes the script is in a subdirectory of the project root.
    # Adjust the number of .parent calls if the structure is different.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., some interactive shells)
    PROJECT_ROOT = Path(os.getcwd())

# Define cache directory within the project structure
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "coin_gecko_map.json"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/list?include_platform=false"
CACHE_TTL_SECONDS = 24 * 3600  # 24 hours

# --- Logger Setup ---
# Attempt to use a centralized logger if available, otherwise configure a basic one.
try:
    from utils.logger import get_logger
    logger = get_logger("CoinSymbolMapper")
except (ImportError, ModuleNotFoundError):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CoinSymbolMapper_Fallback")

class CoinMapper:
    """
    A robust class to handle fetching, caching, and mapping cryptocurrency
    identifiers (names, symbols, IDs) from the CoinGecko API.
    """
    def __init__(self, cache_file: Path = CACHE_FILE, ttl: int = CACHE_TTL_SECONDS):
        """
        Initializes the CoinMapper instance.

        Args:
            cache_file (Path): The path to the cache file.
            ttl (int): The cache's time-to-live in seconds.
        """
        self.cache_file = cache_file
        self.ttl = ttl
        self._coin_map: Dict[str, Dict[str, str]] = {}  # {id: {'symbol': '...', 'name': '...'}}
        self._symbol_map: Dict[str, str] = {}           # {lowercase_symbol: id}
        self._name_map: Dict[str, str] = {}             # {lowercase_name: id}
        
        self._load_or_fetch_data()

    def _load_or_fetch_data(self):
        """
        Orchestrates loading data from the cache or fetching from the API
        if the cache is missing, stale, or corrupt.
        """
        logger.info("Initializing coin mappings...")
        if self._load_from_cache():
            logger.info(f"Successfully loaded coin mappings from cache. {len(self._coin_map)} coins mapped.")
        else:
            logger.warning("Cache invalid, corrupt, or expired. Fetching fresh data from CoinGecko API.")
            if not self._fetch_from_api():
                logger.critical("Failed to fetch data from API and cache is unavailable. Mapper will be non-functional.")

    def _load_from_cache(self) -> bool:
        """
        Loads the coin data from the local JSON cache file if it's valid and not expired.

        Returns:
            bool: True if the cache was successfully loaded, False otherwise.
        """
        if not self.cache_file.exists():
            logger.info("Cache file does not exist.")
            return False
            
        try:
            with self.cache_file.open("r", encoding='utf-8') as f:
                data = json.load(f)
            
            timestamp = data.get("_timestamp", 0)
            if time.time() - timestamp > self.ttl:
                logger.info("Cache has expired (older than 24 hours).")
                return False

            coin_list = data.get("coins")
            if not coin_list or not isinstance(coin_list, list):
                logger.warning("Cache file is malformed; 'coins' key is missing or not a list.")
                return False

            self._build_mappings(coin_list)
            return True

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from cache file. File may be corrupted.", exc_info=True)
            return False
        except (IOError, Exception) as e:
            logger.error(f"An unexpected error occurred while loading cache: {e}", exc_info=True)
            return False

    def _fetch_from_api(self) -> bool:
        """
        Fetches the complete list of coins from the CoinGecko API.

        Returns:
            bool: True if data was successfully fetched, False otherwise.
        """
        logger.info(f"Requesting coin list from CoinGecko API: {COINGECKO_API_URL}")
        try:
            response = requests.get(COINGECKO_API_URL, timeout=20)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            
            coin_list = response.json()
            if not isinstance(coin_list, list):
                 logger.error(f"CoinGecko API returned unexpected data type: {type(coin_list)}")
                 return False

            self._build_mappings(coin_list)
            self._save_to_cache(coin_list)
            logger.info(f"Successfully fetched and processed {len(coin_list)} coins from CoinGecko.")
            return True

        except requests.exceptions.Timeout:
            logger.error("Request to CoinGecko API timed out.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred during API request to CoinGecko: {e}", exc_info=True)
            return False

    def _build_mappings(self, coin_list: List[Dict[str, str]]):
        """
        Builds internal lookup dictionaries for efficient mapping from the raw coin list.
        """
        self._coin_map.clear()
        self._symbol_map.clear()
        self._name_map.clear()
        
        for coin in coin_list:
            # Ensure the coin entry has the required keys
            if 'id' in coin and 'symbol' in coin and 'name' in coin:
                coin_id = coin['id'].lower()
                symbol = coin['symbol'].lower()
                name = coin['name'].lower()
                
                # The main map storing all data
                self._coin_map[coin_id] = {'symbol': coin['symbol'].upper(), 'name': coin['name']}
                
                # Create lookup maps for symbol and name -> id
                if symbol not in self._symbol_map: # Prioritize first entry for a given symbol (e.g., 'eth' for Ethereum)
                    self._symbol_map[symbol] = coin_id
                if name not in self._name_map:
                    self._name_map[name] = coin_id
            else:
                logger.debug(f"Skipping malformed coin entry in list: {coin}")

    def _save_to_cache(self, coin_list: List[Dict[str, str]]):
        """Saves the fetched coin list and a timestamp to the cache file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data_to_cache = {
                "_timestamp": time.time(),
                "coins": coin_list
            }
            with self.cache_file.open("w", encoding='utf-8') as f:
                json.dump(data_to_cache, f, indent=2)
            logger.info(f"Saved fresh coin data to cache file: {self.cache_file}")
        except (IOError, Exception) as e:
            logger.error(f"Failed to save data to cache file: {e}", exc_info=True)

    def get_info(self, identifier: str) -> Optional[Dict[str, str]]:
        """
        Gets all information (symbol, name) for a given identifier (name, symbol, or ID).

        Args:
            identifier (str): The coin name (e.g., "Bitcoin"), symbol ("BTC"), or ID ("bitcoin").

        Returns:
            Optional[Dict[str, str]]: A dictionary with {'symbol': '...', 'name': '...'} or None if not found.
        """
        if not identifier or not isinstance(identifier, str):
            return None
            
        clean_id = identifier.strip().lower()
        
        # Determine the canonical CoinGecko ID
        coin_id = self._name_map.get(clean_id) or self._symbol_map.get(clean_id) or (clean_id if clean_id in self._coin_map else None)
        
        if coin_id:
            return self._coin_map.get(coin_id)
        
        logger.debug(f"Could not resolve '{identifier}' to any known coin information.")
        return None

    def get_symbol(self, identifier: str) -> Optional[str]:
        """
        Gets the canonical trading symbol (e.g., "BTC") for a given identifier.
        
        Args:
            identifier (str): The coin name, symbol, or ID.

        Returns:
            Optional[str]: The uppercase symbol or None if not found.
        """
        info = self.get_info(identifier)
        return info['symbol'] if info else None

# --- Singleton Instance and Wrapper Functions ---
# Create a single instance of the mapper to be used across the application.
# This prevents reloading the cache unnecessarily.
mapper = CoinMapper()

def get_symbol(identifier: str) -> Optional[str]:
    """
    Convenience wrapper to get a coin's symbol from its name, symbol, or ID.
    e.g., get_symbol("Bitcoin") -> "BTC"
    e.g., get_symbol("eth") -> "ETH"
    """
    # If input is a pair like 'BTC/USDT', extract the base.
    base_identifier = identifier.split('/')[0]
    return mapper.get_symbol(base_identifier)

def get_trading_pair(identifier: str, quote: str = "USDT") -> Optional[str]:
    """
    Convenience wrapper to get a full trading pair for a given identifier.
    e.g., get_trading_pair("Bitcoin") -> "BTC/USDT"
    """
    base_symbol = get_symbol(identifier)
    if base_symbol:
        return f"{base_symbol}/{quote.upper()}"
    return None

def generate_symbol_variants(identifier: str, quotes: List[str] = ["USDT", "BUSD", "USDC", "BTC", "ETH"]) -> Dict[str, List[str]]:
    """
    Generates common exchange-specific format variations for a given coin identifier.

    Args:
        identifier (str): The coin name or symbol.
        quotes (List[str]): A list of quote currencies to generate pairs for.

    Returns:
        A dictionary containing lists of symbol variants for different formats.
    """
    base_symbol = get_symbol(identifier)
    if not base_symbol:
        return {}

    variants = {
        "concatenated": [], # e.g., BTCUSDT
        "slash_separated": [], # e.g., BTC/USDT
        "underscore_separated": [] # e.g., btc_usdt
    }
    for quote in quotes:
        variants["concatenated"].append(f"{base_symbol}{quote}")
        variants["slash_separated"].append(f"{base_symbol}/{quote}")
        variants["underscore_separated"].append(f"{base_symbol.lower()}_{quote.lower()}")
    
    return variants

# --- Example Usage (CLI) ---
if __name__ == "__main__":
    print("--- Coin Symbol Mapper CLI ---")
    
    # The mapper is already initialized when the module is imported.
    if not mapper._coin_map:
        print("\nMapper initialization failed. Check logs for details.")
    else:
        print(f"\nMapper initialized with {len(mapper._coin_map)} coins.")
        
        test_inputs = ["Bitcoin", "eth", "sol", "cardano", "XRP", "not-a-coin", "BTC/USDT"]
        
        for test_input in test_inputs:
            print(f"\n--- Testing Input: '{test_input}' ---")
            
            # Test get_info
            info = mapper.get_info(test_input)
            if info:
                print(f"  get_info() -> Name: {info['name']}, Symbol: {info['symbol']}")
            else:
                print(f"  get_info() -> Not Found")

            # Test get_symbol
            symbol = get_symbol(test_input)
            print(f"  get_symbol() -> {symbol or 'Not Found'}")
            
            # Test get_trading_pair
            pair = get_trading_pair(test_input)
            print(f"  get_trading_pair() -> {pair or 'Not Found'}")

            # Test generate_symbol_variants
            if symbol:
                variants = generate_symbol_variants(symbol, quotes=["USDT", "BTC"])
                print(f"  generate_symbol_variants() -> {variants}")

