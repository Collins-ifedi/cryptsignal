# market_data.py
import os
import sys
import logging
import time
from typing import Optional, List, Dict, Tuple
from pathlib import Path
# This version is fully synchronous and uses the standard 'requests' library via ccxt
# It does NOT use asyncio or aiohttp.
import ccxt
import pandas as pd

# --- ROBUST PATH SETUP ---
# This setup ensures Python can find your local modules like 'utils'.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
try:
    from utils.coin_symbol_mapper import get_symbol, generate_symbol_variants
    from utils.logger import get_logger
    logger = get_logger("MarketData")
except (ImportError, ModuleNotFoundError) as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MarketData_Fallback")
    logger.error(f"Failed to import from 'utils': {e}. Using fallback functions.")
    def get_symbol(identifier: str) -> Optional[str]: return identifier.upper().split('/')[0]
    def generate_symbol_variants(identifier: str, quotes: List[str]) -> Dict:
        return {"slash_separated": [f"{identifier}/{q}" for q in quotes], "concatenated": [f"{identifier}{q}" for q in quotes]}


class MarketDataFetcher:
    """
    Fetches historical OHLCV data by searching multiple exchanges using the
    standard synchronous ccxt library (which uses 'requests').
    """
    EXCHANGE_SEARCH_HIERARCHY = [
        {'id': 'binance', 'spot': True, 'futures': True},
        {'id': 'bybit', 'spot': True, 'futures': True},
        {'id': 'okx', 'spot': True, 'futures': True},
        {'id': 'bitget', 'spot': True, 'futures': True},
    ]

    def __init__(self, timeframe: str = "1h", limit: int = 500):
        self.timeframe = timeframe
        self.limit = limit
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        logger.info("MarketDataFetcher (SYNC Version) initialized.")

    def _get_exchange_instance(self, exchange_id: str) -> Optional[ccxt.Exchange]:
        """
        Lazily initializes and caches a synchronous ccxt exchange instance.
        """
        if exchange_id not in self._exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                instance = exchange_class({'enableRateLimit': True, 'timeout': 20000})
                instance.load_markets()
                self._exchanges[exchange_id] = instance
                logger.info(f"Successfully initialized sync exchange for '{exchange_id}'.")
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                logger.error(f"Network error initializing '{exchange_id}': Could not connect. Check internet connection.")
                logger.debug(f"Details: {e}")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize exchange '{exchange_id}': {e}", exc_info=True)
                return None
        return self._exchanges[exchange_id]

    def _find_active_market(self, identifier: str) -> Optional[Tuple[ccxt.Exchange, str]]:
        """
        Searches synchronously to find the first active trading pair.
        """
        base_symbol = get_symbol(identifier)
        if not base_symbol:
            logger.error(f"Could not resolve '{identifier}' to a base symbol.")
            return None

        variants = generate_symbol_variants(base_symbol, quotes=["USDT", "BUSD", "USDC"])
        potential_symbols = variants.get("slash_separated", []) + variants.get("concatenated", [])

        for exchange_config in self.EXCHANGE_SEARCH_HIERARCHY:
            exchange = self._get_exchange_instance(exchange_config['id'])
            if not exchange:
                continue

            for symbol in potential_symbols:
                if symbol in exchange.markets:
                    logger.info(f"Found active market: {symbol} on {exchange_config['id']}.")
                    return exchange, symbol
        
        logger.warning(f"Could not find any active market for '{identifier}' on configured exchanges.")
        return None
        
    def _process_ohlcv_data(self, ohlcv: list, symbol: str) -> Optional[pd.DataFrame]:
        """Helper to process raw OHLCV list into a clean DataFrame."""
        if not ohlcv:
            logger.warning(f"Received no OHLCV data for '{symbol}'.")
            return None
        
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        if df.empty:
            logger.warning(f"DataFrame for '{symbol}' is empty after cleaning.")
            return None
        return df

    def fetch_ohlcv(
        self,
        identifier: str,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
        retries: int = 2
    ) -> Optional[pd.DataFrame]:
        market_info = self._find_active_market(identifier)
        if not market_info:
            return None

        exchange, symbol = market_info
        use_timeframe = timeframe or self.timeframe
        use_limit = limit or self.limit
        
        logger.info(f"Fetching OHLCV for '{symbol}' from '{exchange.id}'...")

        for attempt in range(retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, use_timeframe, limit=use_limit)
                df = self._process_ohlcv_data(ohlcv, symbol)
                if df is not None:
                    logger.info(f"Sync fetch successful for {symbol} on {exchange.id}.")
                    return df
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {type(e).__name__}")
                if attempt < retries - 1:
                    time.sleep(1.5 ** attempt) # Synchronous sleep
                else:
                    logger.error(f"All fetch attempts failed for '{symbol}'.")
        return None

def main_cli():
    """Synchronous command-line interface for testing the MarketDataFetcher."""
    print("--- Market Data Fetcher CLI (Sync Version) ---")
    fetcher = MarketDataFetcher()
    try:
        while True:
            asset_input = input("\nEnter a coin name or symbol (or 'exit' to quit): ").strip()
            if not asset_input: continue
            if asset_input.lower() == 'exit': break

            print(f"Searching for '{asset_input}'...")
            df = fetcher.fetch_ohlcv(asset_input)

            if df is not None and not df.empty:
                print(f"\nSuccessfully fetched {len(df)} candles.")
                print("--- Data Sample (Last 5 rows) ---")
                print(df.tail())
            else:
                print(f"\nCould not fetch data for '{asset_input}'. Please check the asset name and your network connection.")
    finally:
        print("\nFetcher closed. Goodbye!")


if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the CLI: {e}", exc_info=True)