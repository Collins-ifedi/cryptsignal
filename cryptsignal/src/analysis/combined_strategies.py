# combined_strategies.py

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# --- Environment and Path Setup ---
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Module Imports ---
try:
    # Use the latest synchronous version of MarketDataFetcher
    from utils.coin_symbol_mapper import get_symbol, generate_symbol_variants
    from src.data.market_data import MarketDataFetcher
    from src.analysis.technical_analysis import TechnicalAnalyzer
    from utils.logger import get_logger
    logger = get_logger("CombinedStrategies")
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CombinedStrategies_Fallback")
    logger.error(f"A critical module could not be imported: {e}. The script may not function correctly.")
    class MarketDataFetcher: pass
    class TechnicalAnalyzer: pass


# ----------------------------------------------------------------------------
# --- CORE TRADING STRATEGY LOGIC ---
# ----------------------------------------------------------------------------
# Each function now returns a dictionary with the signal and a suggested entry point.

def fvg_strategy(price_data: List[Dict[str, float]], indicators: Dict[str, List]) -> Dict[str, Any]:
    """Identifies a Fair Value Gap and suggests an entry at the edge of the gap."""
    if len(price_data) < 3:
        return {'signal': 'neutral'}
    
    # Bullish FVG: Entry at the top of the gap (the previous candle's high)
    if price_data[-3]['high'] < price_data[-1]['low']:
        return {'signal': 'long', 'entry_point': price_data[-3]['high']}
    
    # Bearish FVG: Entry at the bottom of the gap (the previous candle's low)
    elif price_data[-3]['low'] > price_data[-1]['high']:
        return {'signal': 'short', 'entry_point': price_data[-3]['low']}
        
    return {'signal': 'neutral'}


def ema_rsi_strategy(price_data: List[Dict[str, float]], indicators: Dict[str, List]) -> Dict[str, Any]:
    """Suggests an entry on a pullback to a key moving average."""
    try:
        # Note: Using SMA_200 as it's more commonly used for long-term trend context
        ema_50 = indicators.get('EMA_50', [0])[-1]
        ema_200 = indicators.get('SMA_200', [0])[-1]
        close = price_data[-1]['close']
        rsi = indicators.get('RSI_14', [50])[-1]
    except (KeyError, IndexError):
        return {'signal': 'neutral'}

    # Long signal: Suggests entering on a pullback to the EMA 50
    if close > ema_50 > ema_200 and rsi < 70:
        return {'signal': 'long', 'entry_point': ema_50}
    
    # Short signal: Suggests entering on a bounce to the EMA 50
    elif close < ema_50 < ema_200 and rsi > 30:
        return {'signal': 'short', 'entry_point': ema_50}
        
    return {'signal': 'neutral'}


def breakout_retest_strategy(price_data: List[Dict[str, float]], indicators: Dict[str, List]) -> Dict[str, Any]:
    """Suggests an entry on the retest of a breakout level."""
    if len(price_data) < 10:
        return {'signal': 'neutral'}

    lookback_period = 10
    recent_high = max(p['high'] for p in price_data[-lookback_period-1:-1])
    recent_low = min(p['low'] for p in price_data[-lookback_period-1:-1])
    close = price_data[-1]['close']

    # Long signal: Suggests entering at the previous high (the breakout level)
    if close > recent_high:
        return {'signal': 'long', 'entry_point': recent_high}
    
    # Short signal: Suggests entering at the previous low
    elif close < recent_low:
        return {'signal': 'short', 'entry_point': recent_low}
        
    return {'signal': 'neutral'}


def fib_strategy(price_data: List[Dict[str, float]], indicators: Dict[str, List]) -> Dict[str, Any]:
    """Uses Fibonacci levels as potential entry points."""
    if len(price_data) < 30:
        return {'signal': 'neutral'}

    lookback_period = 30
    high = max(p['high'] for p in price_data[-lookback_period:])
    low = min(p['low'] for p in price_data[-lookback_period:])
    
    if high == low:
        return {'signal': 'neutral'}

    current_price = price_data[-1]['close']
    fib_618_level = high - 0.618 * (high - low)

    if current_price > fib_618_level:
        return {'signal': 'long', 'entry_point': fib_618_level}
    
    elif current_price < fib_618_level:
        return {'signal': 'short', 'entry_point': fib_618_level}
        
    return {'signal': 'neutral'}


def divergence_trendline_strategy(price_data: List[Dict[str, float]], indicators: Dict[str, List]) -> Dict[str, Any]:
    """Identifies divergence; entry is near current price as confirmation."""
    if len(price_data) < 2:
        return {'signal': 'neutral'}
        
    try:
        macd_hist = indicators.get('MACDh_12_26_9', [0, 0])[-2:]
        price = [p['close'] for p in price_data][-2:]
    except (KeyError, IndexError):
        return {'signal': 'neutral'}

    # For divergence, a good entry is often upon confirmation, close to the current price.
    current_price = price_data[-1]['close']
    
    # Bearish divergence: Price makes a higher high while the indicator makes a lower high.
    if macd_hist[1] < macd_hist[0] and price[1] > price[0]:
        return {'signal': 'short', 'entry_point': current_price}
    
    # Bullish divergence: Price makes a lower low while the indicator makes a higher low.
    elif macd_hist[1] > macd_hist[0] and price[1] < price[0]:
        return {'signal': 'long', 'entry_point': current_price}
        
    return {'signal': 'neutral'}


# ----------------------------------------------------------------------------
# --- STRATEGY RUNNER CLASS (UPDATED) ---
# ----------------------------------------------------------------------------

class CombinedStrategiesRunner:
    """
    Orchestrates fetching market data, calculating technical indicators,
    and running a suite of trading strategies synchronously.
    """
    STRATEGIES = {
        'FVG': fvg_strategy,
        'EMA_RSI': ema_rsi_strategy,
        'Breakout_Retest': breakout_retest_strategy,
        'Fibonacci': fib_strategy,
        'Divergence_Trendline': divergence_trendline_strategy
    }

    def __init__(self, timeframe: str = "1h", limit: int = 300):
        if 'MarketDataFetcher' not in globals() or 'TechnicalAnalyzer' not in globals():
            raise ImportError("Dependencies MarketDataFetcher or TechnicalAnalyzer not loaded.")
        
        # Initialize with the synchronous MarketDataFetcher
        self.fetcher = MarketDataFetcher(timeframe=timeframe, limit=limit)
        self.timeframe = timeframe
        self.limit = limit
        logger.info(f"CombinedStrategiesRunner (Sync) initialized for timeframe {timeframe}.")

    def _prepare_data_for_strategies(self, df_indicators: pd.DataFrame) -> (List[Dict], Dict[str, List]):
        price_data = df_indicators[['open', 'high', 'low', 'close', 'volume']].to_dict('records')
        indicator_cols = [col for col in df_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        indicators = {col: df_indicators[col].tolist() for col in indicator_cols}
        return price_data, indicators

    def run_all_strategies(self, symbol: str, df_ohlcv: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Dict]]:
        """
        Executes the full pipeline synchronously.

        Args:
            symbol (str): The trading symbol to analyze.
            df_ohlcv (Optional[pd.DataFrame]): A pre-fetched DataFrame. If None,
                                               data will be fetched internally.

        Returns:
            Optional[Dict[str, Dict]]: A dictionary with the results of each strategy.
        """
        logger.info(f"Running all strategies for {symbol}...")
        
        # --- MODIFIED SECTION ---
        # 1. Fetch Market Data only if not provided
        if df_ohlcv is None or df_ohlcv.empty:
            logger.info(f"No DataFrame provided for {symbol}, fetching data...")
            df_ohlcv = self.fetcher.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.limit)
            if df_ohlcv is None or df_ohlcv.empty:
                logger.error(f"Failed to fetch market data for {symbol}. Cannot run strategies.")
                return None
        else:
            logger.info(f"Using pre-fetched DataFrame for {symbol}.")
        # --- END MODIFIED SECTION ---

        # 2. Calculate Technical Indicators
        analyzer = TechnicalAnalyzer(df_ohlcv)
        df_indicators = analyzer.generate_all_indicators()
        if df_indicators is None or df_indicators.empty:
            logger.error(f"Failed to generate technical indicators for {symbol}.")
            return None

        # 3. Prepare Data
        price_data, indicators = self._prepare_data_for_strategies(df_indicators)
        
        # 4. Run Strategies
        results = {}
        for name, strategy_func in self.STRATEGIES.items():
            try:
                results[name] = strategy_func(price_data, indicators)
            except Exception as e:
                logger.error(f"Error executing strategy '{name}' for {symbol}: {e}", exc_info=True)
                results[name] = {'signal': 'error'}
        
        logger.info(f"Finished running strategies for {symbol}.")
        return results

# ----------------------------------------------------------------------------
# --- COMMAND-LINE INTERFACE (CLI) FOR TESTING ---
# ----------------------------------------------------------------------------

def main_cli():
    """Defines the synchronous main function for the command-line interface."""
    print("--- Combined Strategies Runner CLI (Sync Version) ---")
    
    symbol_input = input("Enter trading symbol (e.g., BTC/USDT) [Default: BTC/USDT]: ").strip().upper() or "BTC/USDT"
    timeframe_input = input("Enter timeframe (e.g., 1h, 4h, 1d) [Default: 1h]: ").strip() or "1h"
    
    try:
        runner = CombinedStrategiesRunner(timeframe=timeframe_input)
        strategy_results = runner.run_all_strategies(symbol=symbol_input)
        
        if strategy_results:
            print("\n--- Strategy Results ---")
            print(f"Asset: {symbol_input}")
            print(f"Timeframe: {timeframe_input}")
            print("-" * 26)
            print(json.dumps(strategy_results, indent=4))
            print("-" * 26)
        else:
            print("\nCould not generate strategy results. Check logs for errors.")
            
    except ImportError as e:
        logger.critical(f"CLI could not start due to missing dependencies: {e}")
        print(f"\nERROR: A required module is missing. Please check your project setup. Details: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the CLI: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")