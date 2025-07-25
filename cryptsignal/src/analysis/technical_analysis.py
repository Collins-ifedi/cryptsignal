# technical_analysis.py
import sys
import os
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTS ---
try:
    # This import is primarily for the CLI testing functionality below.
    from src.data.market_data import MarketDataFetcher
    _MarketDataFetcher_imported = True
except ImportError:
    _MarketDataFetcher_imported = False
    class MarketDataFetcher: pass
    logging.warning("MarketDataFetcher not found. CLI testing requiring it may fail.")

try:
    from utils.logger import get_logger
    logger = get_logger("TechnicalAnalysis")
except ImportError:
    logging.warning("get_logger utility not found. Using basic logging for TechnicalAnalysis.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("TechnicalAnalysis_Fallback")

class TechnicalAnalyzer:
    """
    Analyzes OHLCV data to generate a wide range of technical indicators and
    provides a structured summary of the technical outlook.
    """
    def __init__(self, df_ohlcv: pd.DataFrame):
        """
        Initializes the analyzer with the OHLCV DataFrame.

        Args:
            df_ohlcv (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
                                     A datetime index is expected.
        """
        if not isinstance(df_ohlcv, pd.DataFrame):
            logger.error("TechnicalAnalyzer initialized with non-DataFrame input.")
            self.df_ohlcv_original = pd.DataFrame()
            self.df_with_indicators = pd.DataFrame()
            return

        self.df_ohlcv_original = df_ohlcv.copy()

        if df_ohlcv.empty:
            logger.warning("TechnicalAnalyzer initialized with empty DataFrame.")
            self.df_with_indicators = pd.DataFrame()
            return

        df_processed = self._prepare_dataframe(df_ohlcv)
        self.df_with_indicators = df_processed
        self.strategy = self._get_indicator_strategy()
        logger.info(f"TechnicalAnalyzer initialized. Strategy includes {len(self.strategy.ta)} indicator configurations.")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes and cleans the input DataFrame."""
        df_processed = df.copy()
        rename_map = {
            "Timestamp": "timestamp", "Date": "timestamp", "date": "timestamp",
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        }
        actual_rename_map = {k: v for k, v in rename_map.items() if k in df_processed.columns}
        df_processed.rename(columns=actual_rename_map, inplace=True)

        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
            if not pd.api.types.is_datetime64_any_dtype(df_processed.index):
                df_processed.set_index('timestamp', inplace=True)
        elif not pd.api.types.is_datetime64_any_dtype(df_processed.index):
             logger.warning("DataFrame has no 'timestamp' column and index is not datetime. Time-based ops may fail.")

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            else:
                logger.warning(f"Required column '{col}' not found. Creating empty column.")
                df_processed[col] = np.nan

        initial_rows = len(df_processed)
        df_processed.dropna(subset=required_cols, inplace=True)
        if len(df_processed) < initial_rows:
            logger.info(f"Dropped {initial_rows - len(df_processed)} rows with NaN in essential OHLCV columns.")

        return df_processed

    def _get_indicator_strategy(self) -> ta.Strategy:
        """Defines the comprehensive set of technical indicators to be calculated."""
        return ta.Strategy(
            name="ComprehensiveIndicatorSet",
            description="Extensive set of indicators from Trend, Momentum, Volatility, and Volume categories",
            ta=[
                # Trend
                {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 12}, {"kind": "ema", "length": 26}, {"kind": "ema", "length": 50},
                {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                {"kind": "adx", "length": 14},
                {"kind": "aroon", "length": 14},
                {"kind": "psar"},
                # Momentum
                {"kind": "rsi", "length": 14},
                {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
                {"kind": "cci", "length": 20},
                {"kind": "mom", "length": 10},
                {"kind": "roc", "length": 12},
                # Volatility
                {"kind": "bbands", "length": 20, "std": 2},
                {"kind": "atr", "length": 14},
                {"kind": "kc", "length": 20, "scalar": 2, "mamode": "ema", "append": True},
                {"kind": "donchian", "lower_length": 20, "upper_length": 20},
                # Volume
                {"kind": "obv"},
                {"kind": "cmf", "length": 20},
                {"kind": "mfi", "length": 14},
                {"kind": "vwap", "anchor": "D", "append": True},
            ]
        )

    def _add_golden_death_crosses(self) -> None:
        """Calculates and adds Golden Cross and Death Cross signals to the DataFrame."""
        df = self.df_with_indicators
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            # Golden Cross: SMA50 crosses above SMA200
            condition_golden_curr = (df['SMA_50'] > df['SMA_200'])
            condition_golden_prev = (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
            df['GOLDEN_X'] = (condition_golden_curr & condition_golden_prev).astype(int)

            # Death Cross: SMA50 crosses below SMA200
            condition_death_curr = (df['SMA_50'] < df['SMA_200'])
            condition_death_prev = (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
            df['DEATH_X'] = (condition_death_curr & condition_death_prev).astype(int)
        else:
            logger.warning("SMA_50 or SMA_200 not available. Cannot calculate Golden/Death crosses.")
            df['GOLDEN_X'] = 0
            df['DEATH_X'] = 0

    def generate_all_indicators(self) -> Optional[pd.DataFrame]:
        """
        Calculates all defined technical indicators and adds them to the DataFrame.

        Returns:
            Optional[pd.DataFrame]: A copy of the DataFrame with all indicators, or None on failure.
        """
        if self.df_with_indicators.empty or not all(c in self.df_with_indicators.columns for c in ['open', 'high', 'low', 'close', 'volume']):
            logger.error("DataFrame is empty or missing required columns for indicator generation.")
            return None

        min_required_len = 200
        if len(self.df_with_indicators) < min_required_len:
             logger.warning(f"Data has {len(self.df_with_indicators)} rows, less than the {min_required_len} recommended for long-period indicators (e.g., SMA200).")

        try:
            self.df_with_indicators.ta.strategy(self.strategy, timed=False, verbose=False)
            self._add_golden_death_crosses()
            num_gen_cols = len(self.df_with_indicators.columns) - len(self.df_ohlcv_original.columns)
            logger.info(f"Successfully generated TA indicators. Added {num_gen_cols} columns.")
            return self.df_with_indicators.copy()
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}", exc_info=True)
            return self.df_with_indicators.copy() if not self.df_with_indicators.empty else None

    def get_structured_summary(self) -> Optional[Dict[str, Any]]:
        """
        Generates a structured dictionary summarizing the technical analysis.
        This method replaces the need for an external LLM call.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the technical summary,
                                      or None if data is insufficient.
        """
        if self.df_with_indicators.empty or len(self.df_with_indicators) < 2:
            logger.error("Cannot generate summary: indicator DataFrame is empty or has insufficient data.")
            return None

        latest = self.df_with_indicators.iloc[-1]
        close_price = latest.get('close')
        if pd.isna(close_price):
            logger.error("Cannot generate summary: latest close price is NaN.")
            return None

        # --- 1. Calculate Rule-Based Sentiment Score ---
        score = 0.0
        # RSI
        rsi = latest.get('RSI_14', 50)
        if rsi > 70: score -= 0.2
        elif rsi < 30: score += 0.2
        elif rsi > 55: score += 0.1
        elif rsi < 45: score -= 0.1
        
        # MACD
        macd_line = latest.get('MACD_12_26_9', 0)
        macd_signal = latest.get('MACDs_12_26_9', 0)
        if macd_line > macd_signal: score += 0.15
        else: score -= 0.15

        # Moving Averages
        sma50 = latest.get('SMA_50')
        sma200 = latest.get('SMA_200')
        if sma50 is not None and close_price > sma50: score += 0.1
        if sma200 is not None and close_price > sma200: score += 0.25
        if sma50 is not None and close_price < sma50: score -= 0.1
        if sma200 is not None and close_price < sma200: score -= 0.25

        # ADX
        adx = latest.get('ADX_14', 0)
        if adx > 25:
            dmp = latest.get('DMP_14', 0)
            dmn = latest.get('DMN_14', 0)
            if dmp > dmn: score += 0.15 # Strong uptrend
            else: score -= 0.15 # Strong downtrend
        
        # Golden/Death Cross in last 10 periods
        if latest.get('GOLDEN_X') == 1: score += 0.4
        elif self.df_with_indicators['GOLDEN_X'].tail(10).sum() > 0: score += 0.2
        if latest.get('DEATH_X') == 1: score -= 0.4
        elif self.df_with_indicators['DEATH_X'].tail(10).sum() > 0: score -= 0.2

        numeric_sentiment = np.clip(score, -1.0, 1.0)
        
        # --- 2. Determine Categorical Sentiment ---
        if numeric_sentiment > 0.4: sentiment_category = "Strong Bullish"
        elif numeric_sentiment > 0.15: sentiment_category = "Bullish"
        elif numeric_sentiment < -0.4: sentiment_category = "Strong Bearish"
        elif numeric_sentiment < -0.15: sentiment_category = "Bearish"
        else: sentiment_category = "Neutral"

        # --- 3. Construct Narrative and Cross Event Details ---
        narrative_parts = [f"Overall sentiment is {sentiment_category} (Score: {numeric_sentiment:.2f})."]
        cross_event_details = {"status": "N/A", "details": "SMA 50/200 not available."}

        if sma50 is not None and sma200 is not None:
            if latest.get('GOLDEN_X') == 1:
                cross_event_details = {"status": "Golden Cross", "details": "A Golden Cross (SMA50 over SMA200) just occurred."}
                narrative_parts.append("Major bullish signal: Golden Cross confirmed.")
            elif latest.get('DEATH_X') == 1:
                cross_event_details = {"status": "Death Cross", "details": "A Death Cross (SMA50 under SMA200) just occurred."}
                narrative_parts.append("Major bearish signal: Death Cross confirmed.")
            elif sma50 > sma200:
                cross_event_details = {"status": "Bullish Alignment", "details": "SMA50 is currently above SMA200."}
            else:
                cross_event_details = {"status": "Bearish Alignment", "details": "SMA50 is currently below SMA200."}

        if pd.notna(rsi): narrative_parts.append(f"RSI at {rsi:.2f}.")
        if pd.notna(macd_line): narrative_parts.append(f"MACD line ({macd_line:.2f}) is {'above' if macd_line > macd_signal else 'below'} signal ({macd_signal:.2f}).")
        
        # --- 4. Build Structured Dictionary ---
        obv_value = latest.get('OBV')
        summary = {
            "timestamp_utc": pd.Timestamp.now(tz='utc').isoformat(),
            "latest_candle_timestamp": latest.name.isoformat(),
            "close_price": close_price,
            "sentiment": {
                "category": sentiment_category,
                "numeric_score": round(numeric_sentiment, 3),
                "narrative": " ".join(narrative_parts)
            },
            "cross_event": cross_event_details,
            "key_indicators": {
                "rsi_14": round(latest.get('RSI_14', 0), 2),
                "macd_12_26_9": round(latest.get('MACD_12_26_9', 0), 4),
                "macds_12_26_9": round(latest.get('MACDs_12_26_9', 0), 4),
                "adx_14": round(latest.get('ADX_14', 0), 2),
                "stoch_k_14_3_3": round(latest.get('STOCHk_14_3_3', 0), 2),
                "atr_14": round(latest.get('ATR_14', 0), 4),
            },
            "price_vs_ma": {
                "vs_sma_20": "above" if close_price > latest.get('SMA_20', np.inf) else "below",
                "vs_sma_50": "above" if close_price > latest.get('SMA_50', np.inf) else "below",
                "vs_sma_200": "above" if close_price > latest.get('SMA_200', np.inf) else "below",
                "vs_ema_50": "above" if close_price > latest.get('EMA_50', np.inf) else "below",
            },
            "volatility": {
                "bbands_percentage": round(latest.get('BBP_20_2.0', 0), 3),
                "kc_percentage": round(latest.get('KCP_20_10_2', 0), 3),
            },
            "volume": {
                # UPDATED: Removed int() cast to fix FutureWarning and preserve float precision.
                "obv": obv_value if pd.notna(obv_value) else None,
                "cmf_20": round(latest.get('CMF_20', 0), 3)
            }
        }
        return summary

# --- Main CLI for testing ---
def main_cli():
    """UPDATED: Synchronous command-line interface for testing."""
    print("--- Technical Analysis Module CLI Test (Sync, No LLM) ---")

    if not _MarketDataFetcher_imported:
        print("ERROR: MarketDataFetcher is not available for CLI testing.")
        return

    # UPDATED: The MarketDataFetcher is now exchange-agnostic.
    asset_input = input("Enter coin name or symbol (e.g., BTC, Solana, ETH/USDT): ").strip() or "BTC"
    timeframe_input = input("Enter timeframe (e.g., 1d) [1d]: ").strip() or "1d"
    limit_input = int(input("Enter number of candles [300]: ").strip() or "300")

    df_ohlcv = None
    try:
        # UPDATED: Use the new synchronous, exchange-agnostic fetcher.
        logger.info(f"Fetching OHLCV for {asset_input} across available exchanges...")
        md_fetcher = MarketDataFetcher() # No exchange needed for init.
        df_ohlcv = md_fetcher.fetch_ohlcv(
            identifier=asset_input,
            timeframe=timeframe_input,
            limit=limit_input
        )
    except Exception as e:
        logger.error(f"Error fetching market data: {e}", exc_info=True)

    if df_ohlcv is None or df_ohlcv.empty:
        print(f"\nCould not fetch OHLCV data for {asset_input}. Aborting.")
        return

    print(f"\nFetched {len(df_ohlcv)} candles for {asset_input}.")
    analyzer = TechnicalAnalyzer(df_ohlcv)
    
    # Generate indicators and get the full DataFrame
    df_indicators = analyzer.generate_all_indicators()
    
    if df_indicators is not None and not df_indicators.empty:
        print("\n--- DataFrame with Technical Indicators (Last 5 rows) ---")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(df_indicators.tail())

        # Get the new structured summary
        print("\n--- Structured Technical Analysis Summary ---")
        structured_summary = analyzer.get_structured_summary()
        
        if structured_summary:
            import json
            print(json.dumps(structured_summary, indent=2))
        else:
            print("Failed to generate structured summary.")
    else:
        print("\nIndicator generation failed or resulted in an empty DataFrame.")


if __name__ == "__main__":
    if not logger.handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logger = logging.getLogger("TechnicalAnalysis_Main")

    try:
        # UPDATED: Direct synchronous call, no asyncio needed.
        main_cli()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.critical(f"A critical error occurred in the CLI: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")