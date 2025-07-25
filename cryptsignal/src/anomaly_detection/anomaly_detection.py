# anomaly_detection.py

import sys
import os
import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- IMPORTS ---
try:
    from src.data.market_data import MarketDataFetcher
    from src.analysis.technical_analysis import TechnicalAnalyzer
    from src.analysis.pattern_analyzer import analyze_patterns
    from src.analysis.sentiment_analysis import get_sentiment_snapshot
    from src.anomaly_detection.anomaly_storage import store_anomalies
    from utils.logger import get_logger
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AnomalyDetector_Fallback")
    logger.error(f"Failed to import necessary modules: {e}. Anomaly detection will not function.", exc_info=True)
    # Define dummy classes to prevent crashes on load
    class MarketDataFetcher: pass
    class TechnicalAnalyzer: pass
    def analyze_patterns(df): return {}
    async def get_sentiment_snapshot(symbol): return {}
    def store_anomalies(symbol, anomalies): pass
    def get_logger(name): return logging.getLogger(name)

logger = get_logger("AnomalyDetector")

class AnomalyDetector:
    """
    Detects market anomalies by analyzing technical indicators, price action,
    volume, and sentiment data.
    """
    def __init__(self, symbol: str, exchange: str = 'binance', timeframe: str = '1h', limit: int = 200):
        """
        Initializes the AnomalyDetector.

        Args:
            symbol (str): The trading symbol to analyze (e.g., 'BTC/USDT').
            exchange (str): The exchange to fetch data from.
            timeframe (str): The timeframe for the analysis (e.g., '1h', '4h').
            limit (int): The number of candles to fetch.
        """
        self.trading_symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.limit = limit
        self.market_fetcher = MarketDataFetcher()
        logger.info(f"AnomalyDetector initialized for {self.trading_symbol} on {self.exchange} ({self.timeframe}).")

    async def _fetch_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Fetches and prepares all necessary data for anomaly detection."""
        try:
            # Fetch market data
            ohlcv_df = self.market_fetcher.fetch_ohlcv(
                identifier=self.trading_symbol,
                timeframe=self.timeframe,
                limit=self.limit
            )

            if ohlcv_df is None or ohlcv_df.empty:
                logger.error(f"Could not fetch OHLCV data for {self.trading_symbol}.")
                return None

            # Get technical analysis
            ta_analyzer = TechnicalAnalyzer(ohlcv_df)
            df_with_indicators = ta_analyzer.generate_all_indicators()

            if df_with_indicators is None or df_with_indicators.empty:
                logger.error(f"Indicator generation failed for {self.trading_symbol}.")
                return None

            return df_with_indicators
        except Exception as e:
            logger.error(f"Error in data fetching/preparation pipeline: {e}", exc_info=True)
            return None

    def _check_technical_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks for anomalies based on technical indicators."""
        if df.empty or len(df) < 2:
            return {"score": 0, "factors": []}

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        factors = []

        # 1. Bollinger Bands Excursion
        if 'BBP_20_2.0' in latest and latest['BBP_20_2.0'] > 1.05:
            score += 3
            factors.append(f"Price exceeded Upper Bollinger Band (BBP: {latest['BBP_20_2.0']:.2f}).")
        if 'BBP_20_2.0' in latest and latest['BBP_20_2.0'] < -0.05:
            score += 3
            factors.append(f"Price dropped below Lower Bollinger Band (BBP: {latest['BBP_20_2.0']:.2f}).")

        # 2. Extreme RSI
        if 'RSI_14' in latest and latest['RSI_14'] > 85:
            score += 2
            factors.append(f"RSI is extremely overbought ({latest['RSI_14']:.2f}).")
        if 'RSI_14' in latest and latest['RSI_14'] < 15:
            score += 2
            factors.append(f"RSI is extremely oversold ({latest['RSI_14']:.2f}).")

        # 3. Volume Spike
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        if latest['volume'] > avg_volume * 5:
            score += 4
            factors.append(f"Volume spike detected ({latest['volume']:,.0f} vs avg {avg_volume:,.0f}, >5x).")

        # 4. Unusually Large Candle
        avg_candle_range = (df['high'] - df['low']).rolling(window=20).mean().iloc[-1]
        latest_candle_range = latest['high'] - latest['low']
        if latest_candle_range > avg_candle_range * 3:
            score += 2
            factors.append(f"Unusually large candle range detected ({latest_candle_range:.2f} vs avg {avg_candle_range:.2f}, >3x).")

        return {"score": score, "factors": factors}

    async def _check_sentiment_anomalies(self) -> Dict[str, Any]:
        """Checks for anomalies based on market sentiment."""
        score = 0
        factors = []
        try:
            coin_name = self.trading_symbol.split('/')[0]
            sentiment_snapshot = await get_sentiment_snapshot(coin_name)

            if not sentiment_snapshot or sentiment_snapshot.get("error"):
                logger.warning(f"Could not get sentiment snapshot for {coin_name}.")
                return {"score": 0, "factors": []}

            # Extreme Fear & Greed Index
            fg_index_data = sentiment_snapshot.get('fear_greed_index')
            if fg_index_data and isinstance(fg_index_data, dict):
                fg_value = fg_index_data.get('value')
                if isinstance(fg_value, int):
                    if fg_value > 85:
                        score += 2
                        factors.append(f"Extreme Greed detected in Fear & Greed Index (Value: {fg_value}).")
                    if fg_value < 15:
                        score += 2
                        factors.append(f"Extreme Fear detected in Fear & Greed Index (Value: {fg_value}).")

            # High Social Sentiment Score (Can be contrarian)
            social_score = sentiment_snapshot.get('average_social_sentiment_score')
            if isinstance(social_score, float):
                if social_score > 0.75:
                    score += 1
                    factors.append(f"Extremely high social media sentiment score ({social_score:.2f}).")
                if social_score < -0.75:
                    score += 1
                    factors.append(f"Extremely low social media sentiment score ({social_score:.2f}).")

            return {"score": score, "factors": factors}
        except Exception as e:
            logger.error(f"Error checking sentiment anomalies: {e}", exc_info=True)
            return {"score": 0, "factors": []}

    async def detect(self) -> Dict[str, Any]:
        """
        Runs the full anomaly detection pipeline.

        Returns:
            A dictionary report of any detected anomalies.
        """
        logger.info(f"Starting anomaly detection pipeline for {self.trading_symbol}...")
        df = await self._fetch_and_prepare_data()

        if df is None:
            return {
                "symbol": self.trading_symbol,
                "anomaly_detected": False,
                "anomaly_score": 0,
                "error": "Critical pipeline error: Could not fetch or prepare data.",
                "contributing_factors": {}
            }

        # Run checks concurrently
        technical_task = asyncio.to_thread(self._check_technical_anomalies, df)
        sentiment_task = self._check_sentiment_anomalies()

        tech_anomalies, sentiment_anomalies = await asyncio.gather(technical_task, sentiment_task)

        total_score = tech_anomalies.get("score", 0) + sentiment_anomalies.get("score", 0)
        anomaly_detected = total_score >= 5  # Set a threshold for what constitutes an anomaly

        report = {
            "symbol": self.trading_symbol,
            "timeframe": self.timeframe,
            "anomaly_detected": anomaly_detected,
            "anomaly_score": total_score,
            "error": None,
            "contributing_factors": {
                "technical_factors": tech_anomalies,
                "sentiment_factors": sentiment_anomalies,
            }
        }

        if anomaly_detected:
            logger.warning(f"Anomaly DETECTED for {self.trading_symbol} with score {total_score}.")
            # Prepare a simplified list of anomalies for storage
            anomalies_to_store = []
            if tech_anomalies["factors"]:
                anomalies_to_store.append({"type": "technical", "details": ", ".join(tech_anomalies["factors"])})
            if sentiment_anomalies["factors"]:
                anomalies_to_store.append({"type": "sentiment", "details": ", ".join(sentiment_anomalies["factors"])})

            # Store the detected anomalies
            if anomalies_to_store:
                store_anomalies(self.trading_symbol, anomalies_to_store)
        else:
            logger.info(f"No significant anomalies detected for {self.trading_symbol}. Score: {total_score}.")

        return report

async def main():
    """Main function for CLI testing."""
    symbol = "BTC/USDT"
    logger.info(f"--- Running Anomaly Detection Test for {symbol} ---")
    detector = AnomalyDetector(symbol=symbol, timeframe='1h')
    try:
        report = await detector.detect()
        import json
        print(json.dumps(report, indent=2))
    except Exception as e:
        logger.critical(f"An error occurred during the main test run: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())