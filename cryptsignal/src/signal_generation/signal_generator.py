# signal_generator.py

# --- Global Imports ---
import sys
import os
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Environment and Path Setup ---
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Module Imports with Fallbacks ---
try:
    from src.data.market_data import MarketDataFetcher
    from src.analysis.technical_analysis import TechnicalAnalyzer
    from src.analysis.pattern_analyzer import analyze_patterns as analyze_candlestick_patterns
    from src.analysis.combined_strategies import CombinedStrategiesRunner
    from utils.logger import get_logger
    logger = get_logger("SignalGenerator")
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger("SignalGenerator_Fallback")
    logger.error(f"Critical Import Error: {e}. SignalGenerator will not function correctly.")
    # Define dummy classes to allow the script to be parsed without crashing.
    class MarketDataFetcher: pass
    class TechnicalAnalyzer: pass
    class CombinedStrategiesRunner: pass
    def analyze_candlestick_patterns(df: pd.DataFrame) -> Dict[str, Any]: return {}


class SignalGenerator:
    """
    Generates trading signals using a multi-timeframe analysis (MTF) approach:
    - 4h: Determines the dominant trend and its strength (Trend Filter).
    - 1h: Identifies optimal entry points and setups (Entry Signal).
    - 15m: Confirms short-term momentum before issuing a signal (Confirmation Trigger).
    Includes state management to avoid duplicate signals.
    """
    def __init__(self, default_timeframe: str = "1h", default_limit: int = 300):
        try:
            self.strategy_runner = CombinedStrategiesRunner(timeframe=default_timeframe, limit=default_limit)
        except Exception as e_init:
            logger.critical(f"Failed to initialize a required component: {e_init}", exc_info=True)
            self.strategy_runner = None

        # --- ADJUSTED PARAMETERS FOR HIGHER SENSITIVITY ---
        self.atr_multiplier_sl = 2.0
        self.min_data_points = 100
        # Lowered significantly to accept even weak trends
        self.adx_trend_threshold = 16
        # Lowered to make 1h analysis easier to pass
        self.signal_threshold = 0.25
        
        self.state_file_path = os.path.join(PROJECT_ROOT, "active_signals.json")

        self.weights = {
            "combined_strategies": 0.50,
            "technical_analysis": 0.30,
            "pattern_analysis": 0.20,
        }

    def _manage_state(self, action: str, symbol: str = None, signal: dict = None) -> Any:
        """Manages active signals in a JSON file to prevent duplicates."""
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, 'r') as f:
                    active_signals = json.load(f)
            else:
                active_signals = {}
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading state file: {e}")
            active_signals = {}

        if action == 'check':
            if symbol in active_signals:
                logger.info(f"State Check: An active signal for {symbol} already exists. No new signal will be generated.")
                return True, "An active signal for this symbol already exists."
            return False, ""
            
        if action == 'add' and symbol and signal:
            active_signals[symbol] = signal
            try:
                with open(self.state_file_path, 'w') as f:
                    json.dump(active_signals, f, indent=4)
                logger.info(f"State Update: Added new active signal for {symbol}.")
            except IOError as e:
                logger.error(f"Could not write to state file: {e}")
        return False, "" # Default return

    def _calculate_signal_score(self, ta_summary, patterns, strategy_signals):
        score = 0.0
        strategy_score_component = 0.0
        if strategy_signals:
            numeric_signals = [1 if result.get('signal') == 'long' else -1 for result in strategy_signals.values() if result.get('signal')]
            if numeric_signals: strategy_score_component = np.mean(numeric_signals)
        score += strategy_score_component * self.weights["combined_strategies"]
        
        ta_score_component = ta_summary.get('sentiment', {}).get('numeric_score', 0.0)
        score += ta_score_component * self.weights["technical_analysis"]
        
        pattern_score_component = 0.0
        if patterns:
            if patterns.get("sentiment") == "Bullish": pattern_score_component = 1.0
            elif patterns.get("sentiment") == "Bearish": pattern_score_component = -1.0
        score += pattern_score_component * self.weights["pattern_analysis"]
        return np.clip(score, -1, 1)

    def _determine_4h_trend(self, symbol: str) -> tuple[Optional[str], str]:
        """Analyzes the 4h timeframe for dominant trend and strength."""
        logger.info(f"1/4: Analyzing 4h trend for {symbol}...")
        if not self.strategy_runner: return None, "Strategy Runner not initialized."
        df = self.strategy_runner.fetcher.fetch_ohlcv(symbol, "4h", 200)
        if df is None or len(df) < 100:
            return None, "Insufficient 4h data."

        # Indicators
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # ADX Calculation
        high_minus_low = df['high'] - df['low']
        high_minus_prev_close = abs(df['high'] - df['close'].shift(1))
        low_minus_prev_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_minus_low, high_minus_prev_close, low_minus_prev_close], axis=1).max(axis=1, skipna=False)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        high_diff, low_diff = df['high'].diff(), df['low'].diff()
        plus_dm = (high_diff > low_diff) & (high_diff > 0)
        minus_dm = (low_diff > high_diff) & (low_diff > 0)
        df['plus_dm_14'] = high_diff.where(plus_dm, 0.0).ewm(alpha=1/14, adjust=False).mean()
        df['minus_dm_14'] = low_diff.where(minus_dm, 0.0).ewm(alpha=1/14, adjust=False).mean()
        df['plus_di_14'] = 100 * (df['plus_dm_14'] / atr)
        df['minus_di_14'] = 100 * (df['minus_dm_14'] / atr)
        di_sum = df['plus_di_14'] + df['minus_di_14']
        dx = 100 * (abs(df['plus_di_14'] - df['minus_di_14']) / di_sum.where(di_sum != 0, 1))
        df['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

        last = df.iloc[-1]
        is_bullish = last['close'] > last['EMA_50'] and last['MACD'] > last['MACD_Signal']
        is_bearish = last['close'] < last['EMA_50'] and last['MACD'] < last['MACD_Signal']
        is_strong = last['ADX_14'] > self.adx_trend_threshold
        
        if is_bullish and is_strong:
            return "Uptrend", f"Price > 50-EMA, MACD is bullish, and ADX ({last['ADX_14']:.1f}) > {self.adx_trend_threshold}."
        elif is_bearish and is_strong:
            return "Downtrend", f"Price < 50-EMA, MACD is bearish, and ADX ({last['ADX_14']:.1f}) > {self.adx_trend_threshold}."
        else:
            reason = f"Market is ranging or trend is weak (ADX: {last['ADX_14']:.1f}, Threshold: {self.adx_trend_threshold})."
            if not is_strong: reason = f"4h trend is too weak (ADX is {last['ADX_14']:.1f}, needs to be > {self.adx_trend_threshold})."
            elif is_bullish or is_bearish: reason = "MACD and EMA are not in agreement."
            return "Sideways", reason

    def _get_15m_confirmation(self, symbol: str, direction: str) -> tuple[bool, str]:
        """Analyzes the 15m timeframe for momentum confirmation."""
        logger.info(f"3/4: Seeking 15m confirmation for a {direction} on {symbol}...")
        if not self.strategy_runner: return False, "Strategy Runner not initialized."
        df = self.strategy_runner.fetcher.fetch_ohlcv(symbol, "15m", 100)
        if df is None or len(df) < 50:
            return False, "Insufficient 15m data for confirmation."

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_18'] = df['close'].ewm(span=18, adjust=False).mean()
        last = df.iloc[-1]
        
        if direction == 'long':
            rsi_ok = last['RSI_14'] > 50
            ema_ok = last['EMA_9'] > last['EMA_18']
            if rsi_ok and ema_ok:
                return True, f"Confirmed by 15m RSI ({last['RSI_14']:.1f}) and bullish momentum (9-EMA > 18-EMA)."
            else:
                return False, f"15m confirmation failed. RSI > 50: {rsi_ok} (is {last['RSI_14']:.1f}). EMA Bullish: {ema_ok}."
        elif direction == 'short':
            rsi_ok = last['RSI_14'] < 50
            ema_ok = last['EMA_9'] < last['EMA_18']
            if rsi_ok and ema_ok:
                return True, f"Confirmed by 15m RSI ({last['RSI_14']:.1f}) and bearish momentum (9-EMA < 18-EMA)."
            else:
                return False, f"15m confirmation failed. RSI < 50: {rsi_ok} (is {last['RSI_14']:.1f}). EMA Bearish: {ema_ok}."
        return False, "Invalid direction for 15m confirmation."

    def _find_market_structure_levels(self, df: pd.DataFrame) -> tuple[float, float]:
        """Finds recent swing high/low for dynamic TP."""
        recent_period = 24 * 5
        recent_df = df.tail(recent_period)
        return recent_df['high'].max(), recent_df['low'].min()

    def _generate_1h_entry_analysis(self, trading_symbol: str) -> Optional[Dict[str, Any]]:
        """Core analysis on the 1h timeframe to find a potential trade setup."""
        logger.info(f"2/4: Analyzing 1h timeframe for entry signal on {trading_symbol}...")
        if not self.strategy_runner: return {"error": "Strategy Runner not initialized."}
        strategy_results = self.strategy_runner.run_all_strategies(symbol=trading_symbol)
        df_ohlcv = self.strategy_runner.fetcher.fetch_ohlcv(trading_symbol, "1h")
        if df_ohlcv is None or len(df_ohlcv) < self.min_data_points: return {"error": "Insufficient 1h data for entry analysis."}

        ta_analyzer = TechnicalAnalyzer(df_ohlcv.copy())
        df_with_ta = ta_analyzer.generate_all_indicators()
        ta_summary = ta_analyzer.get_structured_summary()
        pattern_results = analyze_candlestick_patterns(df_with_ta.copy())
        overall_score = self._calculate_signal_score(ta_summary, pattern_results, strategy_results)
        
        current_price = df_with_ta.iloc[-1].get('close')
        if abs(overall_score) < self.signal_threshold:
            return {"error": f"1h score ({overall_score:.3f}) did not meet threshold ({self.signal_threshold})."}

        signal_type = "long" if overall_score > 0 else "short"
        entry_price = current_price
        atr_val = df_with_ta.iloc[-1].get('ATR_14', current_price * 0.02)
        swing_high, swing_low = self._find_market_structure_levels(df_with_ta)
        
        if signal_type == "long":
            stop_loss = entry_price - atr_val * self.atr_multiplier_sl
            take_profit_1 = entry_price + (swing_high - entry_price) * 0.5
            take_profit_2 = swing_high
        else: # short
            stop_loss = entry_price + atr_val * self.atr_multiplier_sl
            take_profit_1 = entry_price - (entry_price - swing_low) * 0.5
            take_profit_2 = swing_low

        if (signal_type == 'long' and take_profit_1 <= entry_price) or \
           (signal_type == 'short' and take_profit_1 >= entry_price):
             risk = abs(entry_price - stop_loss)
             take_profit_1 = entry_price + risk * 1.5 if signal_type == 'long' else entry_price - risk * 1.5
             take_profit_2 = entry_price + risk * 3.0 if signal_type == 'long' else entry_price - risk * 3.0

        return {
            "signal_type": signal_type.upper(), "score": round(overall_score, 3),
            "entry_price": round(entry_price, 5), "stop_loss": round(stop_loss, 5),
            "take_profit_1": round(take_profit_1, 5), "take_profit_2": round(take_profit_2, 5),
            "reason_1h": f"1h score ({overall_score:.2f}) met threshold. Strategies, TA, and patterns align."
        }

    def generate_mft_signal(self, trading_symbol: str) -> Optional[Dict[str, Any]]:
        """Orchestrates the MTF analysis and returns a signal or a detailed HOLD reason."""
        logger.info(f"--- Starting Multi-Timeframe Signal Generation for {trading_symbol} ---")

        is_active, active_reason = self._manage_state(action='check', symbol=trading_symbol)
        if is_active:
            return {"signal_type": "HOLD", "reason": active_reason}

        trend_4h, reason_4h = self._determine_4h_trend(trading_symbol)
        if trend_4h not in ["Uptrend", "Downtrend"]:
            return {"signal_type": "HOLD", "reason": f"4h Filter: {reason_4h}"}
        
        intended_direction = "long" if trend_4h == "Uptrend" else "short"
        signal_1h = self._generate_1h_entry_analysis(trading_symbol)
        
        if not signal_1h or signal_1h.get("error"):
            reason = signal_1h.get("error", "No entry signal found on 1h timeframe.")
            return {"signal_type": "HOLD", "reason": f"1h Filter: {reason}"}
        
        if signal_1h['signal_type'].lower() != intended_direction:
            return {"signal_type": "HOLD", "reason": f"Filter Conflict: 1h signal ({signal_1h['signal_type']}) conflicts with 4h trend ({trend_4h})."}
        
        logger.info(f"Signal alignment PASSED: 4h Trend ({trend_4h}) and 1h Signal ({signal_1h['signal_type']}) match.")
        
        is_confirmed, reason_15m = self._get_15m_confirmation(trading_symbol, intended_direction)
        if not is_confirmed:
            return {"signal_type": "HOLD", "reason": f"15m Filter: {reason_15m}"}
        
        logger.info(f"4/4: SUCCESS! Signal generation process complete for {trading_symbol}.")
        
        signal_timestamp = datetime.now(timezone.utc)
        final_reason = f"4H Trend: {reason_4h} | 1H Entry: {signal_1h['reason_1h']} | 15M Confirm: {reason_15m}"
        
        final_signal = {
            "asset": trading_symbol, "timestamp_utc": signal_timestamp.isoformat(),
            "signal_type": signal_1h['signal_type'], "timeframe": "4h/1h/15m", "confidence": "High",
            "score": signal_1h['score'], "entry_price": signal_1h['entry_price'], "stop_loss": signal_1h['stop_loss'],
            "take_profit_1": signal_1h['take_profit_1'], "take_profit_2": signal_1h['take_profit_2'],
            "suggested_leverage": "5-10x", "valid_until": (signal_timestamp + timedelta(hours=4)).isoformat(),
            "reason": final_reason
        }
        
        self._manage_state(action='add', symbol=trading_symbol, signal=final_signal)
        return final_signal

def run_signal_generation_example():
    """Interactive example to use the new MTF signal generator."""
    logger.info("--- Initializing Multi-Timeframe Signal Generator for Interactive Run ---")
    
    try:
        signal_gen = SignalGenerator()
    except Exception as e:
        logger.critical(f"Could not initialize SignalGenerator: {e}")
        return

    while True:
        print("\n" + "="*50)
        asset_input = input("Enter a coin symbol (e.g., BTC, SOL/USDT) or 'exit' to quit: ").strip().upper()
        
        if not asset_input: continue
        if asset_input.lower() == 'exit': break

        logger.info(f"--- Generating MTF signal for {asset_input} ---")
        try:
            signal = signal_gen.generate_mft_signal(trading_symbol=asset_input)
            
            if signal and signal.get("signal_type") != "HOLD":
                print(f"SUCCESS: Signal Generated for {asset_input}")
                print(json.dumps(signal, indent=4))
            elif signal: # It's a HOLD signal with a reason
                print(f"INFO: No actionable signal for {asset_input}. Reason: {signal.get('reason', 'Unknown')}")
            else: # Should not happen, but a failsafe
                print(f"INFO: No signal generated for {asset_input} (HOLD or error). Check logs.")
                
        except Exception as e:
            logger.error(f"CRITICAL ERROR during signal generation for {asset_input}: {e}", exc_info=True)
            print(f"ERROR: An exception occurred while processing {asset_input}. See logs.")
        print("="*50)

    logger.info("--- Signal Generation Run Complete ---")
    print("Program finished. Goodbye!")


if __name__ == "__main__":
    run_signal_generation_example()