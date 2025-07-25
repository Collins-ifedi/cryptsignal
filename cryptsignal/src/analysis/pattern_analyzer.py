import os
import sys
import logging
import argparse # For CLI argument handling
import pandas as pd
import numpy as np
from pathlib import Path

# --- Environment and Path Setup ---
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Module Imports with Corrected Path Assumptions ---
try:
    from src.data.market_data import MarketDataFetcher
except ImportError as e:
    logging.error(f"Failed to import MarketDataFetcher from src.data: {e}")
    class MarketDataFetcher: pass
    raise

# --- Import CoinSymbolMapper with updated function ---
# Assuming coin_symbol_mapper.py is in the top-level 'utils' directory
try:
    # UPDATED: Use the new get_trading_pair function which is the direct equivalent needed here.
    from utils.coin_symbol_mapper import get_trading_pair
except ImportError as e:
    logging.error(f"Failed to import get_trading_pair from coin_symbol_mapper: {e}")
    # UPDATED: Corrected the error message to point to the right path and function.
    sys.exit("Coin symbol mapper not found or outdated. Please ensure 'cryptsignal/utils/coin_symbol_mapper.py' is accessible and contains the 'get_trading_pair' function.")


# --- Logger Setup ---
logger = logging.getLogger("PatternAnalyzer")
# Basic config will be set in main block based on args

# ------------------------------
# Candlestick Pattern Functions
# ------------------------------

def ensure_series_output(result, index):
    """Helper to ensure output is a boolean pandas Series."""
    if isinstance(result, pd.Series):
        return result.fillna(False).astype(bool)
    elif isinstance(result, np.ndarray):
        return pd.Series(result, index=index).fillna(False).astype(bool)
    else:
        logger.warning(f"Unexpected type in ensure_series_output: {type(result)}. Returning False Series.")
        return pd.Series(False, index=index)

def bullish_engulfing(df):
    df = df.copy()
    prev = df.shift(1)
    result = (
        (prev['close'] < prev['open']) &
        (df['close'] > df['open']) &
        (df['open'] < prev['close']) &
        (df['close'] > prev['open'])
    )
    return result.fillna(False).astype(bool)

def bearish_engulfing(df):
    df = df.copy()
    prev = df.shift(1)
    result = (
        (prev['close'] > prev['open']) &
        (df['close'] < df['open']) &
        (df['open'] > prev['close']) &
        (df['close'] < prev['open'])
    )
    return result.fillna(False).astype(bool)

def hammer(df):
    df = df.copy()
    original_index = df.index
    body = abs(df['close'] - df['open'])
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    body_ratio = np.where(total_range > 1e-9, body / total_range, 0)
    lower_shadow_ratio = np.where(body > 1e-9, lower_shadow / body, np.inf)
    result_array = (body_ratio < 0.3) & (lower_shadow_ratio > 2.0)
    return ensure_series_output(result_array, original_index)

def inverted_hammer(df):
    df = df.copy()
    original_index = df.index
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    total_range = df['high'] - df['low']
    body_ratio = np.where(total_range > 1e-9, body / total_range, 0)
    upper_shadow_ratio = np.where(body > 1e-9, upper_shadow / body, np.inf)
    result_array = (body_ratio < 0.3) & (upper_shadow_ratio > 2.0)
    return ensure_series_output(result_array, original_index)

def shooting_star(df):
    return inverted_hammer(df)

def doji(df):
    df = df.copy()
    original_index = df.index
    total_range = df['high'] - df['low']
    body_ratio = np.where(total_range > 1e-9, abs(df['close'] - df['open']) / total_range, 0)
    result_array = body_ratio <= 0.1
    return ensure_series_output(result_array, original_index)

def gravestone_doji(df):
    df = df.copy()
    original_index = df.index
    total_range = df['high'] - df['low'] + 1e-9
    is_doji_series = doji(df)
    open_low_close = np.where(total_range > 1e-9, abs(df['open'] - df['low']) / total_range < 0.1, False)
    close_low_close = np.where(total_range > 1e-9, abs(df['close'] - df['low']) / total_range < 0.1, False)
    result_combined = is_doji_series & pd.Series(open_low_close, index=original_index) & pd.Series(close_low_close, index=original_index)
    return result_combined.fillna(False).astype(bool)

def dragonfly_doji(df):
    df = df.copy()
    original_index = df.index
    total_range = df['high'] - df['low'] + 1e-9
    is_doji_series = doji(df)
    open_high_close = np.where(total_range > 1e-9, abs(df['open'] - df['high']) / total_range < 0.1, False)
    close_high_close = np.where(total_range > 1e-9, abs(df['close'] - df['high']) / total_range < 0.1, False)
    result_combined = is_doji_series & pd.Series(open_high_close, index=original_index) & pd.Series(close_high_close, index=original_index)
    return result_combined.fillna(False).astype(bool)

def marubozu(df):
    df = df.copy()
    total_range = df['high'] - df['low'] + 1e-9
    open_low_diff = (df['open'] - df['low']) / total_range
    close_high_diff = (df['high'] - df['close']) / total_range
    open_high_diff = (df['high'] - df['open']) / total_range
    close_low_diff = (df['close'] - df['low']) / total_range
    is_bullish_marubozu = (df['close'] > df['open']) & \
                          ensure_series_output(open_low_diff < 0.05, df.index) & \
                          ensure_series_output(close_high_diff < 0.05, df.index)
    is_bearish_marubozu = (df['close'] < df['open']) & \
                          ensure_series_output(open_high_diff < 0.05, df.index) & \
                          ensure_series_output(close_low_diff < 0.05, df.index)
    result = is_bullish_marubozu | is_bearish_marubozu
    return result.fillna(False).astype(bool)

def bearish_harami(df):
    df = df.copy()
    prev = df.shift(1)
    result = (
        (prev['close'] > prev['open']) &
        (df['close'] < df['open']) &
        (df['open'] < prev['close']) & (df['close'] > prev['open']) &
        (df['high'] < prev['close']) & (df['low'] > prev['open'])
    )
    return result.fillna(False).astype(bool)

def bullish_harami(df):
    df = df.copy()
    prev = df.shift(1)
    result = (
        (prev['close'] < prev['open']) &
        (df['close'] > df['open']) &
        (df['open'] > prev['close']) & (df['close'] < prev['open']) &
        (df['high'] < prev['open']) & (df['low'] > prev['close'])
    )
    return result.fillna(False).astype(bool)

def morning_star(df):
    df = df.copy()
    original_index = df.index
    if len(df) < 3: return pd.Series(False, index=original_index)
    prev2 = df.shift(2); prev1 = df.shift(1); curr = df
    prev1_range = prev1['high'] - prev1['low']
    prev2_range = prev2['high'] - prev2['low']
    curr_range = curr['high'] - curr['low']
    cond1 = prev2['close'] < prev2['open']
    cond2 = np.where(prev2_range > 1e-9, (prev2['open'] - prev2['close']) / prev2_range > 0.6, False)
    cond3 = np.where(prev1_range > 1e-9, abs(prev1['close'] - prev1['open']) / prev1_range < 0.3, False)
    cond4 = prev1[['open', 'close']].max(axis=1) < prev2['close']
    cond5 = curr['close'] > curr['open']
    cond6 = np.where(curr_range > 1e-9, (curr['close'] - curr['open']) / curr_range > 0.6, False)
    cond7 = curr['close'] > (prev2['open'] + prev2['close']) / 2
    result_combined = ensure_series_output(cond1, original_index) & \
                      ensure_series_output(cond2, original_index) & \
                      ensure_series_output(cond3, original_index) & \
                      ensure_series_output(cond4, original_index) & \
                      ensure_series_output(cond5, original_index) & \
                      ensure_series_output(cond6, original_index) & \
                      ensure_series_output(cond7, original_index)
    return result_combined

def evening_star(df):
    df = df.copy()
    original_index = df.index
    if len(df) < 3: return pd.Series(False, index=original_index)
    prev2 = df.shift(2); prev1 = df.shift(1); curr = df
    prev1_range = prev1['high'] - prev1['low']
    prev2_range = prev2['high'] - prev2['low']
    curr_range = curr['high'] - curr['low']
    cond1 = prev2['close'] > prev2['open']
    cond2 = np.where(prev2_range > 1e-9, (prev2['close'] - prev2['open']) / prev2_range > 0.6, False)
    cond3 = np.where(prev1_range > 1e-9, abs(prev1['close'] - prev1['open']) / prev1_range < 0.3, False)
    cond4 = prev1[['open', 'close']].min(axis=1) > prev2['close']
    cond5 = curr['close'] < curr['open']
    cond6 = np.where(curr_range > 1e-9, (curr['open'] - curr['close']) / curr_range > 0.6, False)
    cond7 = curr['close'] < (prev2['open'] + prev2['close']) / 2
    result_combined = ensure_series_output(cond1, original_index) & \
                      ensure_series_output(cond2, original_index) & \
                      ensure_series_output(cond3, original_index) & \
                      ensure_series_output(cond4, original_index) & \
                      ensure_series_output(cond5, original_index) & \
                      ensure_series_output(cond6, original_index) & \
                      ensure_series_output(cond7, original_index)
    return result_combined

def piercing_line(df):
    df = df.copy()
    prev = df.shift(1)
    midpoint = (prev['open'] + prev['close']) / 2
    result = (
        (prev['close'] < prev['open']) &
        (df['close'] > df['open']) &
        (df['open'] < prev['close']) &
        (df['close'] > midpoint) &
        (df['close'] < prev['open'])
    )
    return result.fillna(False).astype(bool)

def dark_cloud_cover(df):
    df = df.copy()
    prev = df.shift(1)
    midpoint = (prev['open'] + prev['close']) / 2
    result = (
        (prev['close'] > prev['open']) &
        (df['close'] < df['open']) &
        (df['open'] > prev['close']) &
        (df['close'] < midpoint) &
        (df['close'] > prev['open'])
    )
    return result.fillna(False).astype(bool)

def three_white_soldiers(df):
    df = df.copy()
    original_index = df.index
    if len(df) < 3: return pd.Series(False, index=original_index)
    c1 = df.shift(2); c2 = df.shift(1); c3 = df
    c1_range = c1['high'] - c1['low']; c2_range = c2['high'] - c2['low']; c3_range = c3['high'] - c3['low']
    is_c1_bull = c1['close'] > c1['open']; is_c2_bull = c2['close'] > c2['open']; is_c3_bull = c3['close'] > c3['open']
    higher_closes = (c3['close'] > c2['close']) & (c2['close'] > c1['close'])
    open_in_prev_body = (c3['open'] > c2['open']) & (c3['open'] < c2['close']) & (c2['open'] > c1['open']) & (c2['open'] < c1['close'])
    c1_long = np.where(c1_range > 1e-9, ((c1['close'] - c1['open']) / c1_range) > 0.6, False)
    c2_long = np.where(c2_range > 1e-9, ((c2['close'] - c2['open']) / c2_range) > 0.6, False)
    c3_long = np.where(c3_range > 1e-9, ((c3['close'] - c3['open']) / c3_range) > 0.6, False)
    result_combined = ensure_series_output(is_c1_bull, original_index) & \
                      ensure_series_output(is_c2_bull, original_index) & \
                      ensure_series_output(is_c3_bull, original_index) & \
                      ensure_series_output(higher_closes, original_index) & \
                      ensure_series_output(open_in_prev_body, original_index) & \
                      ensure_series_output(c1_long, original_index) & \
                      ensure_series_output(c2_long, original_index) & \
                      ensure_series_output(c3_long, original_index)
    return result_combined

def three_black_crows(df):
    df = df.copy()
    original_index = df.index
    if len(df) < 3: return pd.Series(False, index=original_index)
    c1 = df.shift(2); c2 = df.shift(1); c3 = df
    c1_range = c1['high'] - c1['low']; c2_range = c2['high'] - c2['low']; c3_range = c3['high'] - c3['low']
    is_c1_bear = c1['close'] < c1['open']; is_c2_bear = c2['close'] < c2['open']; is_c3_bear = c3['close'] < c3['open']
    lower_closes = (c3['close'] < c2['close']) & (c2['close'] < c1['close'])
    open_in_prev_body = (c3['open'] < c2['open']) & (c3['open'] > c2['close']) & (c2['open'] < c1['open']) & (c2['open'] > c1['close'])
    c1_long = np.where(c1_range > 1e-9, ((c1['open'] - c1['close']) / c1_range) > 0.6, False)
    c2_long = np.where(c2_range > 1e-9, ((c2['open'] - c2['close']) / c2_range) > 0.6, False)
    c3_long = np.where(c3_range > 1e-9, ((c3['open'] - c3['close']) / c3_range) > 0.6, False)
    result_combined = ensure_series_output(is_c1_bear, original_index) & \
                      ensure_series_output(is_c2_bear, original_index) & \
                      ensure_series_output(is_c3_bear, original_index) & \
                      ensure_series_output(lower_closes, original_index) & \
                      ensure_series_output(open_in_prev_body, original_index) & \
                      ensure_series_output(c1_long, original_index) & \
                      ensure_series_output(c2_long, original_index) & \
                      ensure_series_output(c3_long, original_index)
    return result_combined

def spinning_top(df):
    df = df.copy()
    original_index = df.index
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low'] + 1e-9
    body_ratio = body / total_range
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    result_array = (body_ratio < 0.3) & (upper_shadow > body) & (lower_shadow > body)
    return ensure_series_output(result_array, original_index)

def tweezer_top(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    high_match = np.where(
        (prev['high'] + df['high']) / 2 > 1e-9,
        abs(prev['high'] - df['high']) / ((prev['high'] + df['high']) / 2) < 0.001,
        abs(prev['high'] - df['high']) < 1e-9
    )
    result_combined = ensure_series_output(high_match, original_index) & \
                      ensure_series_output(prev['close'] > prev['open'], original_index) & \
                      ensure_series_output(df['close'] < df['open'], original_index)
    return result_combined

def tweezer_bottom(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    low_match = np.where(
        (prev['low'] + df['low']) / 2 > 1e-9,
        abs(prev['low'] - df['low']) / ((prev['low'] + df['low']) / 2) < 0.001,
        abs(prev['low'] - df['low']) < 1e-9
    )
    result_combined = ensure_series_output(low_match, original_index) & \
                      ensure_series_output(prev['close'] < prev['open'], original_index) & \
                      ensure_series_output(df['close'] > df['open'], original_index)
    return result_combined

def belt_hold(df):
    df = df.copy()
    original_index = df.index
    body = df['close'] - df['open']
    total_range = df['high'] - df['low'] + 1e-9
    is_bullish = df['close'] > df['open']
    opens_near_low = np.where(total_range > 1e-9, (df['open'] - df['low']) / total_range < 0.05, False)
    closes_near_high = np.where(total_range > 1e-9, (df['high'] - df['close']) / total_range < 0.05, False)
    is_long_body = np.where(total_range > 1e-9, (body / total_range) > 0.7, False)
    result_combined = ensure_series_output(is_bullish, original_index) & \
                      ensure_series_output(opens_near_low, original_index) & \
                      ensure_series_output(closes_near_high, original_index) & \
                      ensure_series_output(is_long_body, original_index)
    return result_combined

def counterattack(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    close_match = np.where(
        (prev['close'] + df['close']) / 2 > 1e-9,
        abs(prev['close'] - df['close']) / ((prev['close'] + df['close']) / 2) < 0.001,
        abs(prev['close'] - df['close']) < 1e-9
    )
    result_combined = ensure_series_output(prev['close'] < prev['open'], original_index) & \
                      ensure_series_output(df['close'] > df['open'], original_index) & \
                      ensure_series_output(close_match, original_index)
    return result_combined

def on_neck(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    low_match = np.where(
        (df['close'] + prev['low']) / 2 > 1e-9,
        abs(df['close'] - prev['low']) / ((df['close'] + prev['low']) / 2) < 0.005,
        abs(df['close'] - prev['low']) < 1e-9
    )
    result_combined = ensure_series_output(prev['close'] < prev['open'], original_index) & \
                      ensure_series_output(df['close'] > df['open'], original_index) & \
                      ensure_series_output(df['open'] < prev['low'], original_index) & \
                      ensure_series_output(low_match, original_index)
    return result_combined

def above_neck(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    closes_above_prev_low = df['close'] > prev['low']
    closes_below_prev_close = df['close'] < prev['close']
    result_combined = ensure_series_output(prev['close'] < prev['open'], original_index) & \
                      ensure_series_output(df['close'] > df['open'], original_index) & \
                      ensure_series_output(df['open'] < prev['low'], original_index) & \
                      ensure_series_output(closes_above_prev_low, original_index) & \
                      ensure_series_output(closes_below_prev_close, original_index)
    return result_combined

def below_neck(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    result_combined = ensure_series_output(prev['close'] < prev['open'], original_index) & \
                      ensure_series_output(df['close'] < df['open'], original_index) & \
                      ensure_series_output(df['open'] < prev['low'], original_index) & \
                      ensure_series_output(df['close'] < prev['low'], original_index)
    return result_combined

def kicker(df):
    df = df.copy()
    original_index = df.index
    prev = df.shift(1)
    result_combined = ensure_series_output(prev['close'] < prev['open'], original_index) & \
                      ensure_series_output(df['close'] > df['open'], original_index) & \
                      ensure_series_output(df['open'] > prev['open'], original_index)
    return result_combined

def tweezer_bottom_reversal(df):
    df = df.copy()
    original_index = df.index
    if len(df) < 3: return pd.Series(False, index=original_index)
    c1 = df.shift(2); c2 = df.shift(1); c3 = df
    is_c1_bear = c1['close'] < c1['open']
    is_c2_bull = c2['close'] > c2['open']
    is_c3_bull = c3['close'] > c3['open']
    low_match = np.where(
        (c1['low'] + c2['low']) / 2 > 1e-9,
        abs(c1['low'] - c2['low']) / ((c1['low'] + c2['low']) / 2) < 0.001,
        abs(c1['low'] - c2['low']) < 1e-9
    )
    confirmation = c3['close'] > c2['high']
    result_combined = ensure_series_output(is_c1_bear, original_index) & \
                      ensure_series_output(is_c2_bull, original_index) & \
                      ensure_series_output(low_match, original_index) & \
                      ensure_series_output(is_c3_bull, original_index) & \
                      ensure_series_output(confirmation, original_index)
    return result_combined


# ---------------------
# Pattern Registration
# ---------------------
pattern_funcs = {
    "Bullish Engulfing": bullish_engulfing, "Bearish Engulfing": bearish_engulfing,
    "Hammer": hammer, "Inverted Hammer": inverted_hammer, "Shooting Star": shooting_star,
    "Doji": doji, "Gravestone Doji": gravestone_doji, "Dragonfly Doji": dragonfly_doji,
    "Marubozu": marubozu, "Bearish Harami": bearish_harami, "Bullish Harami": bullish_harami,
    "Morning Star": morning_star, "Evening Star": evening_star,
    "Piercing Line": piercing_line, "Dark Cloud Cover": dark_cloud_cover,
    "Three White Soldiers": three_white_soldiers, "Three Black Crows": three_black_crows,
    "Spinning Top": spinning_top, "Tweezer Top": tweezer_top, "Tweezer Bottom": tweezer_bottom,
    "Belt Hold": belt_hold, "Counterattack": counterattack, "On Neck": on_neck,
    "Above Neck": above_neck, "Below Neck": below_neck, "Kicker": kicker,
    "Tweezer Bottom Reversal": tweezer_bottom_reversal,
}

# ---------------------
# Pattern Analysis and Prediction
# ---------------------
def analyze_patterns(df):
    if df is None or df.empty:
        logger.warning("Pattern analysis received empty DataFrame.")
        return { "symbol": "Unknown", "latest_patterns": [], "sentiment": "Neutral", "bullish_signals": 0, "bearish_signals": 0, "total_patterns_checked": len(pattern_funcs), "last_timestamp": None }
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.dropna(subset=[df.index.name])
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if df.empty: raise ValueError("DataFrame empty after cleaning/conversion.")
    except Exception as e:
        logger.error(f"Error preparing DataFrame for pattern analysis: {e}")
        return { "symbol": df.attrs.get("symbol", "Unknown"), "latest_patterns": [], "sentiment": "Neutral", "bullish_signals": 0, "bearish_signals": 0, "total_patterns_checked": len(pattern_funcs), "last_timestamp": None }
    signals = []
    logger.debug(f"Analyzing {len(df)} candles for {len(pattern_funcs)} patterns.")
    for name, func in pattern_funcs.items():
        try:
            matches = func(df)
            if not isinstance(matches, pd.Series):
                logger.warning(f"Pattern function '{name}' did not return a Series. Type: {type(matches)}. Skipping.")
                continue
            if not pd.api.types.is_bool_dtype(matches):
                 logger.warning(f"Pattern function '{name}' did not return a boolean Series. Dtype: {matches.dtype}. Skipping.")
                 continue
            if matches.any():
                matched_indices = df.index[matches]
                for idx in matched_indices:
                     signals.append((pd.Timestamp(idx), name))
        except Exception as e:
            logger.error(f"Error applying pattern '{name}': {type(e).__name__} - {e}", exc_info=False)
            continue
    signals.sort(key=lambda x: x[0])
    recent_signals = []
    last_candle_time = None
    if not df.empty:
        last_candle_time = df.index[-1]
        if signals:
            lookback_period = min(5, len(df))
            start_time = df.index[-lookback_period]
            recent_signals = [s for s in signals if s[0] >= start_time]
            logger.debug(f"Found {len(recent_signals)} signals in the last {lookback_period} candles.")
    else:
        logger.warning("Cannot determine last candle time or lookback period as DataFrame is empty.")
    bullish_keywords = ["Bullish", "Hammer", "Morning", "Piercing", "Three White", "Dragonfly", "Tweezer Bottom", "Belt Hold", "Counterattack", "Kicker"]
    bearish_keywords = ["Bearish", "Shooting", "Evening", "Dark", "Three Black", "Gravestone", "Tweezer Top", "On Neck", "Above Neck", "Below Neck"]
    bullish_count = sum(1 for _, name in recent_signals if any(k in name for k in bullish_keywords))
    bearish_count = sum(1 for _, name in recent_signals if any(k in name for k in bearish_keywords))
    sentiment = "Neutral"
    if bullish_count > bearish_count: sentiment = "Bullish"
    elif bearish_count > bullish_count: sentiment = "Bearish"
    latest_patterns_on_last_candle = []
    if last_candle_time:
        latest_patterns_on_last_candle = [name for ts, name in signals if ts == last_candle_time]
    logger.info(f"Pattern Analysis Result: Sentiment={sentiment}, Bull={bullish_count}, Bear={bearish_count}, Latest={latest_patterns_on_last_candle}")
    return {
        "symbol": df.attrs.get("symbol", "Unknown"), "latest_patterns": latest_patterns_on_last_candle,
        "sentiment": sentiment, "bullish_signals": bullish_count, "bearish_signals": bearish_count,
        "total_patterns_checked": len(pattern_funcs),
        "last_timestamp": last_candle_time.isoformat() if pd.notna(last_candle_time) else None
    }


# ---------------------
# Runner Function
# ---------------------
def run_analysis(symbol="BTC/USDT", interval="1h", limit=100):
    # UPDATED: The 'exchange' parameter is removed as the fetcher now searches automatically.
    
    # Resolve the identifier (e.g., "Bitcoin") to a full trading pair (e.g., "BTC/USDT")
    resolved_symbol = get_trading_pair(symbol)
    if not resolved_symbol:
        logger.warning(f"Could not resolve symbol for '{symbol}' using the mapper. Attempting to use original symbol directly.")
        if '/USDT' in symbol.upper():
            final_symbol = symbol.upper()
        else:
            final_symbol = f"{symbol.upper()}/USDT"
        logger.info(f"Using fallback symbol: {final_symbol}")
    else:
        final_symbol = resolved_symbol
        logger.info(f"Resolved '{symbol}' to '{final_symbol}' using mapper.")
    
    logger.info(f"Running pattern analysis for {final_symbol} across available exchanges ({interval}, limit {limit})...")
    
    try:
        # Initialize the fetcher. It no longer needs an exchange name.
        fetcher = MarketDataFetcher()
        
        # UPDATED: Call fetch_ohlcv using 'identifier' as the parameter name for the symbol.
        df = fetcher.fetch_ohlcv(identifier=final_symbol, timeframe=interval, limit=limit)
        
        if df is not None and not df.empty:
            df.attrs["symbol"] = final_symbol
            logger.info(f"Fetched {len(df)} candles. Running analysis...")
            analysis_result = analyze_patterns(df)
            return analysis_result
        else:
            logger.error(f"Failed to fetch OHLCV data for {final_symbol}. Cannot run analysis.")
            return None
    except NameError as ne:
         logger.error(f"NameError during analysis setup: {ne}.", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Error during pattern analysis execution for {final_symbol}: {e}", exc_info=True)
        return None

# ---------------------
# Main Execution Block
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candlestick Pattern Analyzer")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol (e.g., BTC/USDT, ETH/BTC, or a coin name like 'Solana')")
    parser.add_argument("--interval", type=str, default="1h", help="Timeframe/interval (e.g., 15m, 1h, 4h, 1d)")
    # UPDATED: Removed the --exchange argument as it's no longer used by the MarketDataFetcher.
    parser.add_argument("--limit", type=int, default=100, help="Number of candles to fetch and analyze")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    args = parser.parse_args()

    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logger.setLevel(log_level)
    try:
        # Set log levels for other modules if they exist
        logging.getLogger("MarketDataFetcher").setLevel(log_level)
        logging.getLogger("CoinSymbolMapper").setLevel(log_level)
    except:
        pass

    print("--- Pattern Analyzer (CLI Mode) ---")
    # UPDATED: Removed Exchange from the printout.
    print(f"Symbol:   {args.symbol}\nInterval: {args.interval}\nLimit:    {args.limit}\nLog Level:{args.loglevel.upper()}")
    print("-" * 30)

    # UPDATED: The call to run_analysis no longer includes the 'exchange' argument.
    result = run_analysis(symbol=args.symbol, interval=args.interval, limit=args.limit)

    if result:
        print("\n--- Pattern Analysis Results ---")
        print(f"Symbol: {result.get('symbol')}")
        print(f"Last Timestamp: {result.get('last_timestamp')}")
        print(f"Sentiment (Last ~5 candles): {result.get('sentiment')}")
        print(f"Recent Bullish Signals: {result.get('bullish_signals')}")
        print(f"Recent Bearish Signals: {result.get('bearish_signals')}")
        print(f"Patterns on Last Candle: {result.get('latest_patterns')}")
        print(f"Total Patterns Checked: {result.get('total_patterns_checked')}")
    else:
        print("\nPattern analysis failed or returned no data. Check logs for details.")

    print("\n[Program finished]")