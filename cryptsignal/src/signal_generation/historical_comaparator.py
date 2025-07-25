# File: src/signal_generation/historical_comparator.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from utils.logger import get_logger
from data.market_data import MarketFetcher

logger = get_logger("HistoricalComparator", log_file="logs/historical_comparator/comparator.log")

class HistoricalComparator:
    def __init__(self, prediction_file: str):
        self.prediction_file = prediction_file
        self.market_fetcher = MarketFetcher()
        self.predictions_df = self._load_predictions()

    def _load_predictions(self) -> pd.DataFrame:
        if not os.path.exists(self.prediction_file):
            logger.error(f"Prediction file not found: {self.prediction_file}")
            raise FileNotFoundError(f"Missing prediction file: {self.prediction_file}")
        df = pd.read_csv(self.prediction_file)
        if not {'symbol', 'timestamp', 'predicted_price'}.issubset(df.columns):
            logger.error("Invalid prediction file format.")
            raise ValueError("Prediction file must contain 'symbol', 'timestamp', 'predicted_price'")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def fetch_actuals(self) -> pd.DataFrame:
        actuals = []
        for symbol in self.predictions_df['symbol'].unique():
            logger.info(f"Fetching actual price data for {symbol}")
            try:
                start_time = self.predictions_df[self.predictions_df['symbol'] == symbol]['timestamp'].min()
                end_time = self.predictions_df[self.predictions_df['symbol'] == symbol]['timestamp'].max()
                historical_df = self.market_fetcher.fetch_price_data(
                    symbol=symbol,
                    interval='1h',
                    start_time=start_time,
                    end_time=end_time
                )
                historical_df['symbol'] = symbol
                actuals.append(historical_df[['timestamp', 'close', 'symbol']])
            except Exception as e:
                logger.error(f"Failed to fetch actuals for {symbol}: {e}")
        return pd.concat(actuals, ignore_index=True)

    def compare(self) -> pd.DataFrame:
        actuals_df = self.fetch_actuals()
        merged = pd.merge(
            self.predictions_df,
            actuals_df,
            on=['timestamp', 'symbol'],
            how='inner'
        )
        merged['error'] = merged['predicted_price'] - merged['close']
        merged['abs_error'] = merged['error'].abs()
        merged['squared_error'] = merged['error'] ** 2
        merged['directional_accuracy'] = np.sign(merged['predicted_price'].diff()) == np.sign(merged['close'].diff())

        metrics = {
            "MAE": merged['abs_error'].mean(),
            "RMSE": np.sqrt(merged['squared_error'].mean()),
            "Directional Accuracy": merged['directional_accuracy'].mean()
        }

        logger.info("Comparison completed")
        logger.info(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, Directional Accuracy: {metrics['Directional Accuracy']:.2%}")
        self._save_comparison(merged)
        return merged

    def _save_comparison(self, df: pd.DataFrame):
        output_path = f"logs/historical_comparator/comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved comparison results to {output_path}")