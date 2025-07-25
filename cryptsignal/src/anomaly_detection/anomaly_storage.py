import os
import sys
import json
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Directory to store anomaly data
ANOMALY_DIR = "data/anomalies"

def ensure_directory_exists(path):
    """
    Ensures that the directory exists. If not, it creates it.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created anomaly storage directory: {path}")
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")

def get_anomaly_file_path(symbol: str) -> str:
    """
    Returns the file path for storing anomalies for a given symbol.
    """
    # Normalize the symbol to lowercase and replace slashes to avoid issues with file naming
    symbol = symbol.replace("/", "_").lower()
    return os.path.join(ANOMALY_DIR, f"{symbol}_anomalies.json")

def store_anomalies(symbol: str, anomalies: list):
    """
    Store a list of anomalies for a given symbol to a JSON file.
    If the file exists, it appends the new anomalies.
    """
    try:
        ensure_directory_exists(ANOMALY_DIR)
        file_path = get_anomaly_file_path(symbol)

        # Ensure anomalies have timestamps
        timestamped_anomalies = [{
            **anomaly,
            "detected_at": datetime.utcnow().isoformat()
        } for anomaly in anomalies]

        if os.path.exists(file_path):
            # Load the existing anomalies and append the new ones
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        combined = existing_data + timestamped_anomalies

        # Write the combined anomalies back to the file
        with open(file_path, "w") as f:
            json.dump(combined, f, indent=4)

        logger.info(f"Stored {len(timestamped_anomalies)} anomalies to {file_path}")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error storing anomalies for {symbol}: {e}")

def load_anomalies(symbol: str) -> list:
    """
    Load stored anomalies for a given symbol.
    """
    try:
        file_path = get_anomaly_file_path(symbol)
        if not os.path.exists(file_path):
            logger.info(f"No anomaly file found for {symbol}. Returning empty list.")
            return []

        with open(file_path, "r") as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading anomalies for {symbol}: {e}")
        return []

def cleanup_old_anomalies(symbol: str, days_threshold: int = 30):
    """
    Cleanup anomalies older than the specified threshold in days.
    The anomalies are removed from the file and from memory.
    """
    try:
        file_path = get_anomaly_file_path(symbol)
        if not os.path.exists(file_path):
            logger.info(f"No anomaly file found for {symbol}. No cleanup needed.")
            return []

        # Load existing anomalies
        with open(file_path, "r") as f:
            anomalies = json.load(f)

        threshold_time = datetime.utcnow() - timedelta(days=days_threshold)

        # Filter anomalies to keep only recent ones
        recent_anomalies = [
            anomaly for anomaly in anomalies
            if datetime.fromisoformat(anomaly["detected_at"]) >= threshold_time
        ]

        # Save the filtered anomalies back to the file
        with open(file_path, "w") as f:
            json.dump(recent_anomalies, f, indent=4)

        logger.info(f"Cleaned up anomalies for {symbol}. Retained {len(recent_anomalies)} anomalies.")
        return recent_anomalies

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up anomalies for {symbol}: {e}")
        return []

def get_all_anomalies() -> dict:
    """
    Returns all anomalies across all symbols.
    """
    all_anomalies = {}
    try:
        ensure_directory_exists(ANOMALY_DIR)
        
        # Iterate over all files in the anomaly directory
        for filename in os.listdir(ANOMALY_DIR):
            if filename.endswith("_anomalies.json"):
                symbol = filename.split("_")[0]  # Get the symbol from the filename
                file_path = os.path.join(ANOMALY_DIR, filename)

                # Load anomalies for this symbol
                with open(file_path, "r") as f:
                    anomalies = json.load(f)
                    all_anomalies[symbol] = anomalies

        return all_anomalies

    except Exception as e:
        logger.error(f"Error retrieving all anomalies: {e}")
        return {}