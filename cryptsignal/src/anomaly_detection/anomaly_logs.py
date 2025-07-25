import os
import logging
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

# Directory to store anomaly logs
ANOMALY_LOG_DIR = "logs/anomalies"

def setup_anomaly_logger(symbol: str) -> logging.Logger:
    """
    Setup and return a logger that logs anomaly-related events for a specific symbol.
    Logs are rotated at midnight and archived for 30 days.
    """
    try:
        os.makedirs(ANOMALY_LOG_DIR, exist_ok=True)
        
        # Clean symbol name to avoid invalid characters in filenames
        symbol_clean = symbol.replace("/", "_").lower()
        log_file = os.path.join(ANOMALY_LOG_DIR, f"{symbol_clean}.log")

        # Creating or getting the logger for the specific symbol
        logger = logging.getLogger(f"AnomalyLogger_{symbol_clean}")
        logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if the logger already has one
        if not logger.handlers:
            handler = TimedRotatingFileHandler(
                log_file, 
                when="midnight", 
                interval=1, 
                backupCount=30, 
                encoding="utf-8"
            )
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger
    
    except Exception as e:
        # Log errors related to logger setup
        logger = logging.getLogger("AnomalyLogger_General")
        logger.setLevel(logging.ERROR)
        logger.error(f"Error setting up logger for {symbol}: {e}")
        raise

def log_anomalies(symbol: str, anomalies: list):
    """
    Log the detected anomalies for a specific symbol. Each anomaly is logged with a type and description.
    """
    try:
        logger = setup_anomaly_logger(symbol)

        # Log each anomaly
        for anomaly in anomalies:
            anomaly_type = anomaly.get('type', 'Unknown')
            description = anomaly.get('description', 'No description provided')
            logger.info(f"Anomaly Detected - Type: {anomaly_type} | Description: {description}")
    
    except Exception as e:
        # General error logging in case of failure in anomaly logging
        general_logger = logging.getLogger("AnomalyLogger_General")
        general_logger.setLevel(logging.ERROR)
        general_logger.error(f"Error logging anomalies for {symbol}: {e}")

def cleanup_old_anomaly_logs():
    """
    Cleanup anomaly logs older than a certain number of days.
    By default, older logs than 30 days will be deleted.
    """
    try:
        threshold_time = datetime.utcnow() - timedelta(days=30)

        # Check all log files in the anomaly log directory
        for filename in os.listdir(ANOMALY_LOG_DIR):
            file_path = os.path.join(ANOMALY_LOG_DIR, filename)
            if os.path.isfile(file_path):
                # Get the timestamp of the log file (assuming the filename format includes a date)
                file_mod_time = datetime.utcfromtimestamp(os.path.getmtime(file_path))
                
                # Delete the file if it is older than the threshold
                if file_mod_time < threshold_time:
                    os.remove(file_path)
                    logger = logging.getLogger("AnomalyLogger_General")
                    logger.info(f"Deleted old log file: {file_path}")
                    
    except Exception as e:
        # General error logging for the cleanup process
        general_logger = logging.getLogger("AnomalyLogger_General")
        general_logger.setLevel(logging.ERROR)
        general_logger.error(f"Error during old anomaly log cleanup: {e}")

def get_anomaly_logs(symbol: str) -> list:
    """
    Retrieves the logs for a specific symbol from the log files.
    This method assumes that the logs are stored in text format and can be read.
    """
    try:
        symbol_clean = symbol.replace("/", "_").lower()
        log_file = os.path.join(ANOMALY_LOG_DIR, f"{symbol_clean}.log")

        # Check if the log file exists
        if not os.path.exists(log_file):
            logger = logging.getLogger("AnomalyLogger_General")
            logger.info(f"No log file found for symbol: {symbol}")
            return []

        # Read the log file
        with open(log_file, "r", encoding="utf-8") as file:
            logs = file.readlines()

        # Return the logs as a list of lines
        return logs

    except Exception as e:
        # Error logging for failure to retrieve logs
        general_logger = logging.getLogger("AnomalyLogger_General")
        general_logger.setLevel(logging.ERROR)
        general_logger.error(f"Error retrieving anomaly logs for {symbol}: {e}")
        return []