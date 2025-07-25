import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Create log directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOG_DIR / "app.log"

def setup_logger(log_file: str = str(DEFAULT_LOG_FILE), level: int = logging.INFO):
    """
    Sets up the root logger configuration for the application.
    Should be called once at app startup.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # File handler with log rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-specific logger using the root configuration.
    """
    return logging.getLogger(name)