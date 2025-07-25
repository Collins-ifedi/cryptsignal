# File: src/utils/credentials_loader.py
import sys
import os
import yaml
from pathlib import Path
import logging

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_credentials(filepath: str = Path(PROJECT_ROOT)/ "config"/ "credentials.yaml") -> dict:
    """
    Load API credentials from a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Dictionary of credentials.
    """
    try:
        config_path = Path(filepath)
        if not config_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {filepath}")

        with open(config_path, "r") as file:
            credentials = yaml.safe_load(file)

        logger.info(f"Credentials loaded successfully from {filepath}")
        return credentials

    except Exception as e:
        logger.exception(f"Failed to load credentials: {e}")
        raise