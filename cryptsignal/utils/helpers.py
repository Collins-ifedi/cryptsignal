import sys
import os
import yaml
from pathlib import Path

def load_config(path=None):
    """
    Load the YAML config file from the specified path.
    Defaults to the standard project config path if not provided.
    """
    if path is None:
        # Default path inside the project directory
        path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)