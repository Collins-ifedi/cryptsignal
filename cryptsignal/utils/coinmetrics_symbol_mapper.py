# utils/coinmetrics_symbol_mapper.py
import requests
import logging

ASSET_URL = "https://community-api.coinmetrics.io/v4/catalog/assets"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoinMetricsSymbolMapper")


class CoinMetricsSymbolMapper:
    def __init__(self):
        self.name_to_key = {}
        self.symbol_to_key = {}
        self._load_assets()

    def _load_assets(self):
        try:
            response = requests.get(ASSET_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Safely print one example asset to inspect structure
            if "data" in data and data["data"]:
                print("Sample Asset Item:", data["data"][0])

            for asset in data.get("data", []):
                key = asset.get("asset")  # This is the asset key used in API queries
                name = asset.get("name", "").lower()  # e.g., "Bitcoin"
                symbol = asset.get("id", "").upper()  # e.g., "BTC" or "ETH"

                if key:
                    if name:
                        self.name_to_key[name] = key
                    if symbol:
                        self.symbol_to_key[symbol] = key

        except Exception as e:
            logger.error(f"Failed to fetch CoinMetrics assets: {e}")

    def get_asset_key(self, asset_input: str) -> str:
        if not asset_input:
            return None

        normalized = asset_input.strip().lower()
        by_name = self.name_to_key.get(normalized)
        by_symbol = self.symbol_to_key.get(asset_input.strip().upper())

        if by_symbol:
            return by_symbol
        elif by_name:
            return by_name
        else:
            logger.warning(f"Could not resolve CoinMetrics asset key for '{asset_input}'")
            return None