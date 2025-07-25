import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import requests
import yaml

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Project-Level Imports & Local Credential Loader ---
try:
    from utils.logger import get_logger
    logger = get_logger("ImageIO")

    def load_credentials() -> Dict:
        config_path = Path(PROJECT_ROOT) / "config" / "credentials.yaml"
        if not config_path.is_file():
            logger.warning(f"Credentials file not found: {config_path}")
            return {}
        try:
            with config_path.open('r') as f:
                credentials = yaml.safe_load(f)
                return credentials if isinstance(credentials, dict) else {}
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading credentials file: {e}")
            return {}
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
    logger = logging.getLogger("ImageIO_Fallback")

    def load_credentials() -> Dict:
        logger.warning("Using dummy credentials loader.")
        return {}

# --- Configuration ---
def get_hf_api_key() -> Optional[str]:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key:
        logger.info("Hugging Face API key loaded from environment.")
        return api_key

    credentials = load_credentials()
    api_key = credentials.get("HUGGINGFACE_API_KEY")
    if api_key:
        logger.info("Hugging Face API key loaded from credentials file.")
    else:
        logger.error("Hugging Face API key not found.")
    return api_key

# --- Image Generation Endpoint ---
TEXT_TO_IMAGE_ENDPOINT = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

class ImageIO:
    def __init__(self, hf_token: Optional[str] = None, timeout: int = 45):
        self.hf_token = hf_token or get_hf_api_key()
        if not self.hf_token:
            raise ValueError("Hugging Face API key is required but missing.")

        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        logger.info("✅ ImageIO initialized (Stable Diffusion XL for image generation).")

    def generate_image(self, prompt: str, output_file: Path) -> Optional[Path]:
        """
        Generates an image from a text prompt.
        """
        if not prompt.strip():
            logger.warning("⚠️ Image generation skipped: Prompt is empty.")
            return None

        payload = {"inputs": prompt}
        logger.info(f"🎨 Generating image with prompt: '{prompt[:50]}...'")

        try:
            response = self._session.post(TEXT_TO_IMAGE_ENDPOINT, json=payload, timeout=self.timeout)
            if response.status_code == 503:
                logger.warning("Image generation model is loading, retrying...")
                time.sleep(15)
                response = self._session.post(TEXT_TO_IMAGE_ENDPOINT, json=payload, timeout=self.timeout)

            response.raise_for_status()

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(response.content)

            logger.info(f"✅ Image saved: {output_file}")
            return output_file
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Image generation error: {e}")
        return None

    def close(self):
        self._session.close()
        logger.info("ImageIO HTTP session closed.")


def main():
    """
    Simple test for ImageIO image generation.
    """
    print("--- Image Generation Test ---")
    image_io = None
    try:
        image_io = ImageIO()
        output_dir = Path("./generated_media")
        output_dir.mkdir(exist_ok=True)

        # Ask user for a prompt
        user_prompt = input("\nEnter a prompt for image generation: ").strip()
        if not user_prompt:
            print("\n⚠️ No prompt entered. Exiting.")
            return

        # Generate image
        image_output_path = output_dir / "generated_image.png"
        generated_image = image_io.generate_image(user_prompt, image_output_path)
        if generated_image:
            print(f"\n✅ Image generated successfully: {generated_image}")
        else:
            print("\n❌ Image generation failed.")

    except ValueError as e:
        print(f"Initialization Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    finally:
        if image_io:
            image_io.close()
        print("\n--- Test Complete ---")


if __name__ == "__main__":
    main()