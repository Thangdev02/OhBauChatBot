"""
Utility functions for validating and saving uploaded images.
"""

from PIL import Image
import io
import os

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def validate_image_bytes(data: bytes) -> bool:
    """Check if the provided bytes represent a valid image."""
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def save_image_bytes(data: bytes, path: str) -> None:
    """Save image bytes to a specific path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def pil_from_bytes(data: bytes) -> Image.Image:
    """Convert bytes into a Pillow Image object."""
    return Image.open(io.BytesIO(data)).convert("RGB")
