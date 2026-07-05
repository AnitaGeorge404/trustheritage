"""Image loading and preprocessing routines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_SIZE
from modules.utils import read_image


def preprocess_image(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize and normalize an image to BGR uint8 for stable comparison."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty or invalid.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0, 255).astype(np.uint8)


def load_and_preprocess(path: Path, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Read an image from disk and preprocess it."""
    return preprocess_image(read_image(path), size=size)


def decode_uploaded_image(file_bytes: bytes, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Decode uploaded image bytes from Streamlit and preprocess them."""
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return preprocess_image(image, size=size)
