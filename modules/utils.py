"""General file, JSON, logging, and image utility helpers."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def setup_logging() -> None:
    """Configure a small, readable logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def ensure_directories(paths: list[Path]) -> None:
    """Create project data directories if they are missing."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    """Return a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "asset"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save a JSON object with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def json_canonical_bytes(data: dict[str, Any]) -> bytes:
    """Serialize JSON in a stable way for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR uint8, raising a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Write a BGR uint8 image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Could not save image: {path}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR images from OpenCV into RGB images for display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_records(records_dir: Path) -> list[Path]:
    """Return stored record JSON files in display order."""
    return sorted(records_dir.glob("*.json"))
