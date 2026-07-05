"""Shared configuration for the TrustHeritage research prototype."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDS_DIR = DATA_DIR / "records"
ARCHIVED_IMAGES_DIR = DATA_DIR / "archived_images"
WATERMARKED_IMAGES_DIR = DATA_DIR / "watermarked_images"
SUSPECT_UPLOADS_DIR = DATA_DIR / "suspect_uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"

IMAGE_SIZE = (512, 512)
WATERMARK_SIZE = 32
WATERMARK_STRENGTH = 7.0

DEFAULT_WEIGHTS = {
    "alpha": 0.25,
    "beta": 0.25,
    "gamma": 0.20,
    "delta": 0.30,
}

OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
