"""Semantic image similarity using open-clip-torch."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from PIL import Image

from config import OPENCLIP_MODEL, OPENCLIP_PRETRAINED

LOGGER = logging.getLogger(__name__)
_MODEL_CACHE: dict[str, object] = {}


def _load_model(device: str = "cpu") -> tuple[torch.nn.Module, object]:
    """Lazy-load OpenCLIP so the Streamlit app can start quickly."""
    cache_key = f"{OPENCLIP_MODEL}:{OPENCLIP_PRETRAINED}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]  # type: ignore[return-value]

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open-clip-torch is not installed. Install requirements.txt first."
        ) from exc

    LOGGER.info("Loading OpenCLIP model %s (%s)", OPENCLIP_MODEL, OPENCLIP_PRETRAINED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL,
        pretrained=OPENCLIP_PRETRAINED,
        device=device,
    )
    model.eval()
    _MODEL_CACHE[cache_key] = (model, preprocess)
    return model, preprocess


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to PIL RGB."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_embedding(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Generate a normalized OpenCLIP image embedding."""
    model, preprocess = _load_model(device=device)
    tensor = preprocess(_to_pil_rgb(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()


def semantic_similarity(reference: np.ndarray, suspect: np.ndarray, device: str = "cpu") -> dict:
    """Compute cosine similarity and normalize it to a [0, 1] semantic score."""
    ref_embedding = image_embedding(reference, device=device)
    sus_embedding = image_embedding(suspect, device=device)
    cosine = float((ref_embedding @ sus_embedding.T).item())
    semantic_score = max(0.0, min((cosine + 1.0) / 2.0, 1.0))
    return {
        "cosine_similarity": cosine,
        "semantic_score": semantic_score,
    }
