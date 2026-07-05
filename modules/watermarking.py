"""Simple DWT watermarking for demonstration.

Assumptions:
- Images are preprocessed to a fixed size before embedding and extraction.
- The payload is converted to a deterministic 32x32 binary pattern.
- Bits are embedded into the horizontal-detail DWT band of the luminance channel.
- This is a research demo watermark, not a robust production watermark.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pywt

from config import WATERMARK_SIZE, WATERMARK_STRENGTH


def payload_to_bits(payload: str, size: int = WATERMARK_SIZE) -> np.ndarray:
    """Create a deterministic binary payload pattern from text."""
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    needed = size * size
    stream = bytearray()
    counter = 0
    while len(stream) * 8 < needed:
        stream.extend(hashlib.sha256(digest + counter.to_bytes(4, "big")).digest())
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(stream), dtype=np.uint8))[:needed]
    return bits.reshape(size, size).astype(np.uint8)


def _embed_bits_in_band(band: np.ndarray, bits: np.ndarray, strength: float) -> np.ndarray:
    """Embed one payload bit in each block by nudging the block mean parity."""
    result = band.copy()
    block_h = band.shape[0] // bits.shape[0]
    block_w = band.shape[1] // bits.shape[1]
    if block_h < 1 or block_w < 1:
        raise ValueError("Image is too small for the configured watermark size.")

    for row in range(bits.shape[0]):
        for col in range(bits.shape[1]):
            y0, y1 = row * block_h, (row + 1) * block_h
            x0, x1 = col * block_w, (col + 1) * block_w
            block = result[y0:y1, x0:x1]
            mean_value = float(block.mean())
            quantized = np.round(mean_value / strength)
            target_parity = int(bits[row, col])
            if int(quantized) % 2 != target_parity:
                adjustment = strength if mean_value >= 0 else -strength
                block += adjustment
    return result


def _extract_bits_from_band(band: np.ndarray, size: int, strength: float) -> np.ndarray:
    """Recover bits from DWT band block mean parity."""
    bits = np.zeros((size, size), dtype=np.uint8)
    block_h = band.shape[0] // size
    block_w = band.shape[1] // size
    if block_h < 1 or block_w < 1:
        raise ValueError("Image is too small for the configured watermark size.")

    for row in range(size):
        for col in range(size):
            y0, y1 = row * block_h, (row + 1) * block_h
            x0, x1 = col * block_w, (col + 1) * block_w
            mean_value = float(band[y0:y1, x0:x1].mean())
            bits[row, col] = int(np.round(mean_value / strength)) % 2
    return bits


def embed_watermark(
    image: np.ndarray,
    payload: str,
    strength: float = WATERMARK_STRENGTH,
) -> np.ndarray:
    """Embed a deterministic binary watermark payload into a BGR image."""
    bits = payload_to_bits(payload)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel = ycrcb[:, :, 0]
    coeffs = pywt.dwt2(y_channel, "haar")
    ll, (lh, hl, hh) = coeffs
    hl_marked = _embed_bits_in_band(hl, bits, strength)
    y_marked = pywt.idwt2((ll, (lh, hl_marked, hh)), "haar")
    ycrcb[:, :, 0] = np.clip(y_marked[: image.shape[0], : image.shape[1]], 0, 255)
    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def extract_watermark(
    image: np.ndarray,
    payload: str,
    strength: float = WATERMARK_STRENGTH,
) -> tuple[np.ndarray, float]:
    """Extract watermark bits and return a recovery score in [0, 1]."""
    expected = payload_to_bits(payload)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    _, (_, hl, _) = pywt.dwt2(ycrcb[:, :, 0], "haar")
    recovered = _extract_bits_from_band(hl, expected.shape[0], strength)
    score = float((recovered == expected).mean())
    return recovered, score
