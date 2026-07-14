"""SHA-256 hashing and provenance checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from modules.utils import json_canonical_bytes


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_metadata(metadata: dict) -> str:
    """Hash canonical JSON metadata."""
    return sha256_bytes(json_canonical_bytes(metadata))


def perceptual_hash(path: Path, hash_size: int = 16) -> str:
    """Return a compact DCT perceptual hash for visually stable provenance hints."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image for perceptual hash: {path}")
    resized = cv2.resize(image, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:hash_size, :hash_size]
    low_freq[0, 0] = 0.0
    median = float(np.median(low_freq))
    bits = (low_freq > median).astype(np.uint8).flatten()
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def hash_similarity(hash_a: str, hash_b: str) -> float:
    """Return normalized similarity between two same-length hexadecimal bit hashes."""
    if not hash_a or not hash_b or len(hash_a) != len(hash_b):
        return 0.0
    bytes_a = bytes.fromhex(hash_a)
    bytes_b = bytes.fromhex(hash_b)
    distance = sum((a ^ b).bit_count() for a, b in zip(bytes_a, bytes_b))
    total_bits = len(bytes_a) * 8
    return 1.0 - (distance / total_bits)


def verify_provenance(
    suspect_path: Path,
    archived_watermarked_hash: str,
    archived_watermarked_path: Path | None = None,
) -> dict:
    """Compare exact file provenance and report any near-match hint separately."""
    suspect_hash = sha256_file(suspect_path)
    exact_match = suspect_hash == archived_watermarked_hash
    archived_phash = None
    suspect_phash = None
    visual_hash_similarity = None

    if archived_watermarked_path is not None:
        archived_phash = perceptual_hash(archived_watermarked_path)
        suspect_phash = perceptual_hash(suspect_path)
        visual_hash_similarity = hash_similarity(archived_phash, suspect_phash)

    notes = []
    if exact_match:
        notes.append("Exact SHA-256 match to archived watermarked image.")
    else:
        notes.append("SHA-256 mismatch; provenance contributes no positive fused evidence.")
        if visual_hash_similarity is not None:
            notes.append("Perceptual hash similarity is reported only as a non-fused near-match hint.")

    return {
        "suspect_hash": suspect_hash,
        "archived_watermarked_hash": archived_watermarked_hash,
        "exact_match": exact_match,
        "archived_perceptual_hash": archived_phash,
        "suspect_perceptual_hash": suspect_phash,
        "visual_hash_similarity": visual_hash_similarity,
        "provenance_score": 1.0 if exact_match else 0.0,
        "notes": notes,
    }
