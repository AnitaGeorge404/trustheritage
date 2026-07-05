"""SHA-256 hashing and provenance checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

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


def verify_provenance(suspect_path: Path, archived_watermarked_hash: str) -> dict:
    """Compare a suspect file hash against the archived watermarked hash."""
    suspect_hash = sha256_file(suspect_path)
    exact_match = suspect_hash == archived_watermarked_hash
    return {
        "suspect_hash": suspect_hash,
        "exact_match": exact_match,
        "provenance_score": 1.0 if exact_match else 0.25,
    }
