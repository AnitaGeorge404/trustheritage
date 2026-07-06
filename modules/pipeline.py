"""Registration and verification orchestration for TrustHeritage."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
from skimage.metrics import structural_similarity as ssim

from config import (
    ARCHIVED_IMAGES_DIR,
    OUTPUTS_DIR,
    RECORDS_DIR,
    SUSPECT_UPLOADS_DIR,
    WATERMARKED_IMAGES_DIR,
)
from modules.forensics import analyze_forensics
from modules.hashing import sha256_file, sha256_metadata, verify_provenance
from modules.heritage_risk import compute_semantic_integrity_risk
from modules.interpreter import interpret_result
from modules.preprocessing import load_and_preprocess
from modules.scoring import fuse_scores
from modules.semantics import semantic_similarity
from modules.utils import load_json, safe_slug, save_image, save_json, utc_now_iso
from modules.watermarking import embed_watermark, extract_watermark

LOGGER = logging.getLogger(__name__)


def image_quality_metrics(original: Any, watermarked: Any) -> dict[str, float]:
    """Compute imperceptibility metrics for watermark embedding."""
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    return {
        "psnr_original_vs_watermarked": float(cv2.PSNR(original, watermarked)),
        "ssim_original_vs_watermarked": float(
            ssim(original_gray, watermarked_gray, data_range=255)
        ),
    }


def register_image(
    image_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Register an archival image, watermark it, hash it, and save a record."""
    image = load_and_preprocess(image_path)
    asset_id = safe_slug(metadata.get("asset_id") or str(uuid.uuid4())[:8])
    metadata = {
        "asset_id": asset_id,
        "title": metadata.get("title", "Untitled"),
        "category": metadata.get("category", "Unknown"),
        "institution": metadata.get("institution", "Unknown"),
        "date_registered": metadata.get("date_registered") or utc_now_iso(),
    }

    archived_path = ARCHIVED_IMAGES_DIR / f"{asset_id}.png"
    watermarked_path = WATERMARKED_IMAGES_DIR / f"{asset_id}_watermarked.png"
    record_path = RECORDS_DIR / f"{asset_id}.json"

    save_image(archived_path, image)
    payload = f"{asset_id}|{metadata['title']}|{metadata['date_registered']}"
    watermarked = embed_watermark(image, payload=payload)
    save_image(watermarked_path, watermarked)
    quality_metrics = image_quality_metrics(image, watermarked)

    hashes = {
        "original_image_sha256": sha256_file(archived_path),
        "watermarked_image_sha256": sha256_file(watermarked_path),
        "metadata_sha256": sha256_metadata(metadata),
    }
    record = {
        "metadata": metadata,
        "payload": payload,
        "paths": {
            "archived_image": str(archived_path.relative_to(record_path.parent.parent.parent)),
            "watermarked_image": str(watermarked_path.relative_to(record_path.parent.parent.parent)),
        },
        "hashes": hashes,
        "quality_metrics": quality_metrics,
    }
    save_json(record_path, record)
    LOGGER.info("Registered asset %s", asset_id)
    return {
        "record_path": record_path,
        "archived_path": archived_path,
        "watermarked_path": watermarked_path,
        "record": record,
        "quality_metrics": quality_metrics,
    }


def verify_image(record_path: Path, suspect_path: Path, use_semantics: bool = True) -> dict[str, Any]:
    """Verify a suspect image against an archived TrustHeritage record."""
    record = load_json(record_path)
    base_dir = record_path.parent.parent.parent
    archived_path = base_dir / record["paths"]["archived_image"]
    watermarked_path = base_dir / record["paths"]["watermarked_image"]

    archived_image = load_and_preprocess(archived_path)
    watermarked_image = load_and_preprocess(watermarked_path)
    suspect_image = load_and_preprocess(suspect_path)

    _, watermark_score = extract_watermark(suspect_image, payload=record["payload"])
    provenance = verify_provenance(
        suspect_path,
        archived_watermarked_hash=record["hashes"]["watermarked_image_sha256"],
        archived_watermarked_path=watermarked_path,
    )
    forensic = analyze_forensics(watermarked_image, suspect_image)

    if use_semantics:
        try:
            semantic = semantic_similarity(archived_image, suspect_image, device="cpu")
            semantic["available"] = True
            semantic["notes"] = []
        except Exception as exc:
            LOGGER.warning("Semantic similarity unavailable: %s", exc)
            semantic = {
                "cosine_similarity": None,
                "semantic_score": None,
                "available": False,
                "notes": [f"Semantic model unavailable: {exc}"],
            }
    else:
        semantic = {
            "cosine_similarity": None,
            "semantic_score": None,
            "available": False,
            "notes": ["Semantic similarity disabled for this run."],
        }

    heatmap_path = OUTPUTS_DIR / f"{record['metadata']['asset_id']}_heatmap.png"
    save_image(heatmap_path, forensic["heatmap"])

    scores = fuse_scores(
        watermark_score=watermark_score,
        provenance_score=provenance["provenance_score"],
        forensic_score=forensic["forensic_score"],
        semantic_score=semantic["semantic_score"],
    )
    heritage_risk = compute_semantic_integrity_risk(
        reference=watermarked_image,
        difference_map=forensic["difference_map"],
        suspicious_mask=forensic["suspicious_mask"],
        forensic=forensic,
        semantic_score=semantic["semantic_score"],
    )
    interpretation = interpret_result(
        acs=scores["acs"],
        semantic_integrity_risk=heritage_risk["semantic_integrity_risk"],
        evidence_agreement=scores["evidence_agreement"],
        provenance_exact_match=provenance["exact_match"],
        watermark_score=watermark_score,
        semantic_score=semantic["semantic_score"],
    )

    return {
        "record": record,
        "archived_path": archived_path,
        "watermarked_path": watermarked_path,
        "suspect_path": suspect_path,
        "heatmap_path": heatmap_path,
        "watermark_score": watermark_score,
        "provenance": provenance,
        "forensic": {
            k: v
            for k, v in forensic.items()
            if k not in {"heatmap", "difference_map", "suspicious_mask"}
        },
        "semantic": semantic,
        "scores": scores,
        "heritage_risk": heritage_risk,
        "interpretation": interpretation,
        "technical_evidence": {
            "watermark_score": watermark_score,
            "provenance_score": provenance["provenance_score"],
            "forensic_score": forensic["forensic_score"],
            "semantic_score": semantic["semantic_score"],
            "acs": scores["acs"],
            "evidence_agreement": scores["evidence_agreement"],
        },
    }


def persist_uploaded_file(source_path: Path, destination_name: str) -> Path:
    """Copy a local upload temp file into suspect uploads."""
    destination = SUSPECT_UPLOADS_DIR / destination_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return destination
