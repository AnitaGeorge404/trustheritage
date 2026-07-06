"""Heritage-sensitive interpretation of visual change.

This module is intentionally heuristic. It does not identify iconography,
inscriptions, faces, or culturally important objects. Instead, it asks whether
technical differences overlap with simple visual cues that often mark salient
regions in digitized heritage images.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _normalize_map(values: np.ndarray) -> np.ndarray:
    """Return a float map in [0, 1], handling flat inputs safely."""
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _centrality_map(shape: tuple[int, int]) -> np.ndarray:
    """Weight image center more strongly than borders."""
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    max_distance = np.sqrt(cx * cx + cy * cy) or 1.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip(1.0 - (distance / max_distance), 0.0, 1.0).astype(np.float32)


def _keypoint_density_map(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Approximate visually distinctive regions with blurred ORB keypoints."""
    orb = cv2.ORB_create(nfeatures=1200)
    keypoints = orb.detect(gray, None)
    density = np.zeros(gray.shape, dtype=np.float32)
    for keypoint in keypoints:
        x, y = keypoint.pt
        ix = min(max(int(round(x)), 0), gray.shape[1] - 1)
        iy = min(max(int(round(y)), 0), gray.shape[0] - 1)
        density[iy, ix] += 1.0
    if keypoints:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=9)
    return _normalize_map(density), len(keypoints)


def _weighted_overlap(diff_map: np.ndarray, cue_map: np.ndarray) -> float:
    """Measure how much suspicious difference energy overlaps a cue map."""
    diff = _normalize_map(diff_map)
    total_diff = float(diff.sum())
    if total_diff <= 1e-6:
        return 0.0
    return float(np.clip((diff * cue_map).sum() / total_diff, 0.0, 1.0))


def compute_semantic_integrity_risk(
    reference: np.ndarray,
    difference_map: np.ndarray,
    suspicious_mask: np.ndarray,
    forensic: dict[str, Any],
    semantic_score: float | None = None,
) -> dict[str, Any]:
    """Compute a transparent Semantic Integrity Risk score in [0, 1].

    Higher values mean detected changes are more concentrated in visually
    salient regions. This is a preservation-aware review cue, not a detector of
    cultural meaning.
    """
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    diff = cv2.resize(difference_map, (gray.shape[1], gray.shape[0]))
    mask = cv2.resize(suspicious_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    centrality = _centrality_map(gray.shape)
    edges = _normalize_map(cv2.Canny(gray, 80, 160))
    keypoint_density, keypoint_count = _keypoint_density_map(gray)

    suspicious_region_ratio = float(forensic.get("suspicious_region_ratio", float(mask.mean())))
    difference_intensity = float(np.clip(_normalize_map(diff).mean() * 2.5, 0.0, 1.0))
    central_overlap = _weighted_overlap(diff, centrality)
    edge_overlap = _weighted_overlap(diff, edges)
    keypoint_overlap = _weighted_overlap(diff, keypoint_density)

    semantic_drop = 0.0
    if semantic_score is not None:
        semantic_drop = float(np.clip((0.90 - semantic_score) / 0.40, 0.0, 1.0))

    risk = (
        0.22 * np.clip(suspicious_region_ratio * 8.0, 0.0, 1.0)
        + 0.18 * difference_intensity
        + 0.22 * central_overlap
        + 0.16 * edge_overlap
        + 0.14 * keypoint_overlap
        + 0.08 * semantic_drop
    )
    risk = float(np.clip(risk, 0.0, 1.0))

    if risk >= 0.70:
        label = "High semantic integrity risk"
    elif risk >= 0.40:
        label = "Elevated semantic integrity risk"
    elif risk >= 0.20:
        label = "Moderate semantic integrity risk"
    else:
        label = "Low semantic integrity risk"

    strongest_cues = sorted(
        [
            ("central-region overlap", central_overlap),
            ("edge/contour overlap", edge_overlap),
            ("keypoint-density overlap", keypoint_overlap),
            ("semantic similarity drop", semantic_drop),
            ("suspicious-region area", np.clip(suspicious_region_ratio * 8.0, 0.0, 1.0)),
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    return {
        "semantic_integrity_risk": risk,
        "risk_label": label,
        "suspicious_region_ratio": suspicious_region_ratio,
        "difference_intensity": difference_intensity,
        "central_overlap": central_overlap,
        "edge_overlap": edge_overlap,
        "keypoint_overlap": keypoint_overlap,
        "semantic_drop": semantic_drop,
        "reference_keypoints": keypoint_count,
        "strongest_cues": [
            {"cue": name, "score": float(score)}
            for name, score in strongest_cues
        ],
        "method_note": (
            "Heuristic heritage-sensitive indicator based on difference location, "
            "centrality, contour density, ORB keypoint density, and optional CLIP "
            "similarity drop. It does not perform iconographic understanding."
        ),
    }
