"""Lightweight image-difference forensic indicators."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _align_to_reference(reference_gray: np.ndarray, suspect_gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """Align suspect to reference with ORB features when enough matches exist."""
    orb = cv2.ORB_create(nfeatures=1500)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(reference_gray, None)
    sus_keypoints, sus_descriptors = orb.detectAndCompute(suspect_gray, None)
    diagnostics = {
        "method": "none",
        "reference_keypoints": len(ref_keypoints or []),
        "suspect_keypoints": len(sus_keypoints or []),
        "matches": 0,
        "inliers": 0,
    }

    if ref_descriptors is None or sus_descriptors is None:
        return suspect_gray, diagnostics

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(sus_descriptors, ref_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good_matches.append(pair[0])
    diagnostics["matches"] = len(good_matches)

    if len(good_matches) < 12:
        return suspect_gray, diagnostics

    src = np.float32([sus_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([ref_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return suspect_gray, diagnostics

    diagnostics["inliers"] = int(mask.sum())
    inlier_ratio = diagnostics["inliers"] / max(len(good_matches), 1)
    if diagnostics["inliers"] < 10 or inlier_ratio < 0.35:
        return suspect_gray, diagnostics

    aligned = cv2.warpPerspective(
        suspect_gray,
        homography,
        (reference_gray.shape[1], reference_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    diagnostics["method"] = "orb_homography"
    diagnostics["inlier_ratio"] = float(inlier_ratio)
    return aligned, diagnostics


def analyze_forensics(reference: np.ndarray, suspect: np.ndarray) -> dict:
    """Return a forensic consistency score and suspicious-region heatmap.

    The score combines SSIM, normalized absolute difference, and histogram
    similarity. It should be read as a practical warning signal, not as proof.
    """
    suspect_resized = cv2.resize(suspect, (reference.shape[1], reference.shape[0]))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    sus_gray = cv2.cvtColor(suspect_resized, cv2.COLOR_BGR2GRAY)
    aligned_gray, alignment = _align_to_reference(ref_gray, sus_gray)

    ssim_score = float(ssim(ref_gray, aligned_gray, data_range=255))
    abs_diff = cv2.absdiff(ref_gray, aligned_gray)
    diff_score = 1.0 - min(float(abs_diff.mean()) / 80.0, 1.0)

    ref_hist = cv2.calcHist([ref_gray], [0], None, [64], [0, 256])
    sus_hist = cv2.calcHist([aligned_gray], [0], None, [64], [0, 256])
    cv2.normalize(ref_hist, ref_hist)
    cv2.normalize(sus_hist, sus_hist)
    hist_score = float((cv2.compareHist(ref_hist, sus_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
    hist_score = max(0.0, min(hist_score, 1.0))

    ref_edges = cv2.Canny(ref_gray, 80, 160)
    sus_edges = cv2.Canny(aligned_gray, 80, 160)
    edge_diff = cv2.absdiff(ref_edges, sus_edges)
    edge_score = 1.0 - min(float(edge_diff.mean()) / 80.0, 1.0)

    suspicious_mask = abs_diff > max(20, int(abs_diff.mean() + 2.0 * abs_diff.std()))
    suspicious_region_ratio = float(suspicious_mask.mean())

    forensic_score = (
        0.45 * max(0.0, ssim_score)
        + 0.25 * diff_score
        + 0.15 * hist_score
        + 0.15 * edge_score
    )
    forensic_score = float(max(0.0, min(forensic_score, 1.0)))

    heat = cv2.GaussianBlur(abs_diff, (0, 0), sigmaX=3)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    return {
        "forensic_score": forensic_score,
        "ssim": ssim_score,
        "difference_score": diff_score,
        "histogram_score": hist_score,
        "edge_score": float(max(0.0, min(edge_score, 1.0))),
        "suspicious_region_ratio": suspicious_region_ratio,
        "alignment": alignment,
        "difference_map": abs_diff,
        "suspicious_mask": suspicious_mask.astype(np.uint8) * 255,
        "heatmap": heatmap,
    }
