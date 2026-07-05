"""Lightweight image-difference forensic indicators."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def analyze_forensics(reference: np.ndarray, suspect: np.ndarray) -> dict:
    """Return a forensic consistency score and suspicious-region heatmap.

    The score combines SSIM, normalized absolute difference, and histogram
    similarity. It should be read as a practical warning signal, not as proof.
    """
    suspect_resized = cv2.resize(suspect, (reference.shape[1], reference.shape[0]))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    sus_gray = cv2.cvtColor(suspect_resized, cv2.COLOR_BGR2GRAY)

    ssim_score = float(ssim(ref_gray, sus_gray, data_range=255))
    abs_diff = cv2.absdiff(ref_gray, sus_gray)
    diff_score = 1.0 - min(float(abs_diff.mean()) / 80.0, 1.0)

    ref_hist = cv2.calcHist([ref_gray], [0], None, [64], [0, 256])
    sus_hist = cv2.calcHist([sus_gray], [0], None, [64], [0, 256])
    cv2.normalize(ref_hist, ref_hist)
    cv2.normalize(sus_hist, sus_hist)
    hist_score = float((cv2.compareHist(ref_hist, sus_hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0)
    hist_score = max(0.0, min(hist_score, 1.0))

    forensic_score = 0.55 * max(0.0, ssim_score) + 0.30 * diff_score + 0.15 * hist_score
    forensic_score = float(max(0.0, min(forensic_score, 1.0)))

    heat = cv2.GaussianBlur(abs_diff, (0, 0), sigmaX=3)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    return {
        "forensic_score": forensic_score,
        "ssim": ssim_score,
        "difference_score": diff_score,
        "histogram_score": hist_score,
        "heatmap": heatmap,
    }
