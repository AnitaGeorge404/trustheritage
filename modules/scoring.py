"""Evidence fusion for the Authenticity Confidence Score."""

from __future__ import annotations

from config import DEFAULT_WEIGHTS


def classify_acs(acs: float) -> str:
    """Map ACS to a human-readable decision label."""
    if acs >= 0.85:
        return "Authentic"
    if acs >= 0.60:
        return "Suspicious"
    return "Likely Tampered"


def fuse_scores(
    watermark_score: float,
    provenance_score: float,
    forensic_score: float,
    semantic_score: float,
    weights: dict[str, float] | None = None,
) -> dict:
    """Compute ACS = alpha W + beta P + gamma F + delta S."""
    weights = weights or DEFAULT_WEIGHTS
    acs = (
        weights["alpha"] * watermark_score
        + weights["beta"] * provenance_score
        + weights["gamma"] * forensic_score
        + weights["delta"] * semantic_score
    )
    acs = float(max(0.0, min(acs, 1.0)))
    label = classify_acs(acs)
    explanation = (
        f"{label}: ACS={acs:.3f}. Watermark={watermark_score:.3f}, "
        f"provenance={provenance_score:.3f}, forensic={forensic_score:.3f}, "
        f"semantic={semantic_score:.3f}."
    )
    return {
        "watermark_score": watermark_score,
        "provenance_score": provenance_score,
        "forensic_score": forensic_score,
        "semantic_score": semantic_score,
        "acs": acs,
        "label": label,
        "explanation": explanation,
        "weights": weights,
    }
