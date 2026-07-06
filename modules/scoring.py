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
    semantic_score: float | None,
    weights: dict[str, float] | None = None,
) -> dict:
    """Compute weighted ACS, renormalizing if an evidence layer is unavailable."""
    weights = weights or DEFAULT_WEIGHTS
    evidence = {
        "alpha": watermark_score,
        "beta": provenance_score,
        "gamma": forensic_score,
        "delta": semantic_score,
    }
    available = {key: value for key, value in evidence.items() if value is not None}
    active_weight_total = sum(weights[key] for key in available)
    if active_weight_total <= 0:
        raise ValueError("At least one evidence score must be available.")

    normalized_weights = {
        key: weights[key] / active_weight_total
        for key in available
    }
    acs = sum(normalized_weights[key] * float(value) for key, value in available.items())
    acs = float(max(0.0, min(acs, 1.0)))
    label = classify_acs(acs)
    semantic_text = "not run" if semantic_score is None else f"{semantic_score:.3f}"
    explanation = (
        f"{label}: ACS={acs:.3f}. Watermark={watermark_score:.3f}, "
        f"provenance={provenance_score:.3f}, forensic={forensic_score:.3f}, "
        f"semantic={semantic_text}."
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
        "active_weights": normalized_weights,
    }
