"""Evidence fusion for the Authenticity Consistency Score."""

from __future__ import annotations

import statistics

from config import DEFAULT_WEIGHTS


def classify_acs(acs: float) -> str:
    """Map ACS to a human-readable decision label."""
    if acs >= 0.85:
        return "Authentic (high consistency)"
    if acs >= 0.60:
        return "Suspicious (mixed evidence)"
    return "Likely tampered or significantly altered"


def evidence_agreement(evidence_scores: dict[str, float | None]) -> dict:
    """Estimate whether active evidence sources tell a consistent story."""
    active = {
        key: float(value)
        for key, value in evidence_scores.items()
        if value is not None
    }
    if len(active) <= 1:
        return {
            "evidence_agreement": 1.0,
            "confidence_label": "High agreement",
            "score_dispersion": 0.0,
            "active_evidence_count": len(active),
            "active_evidence_scores": active,
        }

    values = list(active.values())
    dispersion = float(statistics.pstdev(values))
    agreement = 1.0 - min(dispersion / 0.50, 1.0)

    high = sum(value >= 0.75 for value in values)
    low = sum(value < 0.50 for value in values)
    if high > 0 and low > 0:
        agreement *= 0.75

    agreement = float(max(0.0, min(agreement, 1.0)))
    if agreement >= 0.75:
        confidence_label = "High agreement"
    elif agreement >= 0.50:
        confidence_label = "Moderate agreement"
    else:
        confidence_label = "Conflicting evidence"

    return {
        "evidence_agreement": agreement,
        "confidence_label": confidence_label,
        "score_dispersion": dispersion,
        "active_evidence_count": len(active),
        "active_evidence_scores": active,
    }


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
    agreement = evidence_agreement(evidence)
    semantic_text = "not run" if semantic_score is None else f"{semantic_score:.3f}"
    explanation = (
        f"{label}: ACS={acs:.3f}. Payload recovery similarity={watermark_score:.3f}, "
        f"exact-hash provenance={provenance_score:.3f}, "
        f"heuristic forensic consistency={forensic_score:.3f}, "
        f"embedding similarity={semantic_text}. Evidence agreement="
        f"{agreement['evidence_agreement']:.3f} ({agreement['confidence_label']}). "
        "ACS is a heuristic technical consistency score."
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
        **agreement,
    }
