"""Rules-based preservation-aware interpretation for TrustHeritage."""

from __future__ import annotations

from typing import Any


def _acs_band(acs: float) -> str:
    if acs >= 0.85:
        return "high"
    if acs >= 0.60:
        return "medium"
    return "low"


def _sir_band(sir: float) -> str:
    if sir >= 0.70:
        return "high"
    if sir >= 0.40:
        return "medium"
    return "low"


def interpret_result(
    acs: float,
    semantic_integrity_risk: float,
    evidence_agreement: float,
    provenance_exact_match: bool,
    watermark_score: float,
    semantic_score: float | None,
) -> dict[str, Any]:
    """Return a human-readable interpretation bundle from transparent rules."""
    acs_band = _acs_band(acs)
    sir_band = _sir_band(semantic_integrity_risk)

    if evidence_agreement >= 0.75:
        confidence_label = "High agreement"
    elif evidence_agreement >= 0.50:
        confidence_label = "Moderate agreement"
    else:
        confidence_label = "Conflicting evidence"

    if acs_band == "high" and sir_band == "low":
        decision_label = "Archive-consistent with low semantic risk"
        risk_label = "Low review priority"
        review_recommendation = "No immediate concern; archive-consistent."
        narrative = (
            "The image shows strong archival consistency, and detected differences "
            "do not appear concentrated in regions likely to affect meaningful "
            "visual content."
        )
    elif acs_band == "high" and sir_band != "low":
        decision_label = "Technically consistent but preservation-sensitive"
        risk_label = "Meaningful-change caution"
        review_recommendation = "Curatorial spot-check recommended due to localized salient change."
        narrative = (
            "The technical evidence is largely consistent with the archive, but "
            "the difference pattern falls in visually salient regions. This may "
            "reflect benign conservation, digitization, or compression effects, "
            "but it deserves preservation-sensitive review."
        )
    elif acs_band == "medium" and sir_band == "high":
        decision_label = "Mixed archival consistency with elevated semantic risk"
        risk_label = "Possible meaningful alteration"
        review_recommendation = "Curatorial review strongly recommended due to likely meaningful alteration."
        narrative = (
            "The image shows mixed archival consistency with elevated semantic "
            "integrity risk. Differences appear concentrated in regions likely "
            "to affect meaningful visual content."
        )
    elif acs_band == "medium":
        decision_label = "Mixed archival consistency"
        risk_label = "Review recommended"
        review_recommendation = "Review recommended due to mixed signals."
        narrative = (
            "The evidence is not fully aligned. The result suggests an archivist "
            "or curator should compare the image against registration context "
            "before drawing a conclusion."
        )
    elif sir_band == "high":
        decision_label = "Low technical consistency with high semantic risk"
        risk_label = "Likely significant content alteration"
        review_recommendation = "Curatorial review strongly recommended due to likely meaningful alteration."
        narrative = (
            "The technical evidence is weak and the detected differences overlap "
            "strongly with visually salient regions. This is consistent with a "
            "potentially meaningful alteration, although expert review is still "
            "required."
        )
    else:
        decision_label = "Low technical consistency with limited semantic concentration"
        risk_label = "Transformed or degraded image"
        review_recommendation = "Review recommended; check for benign degradation, resizing, or workflow changes."
        narrative = (
            "The image has low archival consistency, but the difference pattern "
            "is not strongly concentrated in visually salient regions. This may "
            "indicate degradation, format transformation, or acquisition changes "
            "rather than clear semantic alteration."
        )

    caution_flags: list[str] = []
    if not provenance_exact_match:
        caution_flags.append("No exact SHA-256 match to the archived watermarked file.")
    if watermark_score < 0.75:
        caution_flags.append("Watermark recovery is weak or partially disrupted.")
    if semantic_score is not None and semantic_score < 0.75:
        caution_flags.append("OpenCLIP similarity is below the expected close-match range.")
    if confidence_label == "Conflicting evidence":
        caution_flags.append("Evidence sources disagree, so the fused score should be read cautiously.")

    return {
        "decision_label": decision_label,
        "risk_label": risk_label,
        "confidence_label": confidence_label,
        "narrative_explanation": narrative,
        "review_recommendation": review_recommendation,
        "interpretation_matrix_cell": f"{acs_band} ACS + {sir_band} SIR",
        "caution_flags": caution_flags,
    }
