"""Streamlit interface for the TrustHeritage prototype."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config import (
    ARCHIVED_IMAGES_DIR,
    OUTPUTS_DIR,
    RECORDS_DIR,
    SUSPECT_UPLOADS_DIR,
    WATERMARKED_IMAGES_DIR,
)
from modules.pipeline import register_image, verify_image
from modules.utils import (
    ensure_directories,
    list_records,
    load_json,
    read_image,
    safe_slug,
    setup_logging,
    bgr_to_rgb,
)

setup_logging()
LOGGER = logging.getLogger(__name__)

ensure_directories(
    [
        RECORDS_DIR,
        ARCHIVED_IMAGES_DIR,
        WATERMARKED_IMAGES_DIR,
        SUSPECT_UPLOADS_DIR,
        OUTPUTS_DIR,
    ]
)

st.set_page_config(page_title="TrustHeritage Prototype", page_icon="TH", layout="wide")
st.title("TrustHeritage Prototype")
st.caption("Local research demo for cultural heritage image authenticity verification.")
st.warning(
    "Research prototype: ACS is a heuristic fused score. Results support expert review "
    "and triage, not final certification or legal attribution.",
)


def save_upload(uploaded_file, destination: Path) -> Path:
    """Persist a Streamlit upload to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def show_hashes(hashes: dict[str, str]) -> None:
    """Display hashes in a compact table."""
    st.dataframe(
        [{"name": key, "sha256": value} for key, value in hashes.items()],
        hide_index=True,
        use_container_width=True,
    )


def format_score(value: float | None) -> str:
    """Format optional score values for Streamlit metrics."""
    return "N/A" if value is None else f"{value:.3f}"


def provenance_status(provenance: dict) -> str:
    """Return a clear provenance label for exact hash evidence."""
    return "Exact hash match" if provenance.get("exact_match") else "Hash mismatch"


def show_acs_legend() -> None:
    """Show the decision bands used by the ACS classifier."""
    st.caption(
        "ACS bands: >= 0.85 Authentic (high consistency) | "
        "0.60-0.85 Suspicious (mixed evidence) | "
        "< 0.60 Likely tampered or significantly altered."
    )


def show_semantic_note(semantic: dict, active_weights: dict) -> None:
    """Explain when semantic evidence is unavailable and weights were renormalized."""
    if semantic.get("available", semantic.get("semantic_score") is not None):
        return

    notes = semantic.get("notes") or ["Semantic similarity was not used for this run."]
    st.info(
        "Semantic similarity is unavailable or disabled. ACS was computed using "
        "the remaining active evidence weights shown below."
    )
    for note in notes:
        st.caption(f"- {note}")
    st.caption(f"Active weights: {active_weights}")


def show_interpretation_panel(interpretation: dict) -> None:
    """Render rules-based interpretation results for demo presentation."""
    if not interpretation:
        return

    top_cols = st.columns(3)
    top_cols[0].markdown(f"**Decision**  \n{interpretation.get('decision_label', 'N/A')}")
    top_cols[1].markdown(f"**Review risk**  \n{interpretation.get('risk_label', 'N/A')}")
    top_cols[2].markdown(f"**Evidence confidence**  \n{interpretation.get('confidence_label', 'N/A')}")

    recommendation = interpretation.get("review_recommendation")
    if recommendation:
        st.info(recommendation)

    narrative = interpretation.get("narrative_explanation")
    if narrative:
        st.write(narrative)

    caution_flags = interpretation.get("caution_flags") or []
    if caution_flags:
        st.warning("Caution flags")
        for flag in caution_flags:
            st.markdown(f"- {flag}")


def show_heritage_risk_panel(heritage_risk: dict) -> None:
    """Render semantic-integrity risk and strongest heuristic cues."""
    if not heritage_risk:
        return

    st.markdown("**Heritage-sensitive difference review**")
    risk_cols = st.columns(2)
    risk_cols[0].metric(
        "Semantic Integrity Risk",
        format_score(heritage_risk.get("semantic_integrity_risk")),
    )
    risk_cols[1].metric("Risk label", heritage_risk.get("risk_label", "N/A"))

    strongest_cues = heritage_risk.get("strongest_cues") or []
    if strongest_cues:
        st.caption("Strongest heuristic cues")
        for cue in strongest_cues:
            st.markdown(f"- {cue.get('cue', 'cue')}: {format_score(cue.get('score'))}")

    method_note = heritage_risk.get("method_note")
    if method_note:
        st.caption(method_note)


register_tab, verify_tab = st.tabs(["Register archival image", "Verify suspect image"])

with register_tab:
    st.subheader("Register archival image")
    uploaded = st.file_uploader(
        "Upload original image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="register_upload",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        asset_id = st.text_input("Asset ID", placeholder="e.g. ajanta-panel-001")
        title = st.text_input("Title", placeholder="e.g. Mural fragment")
    with col_b:
        category = st.text_input("Category", placeholder="e.g. Painting, manuscript, sculpture")
        institution = st.text_input("Institution", placeholder="e.g. Museum or archive name")

    if st.button("Register Image", type="primary", disabled=uploaded is None):
        try:
            source_name = safe_slug(asset_id or Path(uploaded.name).stem)
            extension = Path(uploaded.name).suffix.lower() or ".png"
            upload_path = ARCHIVED_IMAGES_DIR / f"{source_name}_upload{extension}"
            save_upload(uploaded, upload_path)
            result = register_image(
                upload_path,
                {
                    "asset_id": asset_id or source_name,
                    "title": title or "Untitled",
                    "category": category or "Unknown",
                    "institution": institution or "Unknown",
                },
            )
            st.success(f"Saved record: {result['record_path'].name}")
            left, right = st.columns(2)
            with left:
                st.image(
                    bgr_to_rgb(read_image(result["archived_path"])),
                    caption="Archived original",
                    use_container_width=True,
                )
            with right:
                st.image(
                    bgr_to_rgb(read_image(result["watermarked_path"])),
                    caption="Watermarked archival image",
                    use_container_width=True,
                )
            quality = result["quality_metrics"]
            st.markdown("**Watermark imperceptibility metrics**")
            quality_cols = st.columns(2)
            quality_cols[0].metric("PSNR original vs watermarked", f"{quality['psnr_original_vs_watermarked']:.2f} dB")
            quality_cols[1].metric("SSIM original vs watermarked", f"{quality['ssim_original_vs_watermarked']:.4f}")
            st.caption(
                "Higher PSNR indicates lower visible distortion. SSIM close to 1.0 means image structure is well preserved."
            )
            st.markdown("**Generated SHA-256 hashes**")
            show_hashes(result["record"]["hashes"])
            st.json(result["record"]["metadata"])
        except Exception as exc:
            LOGGER.exception("Registration failed")
            st.error(f"Registration failed: {exc}")

with verify_tab:
    st.subheader("Verify suspect image")
    records = list_records(RECORDS_DIR)
    if not records:
        st.info("No archived records found. Register an image first.")
    else:
        labels = []
        for record_path in records:
            record = load_json(record_path)
            meta = record.get("metadata", {})
            labels.append(f"{meta.get('asset_id', record_path.stem)} - {meta.get('title', 'Untitled')}")

        selected_label = st.selectbox("Choose archived record", labels)
        selected_record = records[labels.index(selected_label)]
        suspect = st.file_uploader(
            "Upload suspect image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="verify_upload",
        )
        use_semantics = st.checkbox("Run OpenCLIP semantic similarity", value=True)

        if st.button("Verify", type="primary", disabled=suspect is None):
            try:
                suspect_name = f"{safe_slug(Path(suspect.name).stem)}_{selected_record.stem}{Path(suspect.name).suffix.lower() or '.png'}"
                suspect_path = save_upload(suspect, SUSPECT_UPLOADS_DIR / suspect_name)
                with st.spinner("Running verification pipeline..."):
                    result = verify_image(selected_record, suspect_path, use_semantics=use_semantics)

                scores = result["scores"]
                interpretation = result.get("interpretation", {})
                heritage_risk = result.get("heritage_risk", {})
                semantic = result.get("semantic", {})
                provenance = result.get("provenance", {})

                image_cols = st.columns(3)
                with image_cols[0]:
                    st.image(
                        bgr_to_rgb(read_image(result["watermarked_path"])),
                        caption="Archived watermarked image",
                        use_container_width=True,
                    )
                with image_cols[1]:
                    st.image(
                        bgr_to_rgb(read_image(result["suspect_path"])),
                        caption="Suspect image",
                        use_container_width=True,
                    )
                with image_cols[2]:
                    st.image(
                        bgr_to_rgb(read_image(result["heatmap_path"])),
                        caption="Forensic difference heatmap",
                        use_container_width=True,
                    )
                    st.caption(
                        "Heatmap colors mark relative visual-difference concentration only; "
                        "they are not proof of tampering."
                    )

                st.divider()
                st.markdown("### Verification dashboard")
                headline_cols = st.columns(3)
                headline_cols[0].metric("ACS", format_score(scores["acs"]))
                headline_cols[1].metric(
                    "Evidence agreement",
                    format_score(scores.get("evidence_agreement")),
                )
                headline_cols[2].metric(
                    "Semantic Integrity Risk",
                    format_score(heritage_risk.get("semantic_integrity_risk")),
                )
                st.markdown(f"**ACS label:** {scores['label']}")
                show_acs_legend()

                st.markdown("#### Interpretation for review")
                show_interpretation_panel(interpretation)

                st.markdown("#### Evidence summary")
                metric_cols = st.columns(5)
                metric_cols[0].metric(
                    "Payload recovery similarity",
                    format_score(scores["watermark_score"]),
                    help="Fraction of expected watermark payload bits recovered from the suspect image.",
                )
                metric_cols[1].metric(
                    "Exact hash provenance",
                    provenance_status(provenance),
                    help="Strict SHA-256 comparison with the archived watermarked image.",
                )
                metric_cols[2].metric(
                    "Heuristic forensic consistency",
                    format_score(scores["forensic_score"]),
                    help="Lightweight consistency score from SSIM, difference, histogram, and edge cues.",
                )
                metric_cols[3].metric(
                    "Embedding similarity score",
                    format_score(scores["semantic_score"]),
                    help="OpenCLIP image-embedding similarity when semantic analysis is enabled and available.",
                )
                metric_cols[4].metric(
                    "Active evidence count",
                    str(scores.get("active_evidence_count", "N/A")),
                    help="Number of available evidence sources used in ACS fusion.",
                )

                st.caption(
                    "The forensic layer is a lightweight consistency estimate, not proof of tampering. "
                    "ACS is a heuristic fused score for review support."
                )
                show_semantic_note(semantic, scores.get("active_weights", {}))

                st.markdown("#### Heritage-sensitive risk cues")
                show_heritage_risk_panel(heritage_risk)

                with st.expander("Detailed forensic component values", expanded=False):
                    st.json(result["forensic"])
                with st.expander("Provenance and semantic evidence", expanded=False):
                    st.json(
                        {
                            "provenance": provenance,
                            "semantic": semantic,
                        }
                    )
                with st.expander("Active ACS weights", expanded=False):
                    st.json(
                        {
                            "configured_weights": scores["weights"],
                            "active_weights": scores["active_weights"],
                        }
                    )
            except Exception as exc:
                LOGGER.exception("Verification failed")
                st.error(f"Verification failed: {exc}")
