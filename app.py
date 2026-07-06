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

                scores = result["scores"]
                metric_cols = st.columns(5)
                metric_cols[0].metric("Payload recovery similarity", format_score(scores["watermark_score"]))
                metric_cols[1].metric("Exact hash provenance", provenance_status(result["provenance"]))
                metric_cols[2].metric("Heuristic forensic consistency", format_score(scores["forensic_score"]))
                metric_cols[3].metric("Embedding similarity score", format_score(scores["semantic_score"]))
                metric_cols[4].metric("ACS", format_score(scores["acs"]))

                st.markdown(f"### {scores['label']}")
                st.write(scores["explanation"])
                st.caption(
                    "The forensic layer is a lightweight consistency estimate, not proof of tampering."
                )

                with st.expander("Detailed forensic component values", expanded=False):
                    st.json(result["forensic"])
                with st.expander("Provenance and semantic evidence", expanded=False):
                    st.json(
                        {
                            "provenance": result["provenance"],
                            "semantic": result["semantic"],
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
