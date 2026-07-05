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
                metric_cols[0].metric("Watermark", f"{scores['watermark_score']:.3f}")
                metric_cols[1].metric("Provenance", f"{scores['provenance_score']:.3f}")
                metric_cols[2].metric("Forensic", f"{scores['forensic_score']:.3f}")
                metric_cols[3].metric("Semantic", f"{scores['semantic_score']:.3f}")
                metric_cols[4].metric("ACS", f"{scores['acs']:.3f}")

                st.markdown(f"### {scores['label']}")
                st.write(scores["explanation"])
                st.markdown("**Detailed evidence**")
                st.json(
                    {
                        "provenance": result["provenance"],
                        "forensic": result["forensic"],
                        "semantic": result["semantic"],
                        "weights": scores["weights"],
                    }
                )
            except Exception as exc:
                LOGGER.exception("Verification failed")
                st.error(f"Verification failed: {exc}")
