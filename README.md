# TrustHeritage Prototype

TrustHeritage is a local proof-of-concept for the research project **"TrustHeritage: A Multi-Layer AI Framework for Authenticity Preservation of Digitized Cultural Heritage."** It demonstrates a complete image-registration and verification pipeline using open-source Python tools only.

This is a research prototype for demos, screenshots, and presentations. It is not a production authenticity, legal evidence, or conservation decision system.

## Features

- Register an original cultural heritage image.
- Embed a small invisible watermark using a discrete wavelet transform (DWT).
- Store SHA-256 hashes for original image bytes, watermarked image bytes, and canonical metadata JSON.
- Verify a suspect image using four evidence layers:
  - DWT watermark recovery.
  - Local provenance hash consistency.
  - Lightweight image forensics with SSIM, absolute difference, histogram consistency, and heatmap output.
  - OpenCLIP image embedding similarity on CPU.
- Fuse evidence into an Authenticity Confidence Score (ACS).
- Run the workflow through a minimal Streamlit interface.
- Generate demo suspect images with crop, blur, and contrast attacks.

## Folder Structure

```text
trustheritage_prototype/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── sample_config.json
├── data/
│   ├── records/
│   ├── archived_images/
│   ├── watermarked_images/
│   ├── suspect_uploads/
│   └── outputs/
├── modules/
│   ├── preprocessing.py
│   ├── watermarking.py
│   ├── hashing.py
│   ├── forensics.py
│   ├── semantics.py
│   ├── scoring.py
│   ├── pipeline.py
│   └── utils.py
├── schemas/
│   └── record_schema.json
├── scripts/
│   └── generate_attacks.py
├── notebooks/
│   └── backend_smoke_test.ipynb
└── assets/
    └── sample_images/
```

## Setup

Create and activate a virtual environment:

```bash
cd trustheritage_prototype
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first semantic verification may download OpenCLIP model weights through `open-clip-torch`. After that, the model is cached locally by the underlying libraries.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Workflow

1. Open the **Register archival image** tab.
2. Upload an image and fill in metadata.
3. Click **Register Image**.
4. Open the **Verify suspect image** tab.
5. Select the archived record.
6. Upload a suspect image.
7. Click **Verify**.

The app displays the archived image, suspect image, forensic heatmap, individual evidence scores, ACS, label, and a short interpretation.

## Attack Generator

After registering an image, you can create demo suspect variants:

```bash
python scripts/generate_attacks.py data/watermarked_images/YOUR_ASSET_watermarked.png --output data/suspect_uploads
```

This creates cropped, blurred, and contrast-modified variants that can be uploaded in the verification tab.

## How The Modules Work

- `preprocessing.py`: loads images, converts them to standard BGR format, and resizes them to a fixed 512x512 shape.
- `watermarking.py`: converts a text payload to deterministic binary bits, embeds those bits into the DWT horizontal-detail band of the luminance channel, and estimates recovery score during extraction.
- `hashing.py`: computes SHA-256 digests for files and metadata and checks whether a suspect image is an exact byte-level match to the archived watermarked image.
- `forensics.py`: compares reference and suspect images with SSIM, absolute difference, histogram correlation, and produces a heatmap for suspicious regions.
- `semantics.py`: uses OpenCLIP image embeddings and cosine similarity to estimate whether the archived and suspect images are semantically similar.
- `scoring.py`: computes `ACS = alpha W + beta P + gamma F + delta S` with default weights 0.25, 0.25, 0.20, and 0.30.
- `pipeline.py`: connects registration and verification into reusable backend functions for the app and notebook.

## ACS Labels

- `Authentic`: ACS >= 0.85
- `Suspicious`: 0.60 <= ACS < 0.85
- `Likely Tampered`: ACS < 0.60

## Limitations

- The watermark is intentionally simple and can be damaged by resizing, compression, cropping, or aggressive edits.
- The forensic layer is a lightweight signal, not a deep forgery detector.
- OpenCLIP semantic similarity says whether two images are visually/semantically close; it does not prove authenticity.
- Hash provenance only confirms exact byte-level consistency with the local archived watermarked file.
- No benchmark claims are made or implied.

## Disclaimer

TrustHeritage Prototype is an educational and research demonstration. It should not be used as a production cultural heritage authenticity system, a legal evidence tool, or a replacement for expert conservation review.
