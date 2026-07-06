"""Generate and evaluate controlled TrustHeritage research experiments.

This script intentionally preserves the existing TrustHeritage verification
methodology. It generates deterministic suspect images from already registered
watermarked archival assets, then calls modules.pipeline.verify_image for all
verification and score computation.
"""

from __future__ import annotations

import csv
import importlib.metadata
import platform
import random
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None  # type: ignore[assignment]

from config import (
    ARCHIVED_IMAGES_DIR,
    DEFAULT_WEIGHTS,
    IMAGE_SIZE,
    OPENCLIP_MODEL,
    OPENCLIP_PRETRAINED,
    OUTPUTS_DIR,
    RECORDS_DIR,
    SUSPECT_UPLOADS_DIR,
    WATERMARK_SIZE,
    WATERMARK_STRENGTH,
    WATERMARKED_IMAGES_DIR,
)
from modules.pipeline import verify_image
from modules.utils import ensure_directories, load_json, save_image


SEED = 42
DATASET_LABELS = ["H1", "H2", "H3", "H4", "H5"]
EXPERIMENTS_DIR = Path("research_experiments")
RESULTS_DIR = Path("research_results")
RESULTS_CSV = RESULTS_DIR / "trustheritage_experiment_results.csv"
ATTACK_SUMMARY_CSV = RESULTS_DIR / "trustheritage_attack_summary.csv"
REGISTRATION_SUMMARY_CSV = RESULTS_DIR / "trustheritage_registration_summary.csv"
MANIFEST_CSV = RESULTS_DIR / "experiment_manifest.csv"
FAILURES_CSV = RESULTS_DIR / "experiment_failures.csv"

RESULT_FIELDS = [
    "image_id",
    "category",
    "attack_family",
    "attack_name",
    "attack_parameter",
    "suspect_path",
    "watermark_score",
    "provenance_score",
    "exact_hash_match",
    "forensic_score",
    "semantic_score",
    "cosine_similarity",
    "acs",
    "decision",
    "ssim",
    "difference_score",
    "histogram_score",
    "edge_score",
    "suspicious_region_ratio",
    "evidence_agreement",
    "semantic_integrity_risk",
    "alignment_method",
]

NUMERIC_METRICS = [
    "watermark_score",
    "provenance_score",
    "forensic_score",
    "semantic_score",
    "cosine_similarity",
    "acs",
    "ssim",
    "difference_score",
    "histogram_score",
    "edge_score",
    "suspicious_region_ratio",
    "evidence_agreement",
    "semantic_integrity_risk",
]

FAILURE_FIELDS = ["image_id", "attack_name", "error_type", "error_message"]


def set_deterministic_seeds() -> None:
    """Set deterministic seeds for reproducible experiment generation."""
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: Any) -> Any:
    """Use NA for missing values without inventing unavailable metrics."""
    if value is None:
        return "NA"
    if isinstance(value, Path):
        return str(value)
    return value


def read_image(path: Path) -> np.ndarray:
    """Read a BGR image, raising a clear error if OpenCV cannot load it."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_jpeg_roundtrip(path: Path, image: np.ndarray, quality: int) -> None:
    """Encode directly as JPEG at a requested quality."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError(f"Could not encode JPEG quality {quality} for {path}")
    path.write_bytes(encoded.tobytes())


def centered_crop_remove_area(image: np.ndarray, removed_area_fraction: float) -> np.ndarray:
    """Crop centrally, removing approximately the requested total image area.

    If fraction r of total area should be removed, the retained crop area is
    (1-r). For a centered crop preserving aspect ratio, each linear dimension is
    multiplied by sqrt(1-r), so retained_width * retained_height approximates
    (1-r) * original_area before resizing back to the original dimensions.
    """
    h, w = image.shape[:2]
    retained_scale = float(np.sqrt(max(0.0, 1.0 - removed_area_fraction)))
    crop_w = max(1, int(round(w * retained_scale)))
    crop_h = max(1, int(round(h * retained_scale)))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def resize_restore(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize by scale, then restore to original size using deterministic interpolation."""
    h, w = image.shape[:2]
    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    down_or_up = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(down_or_up, (w, h), interpolation=cv2.INTER_LINEAR)


def rotate_center(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate around image center with fixed output dimensions and reflected borders."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def gaussian_blur_sigma(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with a valid odd kernel derived from sigma."""
    kernel = int(2 * round(3 * sigma) + 1)
    kernel = max(3, kernel if kernel % 2 == 1 else kernel + 1)
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def brightness_shift(image: np.ndarray, beta: int) -> np.ndarray:
    """Apply deterministic additive intensity shift with clipping."""
    shifted = image.astype(np.int16) + int(beta)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def contrast_scale(image: np.ndarray, alpha: float) -> np.ndarray:
    """Apply deterministic multiplicative contrast with clipping."""
    contrasted = image.astype(np.float32) * float(alpha)
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def patch_blur(image: np.ndarray, area_fraction: float) -> np.ndarray:
    """Replace a centered rectangular area with a blurred version of itself."""
    result = image.copy()
    h, w = image.shape[:2]
    side_scale = float(np.sqrt(area_fraction))
    patch_w = max(1, int(round(w * side_scale)))
    patch_h = max(1, int(round(h * side_scale)))
    x0 = max(0, (w - patch_w) // 2)
    y0 = max(0, (h - patch_h) // 2)
    region = result[y0 : y0 + patch_h, x0 : x0 + patch_w]
    sigma = max(3.0, min(patch_w, patch_h) / 18.0)
    kernel = int(2 * round(3 * sigma) + 1)
    kernel = max(3, kernel if kernel % 2 == 1 else kernel + 1)
    result[y0 : y0 + patch_h, x0 : x0 + patch_w] = cv2.GaussianBlur(
        region,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return result


def png_writer(path: Path, image: np.ndarray) -> None:
    """Write a PNG suspect image through the project's image helper."""
    save_image(path, image)


def build_attack_specs() -> list[dict[str, Any]]:
    """Return deterministic attack definitions and generation functions."""
    specs: list[dict[str, Any]] = [
        {
            "family": "control",
            "name": "control_exact",
            "parameter": "exact_byte_copy",
            "extension": ".png",
            "copy_exact": True,
        }
    ]

    for quality in [95, 75, 50, 25]:
        specs.append(
            {
                "family": "jpeg",
                "name": f"jpeg_q{quality}",
                "parameter": quality,
                "extension": ".jpg",
                "writer": lambda path, image, q=quality: write_jpeg_roundtrip(path, image, q),
            }
        )
    for percent in [10, 25, 50]:
        specs.append(
            {
                "family": "crop",
                "name": f"crop_{percent}",
                "parameter": f"{percent}%_area_removed",
                "extension": ".png",
                "transform": lambda image, p=percent: centered_crop_remove_area(image, p / 100.0),
                "writer": png_writer,
            }
        )
    for label, scale in [("050", 0.50), ("075", 0.75), ("150", 1.50)]:
        specs.append(
            {
                "family": "resize",
                "name": f"resize_{label}",
                "parameter": f"scale_{scale};interpolation_INTER_LINEAR",
                "extension": ".png",
                "transform": lambda image, s=scale: resize_restore(image, s),
                "writer": png_writer,
            }
        )
    for degrees in [2, 5, 10]:
        specs.append(
            {
                "family": "rotation",
                "name": f"rotate_{degrees}",
                "parameter": f"{degrees}_degrees;border_BORDER_REFLECT_101",
                "extension": ".png",
                "transform": lambda image, d=degrees: rotate_center(image, d),
                "writer": png_writer,
            }
        )
    for sigma in [1, 3, 5]:
        specs.append(
            {
                "family": "blur",
                "name": f"blur_sigma{sigma}",
                "parameter": f"sigma_{sigma}",
                "extension": ".png",
                "transform": lambda image, s=sigma: gaussian_blur_sigma(image, float(s)),
                "writer": png_writer,
            }
        )
    for name, beta in [("brightness_minus20", -20), ("brightness_plus20", 20), ("brightness_plus40", 40)]:
        specs.append(
            {
                "family": "brightness",
                "name": name,
                "parameter": beta,
                "extension": ".png",
                "transform": lambda image, b=beta: brightness_shift(image, b),
                "writer": png_writer,
            }
        )
    for label, alpha in [("075", 0.75), ("125", 1.25), ("150", 1.50)]:
        specs.append(
            {
                "family": "contrast",
                "name": f"contrast_{label}",
                "parameter": alpha,
                "extension": ".png",
                "transform": lambda image, a=alpha: contrast_scale(image, a),
                "writer": png_writer,
            }
        )
    for percent in [5, 10, 20]:
        specs.append(
            {
                "family": "patch",
                "name": f"patch_{percent:02d}",
                "parameter": f"{percent}%_center_area_blurred",
                "extension": ".png",
                "transform": lambda image, p=percent: patch_blur(image, p / 100.0),
                "writer": png_writer,
            }
        )
    return specs


def load_registered_records() -> list[dict[str, Any]]:
    """Load all available TrustHeritage registration records."""
    records = []
    for record_path in sorted(RECORDS_DIR.glob("*.json")):
        record = load_json(record_path)
        metadata = record.get("metadata", {})
        records.append(
            {
                "record_path": record_path,
                "record": record,
                "asset_id": metadata.get("asset_id", record_path.stem),
                "category": metadata.get("category", "NA"),
                "title": metadata.get("title", "NA"),
                "date_registered": metadata.get("date_registered", "NA"),
            }
        )
    return records


def map_dataset_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map H1-H5 labels to explicit H records or to sorted existing records."""
    by_asset_id = {str(item["asset_id"]).lower(): item for item in records}
    by_stem = {item["record_path"].stem.lower(): item for item in records}
    mapped = []
    used: set[Path] = set()

    for label in DATASET_LABELS:
        key = label.lower()
        item = by_asset_id.get(key) or by_stem.get(key)
        if item is not None:
            mapped.append({**item, "image_id": label, "mapping_method": "explicit_id"})
            used.add(item["record_path"])

    if len(mapped) == len(DATASET_LABELS):
        return mapped

    remaining = [item for item in records if item["record_path"] not in used]
    for label in DATASET_LABELS[len(mapped) :]:
        if not remaining:
            raise RuntimeError(
                f"Need {len(DATASET_LABELS)} registered assets, found {len(records)} in {RECORDS_DIR}."
            )
        item = remaining.pop(0)
        mapped.append({**item, "image_id": label, "mapping_method": "sorted_record_fallback"})
    return mapped


def resolve_record_path(record_path: Path, relative_path: str) -> Path:
    """Resolve paths stored in a TrustHeritage record."""
    base_dir = record_path.parent.parent.parent
    return base_dir / relative_path


def nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    """Read a nested dictionary path, returning None when unavailable."""
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def flatten_result(
    image_id: str,
    category: str,
    spec: dict[str, Any],
    suspect_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten the actual verify_image result into the requested CSV schema."""
    decision = nested_get(result, "interpretation", "decision_label")
    if decision is None:
        decision = nested_get(result, "scores", "label")

    row = {
        "image_id": image_id,
        "category": category,
        "attack_family": spec["family"],
        "attack_name": spec["name"],
        "attack_parameter": spec["parameter"],
        "suspect_path": suspect_path,
        "watermark_score": result.get("watermark_score"),
        "provenance_score": nested_get(result, "provenance", "provenance_score"),
        "exact_hash_match": nested_get(result, "provenance", "exact_match"),
        "forensic_score": nested_get(result, "forensic", "forensic_score"),
        "semantic_score": nested_get(result, "semantic", "semantic_score"),
        "cosine_similarity": nested_get(result, "semantic", "cosine_similarity"),
        "acs": nested_get(result, "scores", "acs"),
        "decision": decision,
        "ssim": nested_get(result, "forensic", "ssim"),
        "difference_score": nested_get(result, "forensic", "difference_score"),
        "histogram_score": nested_get(result, "forensic", "histogram_score"),
        "edge_score": nested_get(result, "forensic", "edge_score"),
        "suspicious_region_ratio": nested_get(result, "forensic", "suspicious_region_ratio"),
        "evidence_agreement": nested_get(result, "scores", "evidence_agreement"),
        "semantic_integrity_risk": nested_get(result, "heritage_risk", "semantic_integrity_risk"),
        "alignment_method": nested_get(result, "forensic", "alignment", "method"),
    }
    return {key: format_cell(value) for key, value in row.items()}


def coerce_float(value: Any) -> float | None:
    """Convert real numeric values for summary statistics."""
    if value in (None, "", "NA"):
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_attack_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Group results by attack family and parameter and summarize numeric metrics."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["attack_family"]), str(row["attack_parameter"]))].append(row)

    fields = ["attack_family", "attack_parameter"]
    for metric in NUMERIC_METRICS:
        fields.extend(
            [
                f"{metric}_count",
                f"{metric}_mean",
                f"{metric}_standard_deviation",
                f"{metric}_median",
                f"{metric}_minimum",
                f"{metric}_maximum",
            ]
        )

    summary_rows = []
    for (family, parameter), group_rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "attack_family": family,
            "attack_parameter": parameter,
        }
        for metric in NUMERIC_METRICS:
            values = [coerce_float(row.get(metric)) for row in group_rows]
            available = [value for value in values if value is not None]
            if not available:
                summary.update(
                    {
                        f"{metric}_count": 0,
                        f"{metric}_mean": "NA",
                        f"{metric}_standard_deviation": "NA",
                        f"{metric}_median": "NA",
                        f"{metric}_minimum": "NA",
                        f"{metric}_maximum": "NA",
                    }
                )
                continue

            summary.update(
                {
                    f"{metric}_count": len(available),
                    f"{metric}_mean": statistics.mean(available),
                    f"{metric}_standard_deviation": statistics.stdev(available)
                    if len(available) > 1
                    else 0.0,
                    f"{metric}_median": statistics.median(available),
                    f"{metric}_minimum": min(available),
                    f"{metric}_maximum": max(available),
                }
            )
        summary_rows.append(summary)
    return summary_rows, fields


def package_version(distribution_name: str) -> str:
    """Return an installed package version or NA."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "NA"


def write_manifest(timestamp: str) -> None:
    """Record environment and immutable configuration values for reproducibility."""
    torch_version = getattr(torch, "__version__", "NA") if torch is not None else "NA"
    rows = [
        {"experiment_timestamp": timestamp, "key": "python_version", "value": sys.version.replace("\n", " ")},
        {"experiment_timestamp": timestamp, "key": "opencv_version", "value": cv2.__version__},
        {"experiment_timestamp": timestamp, "key": "pytorch_version", "value": torch_version},
        {"experiment_timestamp": timestamp, "key": "openclip_version", "value": package_version("open-clip-torch")},
        {"experiment_timestamp": timestamp, "key": "operating_system", "value": platform.platform()},
        {"experiment_timestamp": timestamp, "key": "random_seed", "value": SEED},
        {"experiment_timestamp": timestamp, "key": "config.IMAGE_SIZE", "value": IMAGE_SIZE},
        {"experiment_timestamp": timestamp, "key": "config.WATERMARK_SIZE", "value": WATERMARK_SIZE},
        {"experiment_timestamp": timestamp, "key": "config.WATERMARK_STRENGTH", "value": WATERMARK_STRENGTH},
        {"experiment_timestamp": timestamp, "key": "config.DEFAULT_WEIGHTS", "value": DEFAULT_WEIGHTS},
        {"experiment_timestamp": timestamp, "key": "acs_weight_alpha", "value": DEFAULT_WEIGHTS.get("alpha")},
        {"experiment_timestamp": timestamp, "key": "acs_weight_beta", "value": DEFAULT_WEIGHTS.get("beta")},
        {"experiment_timestamp": timestamp, "key": "acs_weight_gamma", "value": DEFAULT_WEIGHTS.get("gamma")},
        {"experiment_timestamp": timestamp, "key": "acs_weight_delta", "value": DEFAULT_WEIGHTS.get("delta")},
        {"experiment_timestamp": timestamp, "key": "decision_threshold_authentic_min", "value": 0.85},
        {"experiment_timestamp": timestamp, "key": "decision_threshold_suspicious_min", "value": 0.60},
        {"experiment_timestamp": timestamp, "key": "decision_threshold_likely_tampered_below", "value": 0.60},
        {"experiment_timestamp": timestamp, "key": "openclip_model", "value": OPENCLIP_MODEL},
        {"experiment_timestamp": timestamp, "key": "openclip_pretrained_checkpoint", "value": OPENCLIP_PRETRAINED},
        {"experiment_timestamp": timestamp, "key": "records_dir", "value": RECORDS_DIR},
        {"experiment_timestamp": timestamp, "key": "archived_images_dir", "value": ARCHIVED_IMAGES_DIR},
        {"experiment_timestamp": timestamp, "key": "watermarked_images_dir", "value": WATERMARKED_IMAGES_DIR},
        {"experiment_timestamp": timestamp, "key": "suspect_uploads_dir", "value": SUSPECT_UPLOADS_DIR},
        {"experiment_timestamp": timestamp, "key": "outputs_dir", "value": OUTPUTS_DIR},
    ]
    write_csv(MANIFEST_CSV, rows, ["experiment_timestamp", "key", "value"])


def write_registration_summary(mapped_records: list[dict[str, Any]]) -> None:
    """Record the H1-H5 dataset mapping to existing TrustHeritage records."""
    rows = []
    for item in mapped_records:
        record = item["record"]
        rows.append(
            {
                "image_id": item["image_id"],
                "asset_id": item["asset_id"],
                "category": item["category"],
                "title": item["title"],
                "date_registered": item["date_registered"],
                "mapping_method": item["mapping_method"],
                "record_path": item["record_path"],
                "archived_image": resolve_record_path(item["record_path"], record["paths"]["archived_image"]),
                "watermarked_image": resolve_record_path(item["record_path"], record["paths"]["watermarked_image"]),
                "original_image_sha256": nested_get(record, "hashes", "original_image_sha256") or "NA",
                "watermarked_image_sha256": nested_get(record, "hashes", "watermarked_image_sha256") or "NA",
                "metadata_sha256": nested_get(record, "hashes", "metadata_sha256") or "NA",
            }
        )
    write_csv(
        REGISTRATION_SUMMARY_CSV,
        rows,
        [
            "image_id",
            "asset_id",
            "category",
            "title",
            "date_registered",
            "mapping_method",
            "record_path",
            "archived_image",
            "watermarked_image",
            "original_image_sha256",
            "watermarked_image_sha256",
            "metadata_sha256",
        ],
    )


def generate_suspect(
    watermarked_path: Path,
    output_path: Path,
    image: np.ndarray,
    spec: dict[str, Any],
) -> None:
    """Generate one suspect image for a registered watermarked asset."""
    if spec.get("copy_exact"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(watermarked_path, output_path)
        return

    transform: Callable[[np.ndarray], np.ndarray] = spec.get("transform", lambda source: source)
    writer: Callable[[Path, np.ndarray], None] = spec["writer"]
    writer(output_path, transform(image))


def main() -> None:
    set_deterministic_seeds()
    ensure_directories([EXPERIMENTS_DIR, RESULTS_DIR])
    timestamp = datetime.now(timezone.utc).isoformat()

    records = load_registered_records()
    mapped_records = map_dataset_records(records)
    attack_specs = build_attack_specs()

    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    total_attempted = 0
    total_successful = 0

    write_manifest(timestamp)
    write_registration_summary(mapped_records)

    for index, item in enumerate(mapped_records, start=1):
        image_id = item["image_id"]
        record_path = item["record_path"]
        record = item["record"]
        category = item["category"]
        watermarked_path = resolve_record_path(record_path, record["paths"]["watermarked_image"])

        print(f"[{index}/{len(mapped_records)}] Processing {image_id}")
        watermarked_image = read_image(watermarked_path)
        asset_successes = 0

        for spec in attack_specs:
            total_attempted += 1
            attack_name = spec["name"]
            suspect_path = (
                EXPERIMENTS_DIR
                / image_id
                / spec["family"]
                / f"{image_id}_{attack_name}{spec['extension']}"
            )

            try:
                print(f"Generating {attack_name}")
                generate_suspect(watermarked_path, suspect_path, watermarked_image, spec)
                print(f"Verifying {attack_name}")
                result = verify_image(record_path, suspect_path, use_semantics=True)
                result_rows.append(flatten_result(image_id, category, spec, suspect_path, result))
                asset_successes += 1
                total_successful += 1
            except Exception as exc:
                failure_rows.append(
                    {
                        "image_id": image_id,
                        "attack_name": attack_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                print(f"FAILED {image_id} {attack_name}: {type(exc).__name__}: {exc}")

        print(f"Completed {image_id}: {asset_successes}/{len(attack_specs)} experiments successful")

    write_csv(RESULTS_CSV, result_rows, RESULT_FIELDS)
    summary_rows, summary_fields = build_attack_summary(result_rows)
    write_csv(ATTACK_SUMMARY_CSV, summary_rows, summary_fields)
    write_csv(FAILURES_CSV, failure_rows, FAILURE_FIELDS)

    print(f"TOTAL ASSETS {len(mapped_records)}")
    print(f"TOTAL EXPERIMENTS ATTEMPTED {total_attempted}")
    print(f"TOTAL SUCCESSFUL {total_successful}")
    print(f"TOTAL FAILED {len(failure_rows)}")
    print(f"RESULT CSV PATH {RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()
