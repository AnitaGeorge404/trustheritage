"""Run batch verification for one archive record against a folder of suspects."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.pipeline import verify_image
from modules.utils import IMAGE_EXTENSIONS, load_json, utc_now_iso


ATTACK_MARKERS = {
    "jpeg": "jpeg_compression",
    "jpg": "jpeg_compression",
    "bright_plus": "brightness_increase",
    "bright_minus": "brightness_decrease",
    "noise": "gaussian_noise",
    "occlusion": "occlusion_patch",
    "rotate": "rotation",
    "rotated": "rotation",
    "sharpen": "sharpening",
    "desatur": "desaturation",
    "crop": "crop_resize",
    "cropped": "crop_resize",
    "blur": "gaussian_blur",
    "blurred": "gaussian_blur",
    "contrast": "contrast_change",
}


DETAIL_FIELDS = [
    "record_id",
    "suspect_filename",
    "attack_type",
    "watermark_score",
    "provenance_score",
    "forensic_score",
    "semantic_score",
    "acs",
    "label",
    "exact_hash_match",
    "cosine_similarity",
    "forensic_ssim",
    "forensic_difference_score",
    "forensic_histogram_score",
    "forensic_edge_score",
    "forensic_suspicious_region_ratio",
    "forensic_alignment_method",
    "visual_hash_similarity_hint",
    "timestamp",
]

SUMMARY_FIELDS = [
    "attack_type",
    "count",
    "mean_watermark_score",
    "mean_provenance_score",
    "mean_forensic_score",
    "mean_semantic_score",
    "mean_acs",
]


def infer_attack_type(path: Path) -> str:
    """Infer a simple attack label from a descriptive suspect filename."""
    name = path.stem.lower()
    for marker, attack_type in ATTACK_MARKERS.items():
        if marker in name:
            return attack_type
    return "unknown"


def iter_images(folder: Path) -> list[Path]:
    """Return candidate image files from a suspect folder."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def mean_optional(values: list[float | None]) -> float | None:
    """Mean of available numeric values, or None if all are missing."""
    available = [value for value in values if value is not None]
    if not available:
        return None
    return float(mean(available))


def format_optional(value: float | None) -> float | str:
    """Keep CSV cells readable when an optional metric is unavailable."""
    return "" if value is None else float(value)


def row_from_result(record_id: str, suspect_path: Path, result: dict, timestamp: str) -> dict:
    """Flatten a verification result into one CSV row."""
    scores = result["scores"]
    forensic = result["forensic"]
    provenance = result["provenance"]
    semantic = result["semantic"]
    alignment = forensic.get("alignment") or {}
    return {
        "record_id": record_id,
        "suspect_filename": suspect_path.name,
        "attack_type": infer_attack_type(suspect_path),
        "watermark_score": scores["watermark_score"],
        "provenance_score": scores["provenance_score"],
        "forensic_score": scores["forensic_score"],
        "semantic_score": format_optional(scores["semantic_score"]),
        "acs": scores["acs"],
        "label": scores["label"],
        "exact_hash_match": provenance["exact_match"],
        "cosine_similarity": format_optional(semantic.get("cosine_similarity")),
        "forensic_ssim": forensic.get("ssim"),
        "forensic_difference_score": forensic.get("difference_score"),
        "forensic_histogram_score": forensic.get("histogram_score"),
        "forensic_edge_score": forensic.get("edge_score"),
        "forensic_suspicious_region_ratio": forensic.get("suspicious_region_ratio"),
        "forensic_alignment_method": alignment.get("method", "none"),
        "visual_hash_similarity_hint": format_optional(provenance.get("visual_hash_similarity")),
        "timestamp": timestamp,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write dictionaries to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> list[dict]:
    """Create grouped mean score rows by inferred attack type."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["attack_type"]].append(row)

    summary_rows = []
    for attack_type, attack_rows in sorted(groups.items()):
        summary_rows.append(
            {
                "attack_type": attack_type,
                "count": len(attack_rows),
                "mean_watermark_score": mean_optional([r["watermark_score"] for r in attack_rows]),
                "mean_provenance_score": mean_optional([r["provenance_score"] for r in attack_rows]),
                "mean_forensic_score": mean_optional([r["forensic_score"] for r in attack_rows]),
                "mean_semantic_score": mean_optional(
                    [r["semantic_score"] if r["semantic_score"] != "" else None for r in attack_rows]
                ),
                "mean_acs": mean_optional([r["acs"] for r in attack_rows]),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate suspect images for one TrustHeritage record.")
    parser.add_argument("record", type=Path, help="Path to a TrustHeritage record JSON")
    parser.add_argument("suspects", type=Path, help="Folder containing suspect images")
    parser.add_argument("--output", type=Path, default=Path("data/outputs/batch_evaluation.csv"))
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--no-semantics", action="store_true", help="Skip OpenCLIP semantic similarity")
    args = parser.parse_args()

    if not args.record.exists():
        raise FileNotFoundError(f"Record not found: {args.record}")
    if not args.suspects.is_dir():
        raise NotADirectoryError(f"Suspect folder not found: {args.suspects}")

    record = load_json(args.record)
    record_id = record.get("metadata", {}).get("asset_id", args.record.stem)
    timestamp = utc_now_iso()
    rows = []

    for suspect_path in iter_images(args.suspects):
        result = verify_image(args.record, suspect_path, use_semantics=not args.no_semantics)
        rows.append(row_from_result(record_id, suspect_path, result, timestamp))

    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_name(f"{args.output.stem}_summary.csv")

    write_csv(args.output, rows, DETAIL_FIELDS)
    write_csv(summary_output, build_summary(rows), SUMMARY_FIELDS)
    print(f"Wrote {len(rows)} detailed rows to {args.output.resolve()}")
    print(f"Wrote grouped summary to {summary_output.resolve()}")


if __name__ == "__main__":
    main()
