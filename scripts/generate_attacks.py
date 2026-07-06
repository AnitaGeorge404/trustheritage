"""Create simple suspect-image variants for TrustHeritage demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save {path}")


def crop_and_resize(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    y0, y1 = int(h * 0.08), int(h * 0.92)
    x0, x1 = int(w * 0.08), int(w * 0.92)
    cropped = image[y0:y1, x0:x1]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def contrast_modified(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.25, beta=8)


def jpeg_roundtrip(image: np.ndarray, quality: int = 35) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError("Could not encode JPEG attack variant.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise IOError("Could not decode JPEG attack variant.")
    return decoded


def occlude_center(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    h, w = image.shape[:2]
    x0, x1 = int(w * 0.38), int(w * 0.62)
    y0, y1 = int(h * 0.38), int(h * 0.62)
    cv2.rectangle(result, (x0, y0), (x1, y1), (24, 24, 24), thickness=-1)
    return result


def rotate_slightly(image: np.ndarray, degrees: float = 3.0) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo suspect image variants.")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, default=Path("data/suspect_uploads"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    stem = args.image.stem
    save(args.output / f"{stem}_cropped.png", crop_and_resize(image))
    save(args.output / f"{stem}_blurred.png", cv2.GaussianBlur(image, (9, 9), 0))
    save(args.output / f"{stem}_contrast.png", contrast_modified(image))
    save(args.output / f"{stem}_jpeg_q35.png", jpeg_roundtrip(image))
    save(args.output / f"{stem}_occluded.png", occlude_center(image))
    save(args.output / f"{stem}_rotated.png", rotate_slightly(image))
    print(f"Attack variants written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
