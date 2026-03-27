#!/usr/bin/env python3
"""
ocr_processor.py — Run OCR on a folder of images using two methods:
  1. Custom: preprocessed image + tuned Tesseract config
  2. Baseline: vanilla Tesseract on the original image

Usage:
    python src/ocr_processor.py --image-dir data/raw/images --output-dir output
    python src/ocr_processor.py --image-dir data/raw/images --output-dir output --no-preprocess
    python src/ocr_processor.py --help

Install dependencies:
    pip install pytesseract opencv-python Pillow
    # Ubuntu system package:
    # sudo apt-get install tesseract-ocr
"""
from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import pytesseract

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# I/O helpers (self-contained, no external module needed)
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _write_csv_rows(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

def _preprocess_image(image_path: Path) -> "cv2.typing.MatLike":
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    return processed


def _load_image(image_path: Path, preprocess: bool) -> "cv2.typing.MatLike":
    if preprocess:
        return _preprocess_image(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Main OCR runner
# ---------------------------------------------------------------------------

def run_ocr(
    image_dir: Path,
    raw_text_dir: Path,
    baseline_text_dir: Path,
    manifest_path: Path,
    comparison_manifest_path: Path,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    preprocess: bool = True,
) -> list[dict]:
    _ensure_dir(raw_text_dir)
    _ensure_dir(baseline_text_dir)

    image_paths = sorted(
        p for p in image_dir.glob("**/*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    rows: list[dict] = []
    comparison_rows: list[dict] = []
    tesseract_config = f"--oem {oem} --psm {psm}"

    for image_path in image_paths:
        image_custom = _load_image(image_path, preprocess=preprocess)
        extracted_text = pytesseract.image_to_string(
            image_custom,
            lang=lang,
            config=tesseract_config,
        )

        # Baseline uses vanilla Tesseract directly on the original image.
        baseline_text = pytesseract.image_to_string(str(image_path), lang=lang)

        raw_text_path = raw_text_dir / f"{image_path.stem}.txt"
        baseline_text_path = baseline_text_dir / f"{image_path.stem}.txt"
        _write_text(raw_text_path, extracted_text)
        _write_text(baseline_text_path, baseline_text)

        similarity_ratio = SequenceMatcher(a=baseline_text, b=extracted_text).ratio()

        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "raw_text_path": str(raw_text_path),
                "baseline_text_path": str(baseline_text_path),
                "char_count": len(extracted_text),
                "word_count": len(extracted_text.split()),
                "baseline_char_count": len(baseline_text),
                "baseline_word_count": len(baseline_text.split()),
                "method": "custom_preprocessed_tesseract",
                "baseline_method": "vanilla_tesseract",
                "ocr_lang": lang,
                "ocr_psm": psm,
                "ocr_oem": oem,
                "preprocess": preprocess,
            }
        )

        comparison_rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "custom_text_path": str(raw_text_path),
                "baseline_text_path": str(baseline_text_path),
                "custom_char_count": len(extracted_text),
                "baseline_char_count": len(baseline_text),
                "custom_word_count": len(extracted_text.split()),
                "baseline_word_count": len(baseline_text.split()),
                "text_similarity_ratio": f"{similarity_ratio:.4f}",
            }
        )

    fieldnames = [
        "image_name",
        "image_path",
        "raw_text_path",
        "baseline_text_path",
        "char_count",
        "word_count",
        "baseline_char_count",
        "baseline_word_count",
        "method",
        "baseline_method",
        "ocr_lang",
        "ocr_psm",
        "ocr_oem",
        "preprocess",
    ]
    _write_csv_rows(manifest_path, rows, fieldnames)
    _write_csv_rows(
        comparison_manifest_path,
        comparison_rows,
        [
            "image_name",
            "image_path",
            "custom_text_path",
            "baseline_text_path",
            "custom_char_count",
            "baseline_char_count",
            "custom_word_count",
            "baseline_word_count",
            "text_similarity_ratio",
        ],
    )
    return rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OCR on a folder of images (custom + baseline Tesseract)."
    )
    parser.add_argument("--image-dir", default="data/raw/images", help="Input image directory")
    parser.add_argument("--output-dir", default="output", help="Root output directory")
    parser.add_argument("--lang", default="eng", help="Tesseract language code (default: eng)")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OCR engine mode")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip image preprocessing (grayscale + denoise + threshold)",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    output_root = Path(args.output_dir)
    rows = run_ocr(
        image_dir=Path(args.image_dir),
        raw_text_dir=output_root / "raw_text",
        baseline_text_dir=output_root / "raw_text_baseline",
        manifest_path=output_root / "ocr_manifest.csv",
        comparison_manifest_path=output_root / "ocr_comparison_manifest.csv",
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        preprocess=not args.no_preprocess,
    )
    print(json.dumps({"num_images": len(rows), "output_dir": str(output_root)}, indent=2))
