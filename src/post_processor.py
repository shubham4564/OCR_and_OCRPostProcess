#!/usr/bin/env python3
"""
post_processor.py — Apply heuristic OCR corrections to raw OCR text files and
produce corrected text outputs plus an edit-events log.

Reads from the OCR manifest produced by ocr_processor.py.

Usage:
    python src/post_processor.py --output-dir output
    python src/post_processor.py --ocr-manifest path/to/ocr_manifest.csv --output-dir output
    python src/post_processor.py --help

Install dependencies:
    pip install pytesseract opencv-python Pillow
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# I/O helpers (self-contained, no external module needed)
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_csv_rows(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)

LETTER_CONFUSIONS = {
    "0": "O",
    "1": "I",
    "5": "S",
    "6": "G",
    "8": "B",
    "|": "I",
    "$": "S",
}


def _cleanup_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[\r\f\v]+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _replace_confusions_in_token(token: str) -> str:
    has_alpha = any(char.isalpha() for char in token)
    has_digit = any(char.isdigit() for char in token)
    if not has_alpha and not has_digit:
        return token

    if has_alpha and has_digit:
        return "".join(LETTER_CONFUSIONS.get(char, char) for char in token)

    alpha_ratio = sum(1 for char in token if char.isalpha()) / max(len(token), 1)
    if alpha_ratio >= 0.7:
        return "".join(LETTER_CONFUSIONS.get(char, char) for char in token)

    return token


def _apply_heuristics(text: str) -> str:
    cleaned = _cleanup_text(text)

    pieces = re.findall(r"\w+|\W+", cleaned)
    normalized: list[str] = []
    for piece in pieces:
        if piece.isalnum() or re.search(r"\w", piece):
            normalized.append(_replace_confusions_in_token(piece))
        else:
            normalized.append(piece)

    joined = "".join(normalized)
    joined = re.sub(r"([!?.,;:]){2,}", r"\1", joined)
    joined = re.sub(r"\s+([!?.,;:])", r"\1", joined)
    joined = re.sub(r"\n[ ]+", "\n", joined)
    return joined.strip() + "\n"


def _categorize_event(old: str, new: str) -> str:
    if old.isspace() or new.isspace():
        return "spacing"
    if (old and not old.isalnum()) or (new and not new.isalnum()):
        return "punctuation_noise"
    if not old:
        return "insertion"
    if not new:
        return "deletion"
    return "substitution"


def _diff_events(before: str, after: str) -> Iterable[dict]:
    matcher = SequenceMatcher(a=before, b=after)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        old_text = before[i1:i2]
        new_text = after[j1:j2]

        if tag == "replace":
            longest = max(len(old_text), len(new_text))
            for index in range(longest):
                old_char = old_text[index] if index < len(old_text) else ""
                new_char = new_text[index] if index < len(new_text) else ""
                yield {
                    "event_type": _categorize_event(old_char, new_char),
                    "source_char": old_char,
                    "target_char": new_char,
                    "before_fragment": old_text,
                    "after_fragment": new_text,
                }
        elif tag == "delete":
            for old_char in old_text:
                yield {
                    "event_type": _categorize_event(old_char, ""),
                    "source_char": old_char,
                    "target_char": "",
                    "before_fragment": old_text,
                    "after_fragment": "",
                }
        elif tag == "insert":
            for new_char in new_text:
                yield {
                    "event_type": _categorize_event("", new_char),
                    "source_char": "",
                    "target_char": new_char,
                    "before_fragment": "",
                    "after_fragment": new_text,
                }


# ---------------------------------------------------------------------------
# Main post-processing runner
# ---------------------------------------------------------------------------

def run_postprocess(
    ocr_manifest_path: Path,
    corrected_dir: Path,
    corrected_manifest_path: Path,
    edit_events_path: Path,
) -> tuple[list[dict], list[dict]]:
    _ensure_dir(corrected_dir)

    ocr_rows = _read_csv_rows(ocr_manifest_path)
    corrected_rows: list[dict] = []
    all_events: list[dict] = []

    for row in ocr_rows:
        raw_text_path = Path(row["raw_text_path"])
        raw_text = _read_text(raw_text_path)
        corrected_text = _apply_heuristics(raw_text)

        corrected_path = corrected_dir / raw_text_path.name
        _write_text(corrected_path, corrected_text)

        row_events = list(_diff_events(raw_text, corrected_text))
        for event in row_events:
            event["image_name"] = row["image_name"]
            event["raw_text_path"] = str(raw_text_path)
            event["corrected_text_path"] = str(corrected_path)
        all_events.extend(row_events)

        corrected_rows.append(
            {
                "image_name": row["image_name"],
                "raw_text_path": str(raw_text_path),
                "corrected_text_path": str(corrected_path),
                "raw_char_count": len(raw_text),
                "corrected_char_count": len(corrected_text),
                "num_edit_events": len(row_events),
            }
        )

    _write_csv_rows(
        corrected_manifest_path,
        corrected_rows,
        [
            "image_name",
            "raw_text_path",
            "corrected_text_path",
            "raw_char_count",
            "corrected_char_count",
            "num_edit_events",
        ],
    )

    _write_csv_rows(
        edit_events_path,
        all_events,
        [
            "image_name",
            "event_type",
            "source_char",
            "target_char",
            "before_fragment",
            "after_fragment",
            "raw_text_path",
            "corrected_text_path",
        ],
    )

    return corrected_rows, all_events


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply heuristic post-processing corrections to raw OCR text files."
    )
    parser.add_argument(
        "--ocr-manifest",
        default="output/ocr_manifest.csv",
        help="Path to OCR manifest CSV produced by ocr_processor.py",
    )
    parser.add_argument("--output-dir", default="output", help="Root output directory")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    output_root = Path(args.output_dir)
    ocr_manifest_path = Path(args.ocr_manifest)

    if not ocr_manifest_path.exists():
        raise SystemExit(
            f"OCR manifest not found: {ocr_manifest_path}\n"
            "Run ocr_processor.py first to generate it."
        )

    corrected_rows, events = run_postprocess(
        ocr_manifest_path=ocr_manifest_path,
        corrected_dir=output_root / "corrected_text",
        corrected_manifest_path=output_root / "postprocess_manifest.csv",
        edit_events_path=output_root / "edit_events.csv",
    )
    print(
        json.dumps(
            {
                "num_documents": len(corrected_rows),
                "num_edit_events": len(events),
                "output_dir": str(output_root),
            },
            indent=2,
        )
    )
