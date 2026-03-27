# OCR Assignment 3 Pipeline

This project implements an end-to-end OCR workflow:

1. Image OCR with two methods:
	- Custom method: preprocessed image + tuned Tesseract config
	- Baseline method: vanilla Tesseract on original image
2. OCR text post-processing (heuristic correction)
3. Error analysis (frequency table + confusion-style matrix)

## 1) Ubuntu setup

```bash
bash setup_ubuntu.sh
```

If you already have a virtual environment, install only Python dependencies:

```bash
pip install -r requirements.txt
```

## 2) Dataset placement

Put your small image dataset in:

```text
data/raw/images/
```

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`

## 3) Run the full pipeline

```bash
python main.py --stage all --image-dir data/raw/images --output-dir output
```

### Optional OCR controls

```bash
python main.py --stage ocr --psm 6 --oem 3 --lang eng
python main.py --stage ocr --no-preprocess
```

## 4) Run each stage separately

```bash
python main.py --stage ocr
python main.py --stage postprocess
python main.py --stage analyze
```

## 5) Output artifacts

Main outputs are written in `output/`:

- `output/ocr_manifest.csv`: OCR per-image metadata and paths
- `output/raw_text/*.txt`: custom method OCR text per image
- `output/raw_text_baseline/*.txt`: baseline OCR text per image
- `output/ocr_comparison_manifest.csv`: side-by-side comparison between custom and baseline OCR
- `output/postprocess_manifest.csv`: corrected text metadata and edit counts
- `output/corrected_text/*.txt`: post-processed text per image
- `output/edit_events.csv`: extracted correction events
- `output/analysis/error_frequency_table.csv`: required frequency table
- `output/analysis/top_confusion_pairs.csv`: top inferred confusion pairs
- `output/analysis/confusion_matrix.json`: confusion-style matrix data
- `output/analysis/events_per_image.csv`: event counts per image
- `output/analysis/summary.json`: analysis summary

## 6) Notes for report (PDF)

- Include your dataset source and sample characteristics.
- Include workflow diagram: Image -> OCR -> Post-Processing -> Error Analysis.
- Include a short section for common error types found.
- Include a comparison section between custom and baseline OCR using `output/ocr_comparison_manifest.csv`.
- Include the frequency table and confusion-style matrix from outputs.
- Mention that without ground-truth pairs, confusion statistics are inferred from post-processing edits.
- Include GenAI usage disclosure (if used).

## 7) Reproducibility

- Tested as script-based pipeline (no notebook dependency).
- Ubuntu setup script included for system + Python dependencies.
- Deterministic outputs given the same input images and OCR parameters.
