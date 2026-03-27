# OCR Assignment Report Template

## 1. How to run

- Ubuntu system setup:
  - `bash setup_ubuntu.sh`
- Run pipeline:
  - `python main.py --stage all --image-dir data/raw/images --output-dir output`

## 2. Dataset

- Source:
- Number of images:
- Image types and quality:
- Why this dataset is useful for OCR testing:

## 3. Goal and workflow

- Goal:
- Workflow diagram:
  - Image -> OCR -> Post-Processing -> Error Analysis

## 4. OCR issues observed

- Common OCR errors:
- Examples before correction:
- Issues caused by image quality, font, spacing, etc.:

## 5. Error analysis

- Frequency table source:
  - `output/analysis/error_frequency_table.csv`
- Confusion matrix source:
  - `output/analysis/confusion_matrix.json`
- Top confusion pairs:
  - `output/analysis/top_confusion_pairs.csv`
- Interpretation:

## 6. Challenges and limitations

- What was difficult:
- What could improve results:
- Limitation: no external ground-truth labels (if applicable)

## 7. GenAI usage disclosure

- Tools used:
- What was generated with AI:
- What was manually implemented/verified:
