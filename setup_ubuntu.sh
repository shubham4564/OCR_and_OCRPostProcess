#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Tesseract version:"
tesseract --version

echo "Python package check:"
python -c "import pytesseract, cv2; print('Imports OK')"
