"""
Stage A.2 — OCR Processing

Converts raw PDF agreements (data/raw_pdfs/) into machine-readable text
(data/ocr_output/) using SURYA.

Not runnable from the public release: raw PDFs are not redistributed here
(see data/raw_pdfs/README.md) and the full OCR'd corpus is not published
either — only the two downstream CSVs are (data/labeled/train.csv and
test.csv, data/network/extracted_financial_entities.csv). This script is
provided so the OCR method is fully documented and runnable if you supply
your own PDF corpus (e.g. re-downloaded from the Iowa Secretary of State).

Validation checkpoint: 100-document manual reference sample, Character Error
Rate (CER) computed against that reference. SURYA CER = 8.7%.

Requires (not in requirements.txt by default — pull in only what you use):
    pip install pymupdf                       # PDF -> page images
    pip install surya-ocr==0.6.3               # SURYA (version pinned for the paper's CER benchmark)
    pip install evaluate jiwer                 # CER validation (HF `evaluate`'s cer metric)


Input:  data/raw_pdfs/*.pdf
Output: data/ocr_output/*.txt
"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common.text_utils import iter_with_progress

RAW_PDF_DIR = Path("data/raw_pdfs")
OCR_OUTPUT_DIR = Path("data/ocr_output")
RENDER_DPI = 300


def _pdf_to_images(pdf_path: Path):
    """Render each page of a PDF to a PIL Image at RENDER_DPI."""
    import fitz  # PyMuPDF
    from PIL import Image

    images = []
    doc = fitz.open(pdf_path)
    zoom = RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


@lru_cache(maxsize=1)
def _surya_predictors():
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    return recognition_predictor, detection_predictor


def run_surya(pdf_path: Path) -> str:
    """Extract text from every page of a PDF using SURYA, concatenated with
    blank-line page separators."""
    recognition_predictor, detection_predictor = _surya_predictors()
    images = _pdf_to_images(pdf_path)
    predictions = recognition_predictor(images, [None] * len(images), detection_predictor)
    page_texts = ["\n".join(line.text for line in page.text_lines) for page in predictions]
    return "\n\n".join(page_texts)


@lru_cache(maxsize=1)
def _cer_metric():
    from evaluate import load

    return load("cer")


def compute_cer(references: list[str], predictions: list[str]) -> float:
    """Corpus-level Character Error Rate via HuggingFace `evaluate`'s cer
    metric — the method used for the paper's validation checkpoint
    (SURYA CER = 8.7%)."""
    return _cer_metric().compute(predictions=predictions, references=references)


def run_cer_validation(samples_csv: Path):
    """Score SURYA against a manually-transcribed reference sample.

    Expects a CSV with columns `ocr_reference`, `surya_ocr` (one row per
    validated document) — the format of the 100-document sample used for
    the paper's validation checkpoint.
    """
    import pandas as pd

    df = pd.read_csv(samples_csv)
    ref = df["ocr_reference"].tolist()
    surya_cer = compute_cer(references=ref, predictions=df["surya_ocr"].tolist())
    print(f"SURYA CER: {surya_cer:.4f}")
    return surya_cer


def main():
    parser = argparse.ArgumentParser(description="Run OCR extraction on raw PDFs.")
    parser.add_argument("--input-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OCR_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Re-OCR files with existing output")
    parser.add_argument(
        "--validate-cer",
        type=Path,
        default=None,
        help="Skip OCR and instead score SURYA CER against this reference-sample CSV",
    )
    args = parser.parse_args()

    if args.validate_cer is not None:
        run_cer_validation(args.validate_cer)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    pdf_paths = [p for p in pdf_paths if "_noisy" not in p.stem]
    if not pdf_paths:
        print(f"No PDFs found in {args.input_dir}")
        return

    # Randomize order so concurrent workers on a shared queue (e.g. a Slurm
    # array on a cluster) don't all race through the same prefix range first.
    np.random.shuffle(pdf_paths)

    for pdf_path in iter_with_progress(pdf_paths, desc="OCR"):
        out_path = args.output_dir / (pdf_path.stem + ".txt")
        if out_path.exists() and not args.overwrite:
            continue
        try:
            text = run_surya(pdf_path)
        except Exception as exc:  # noqa: BLE001 — log and continue the batch
            print(f"[ERROR] surya failed on {pdf_path.name}: {exc}")
            continue
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
